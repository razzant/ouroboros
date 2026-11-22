// Extensions › MCP servers and Settings › Models (docs/DESIGN.md "Settings"), against a fake settings server that answers
// like the real one (`changes` with a token per key, `_meta.effects`, `_meta.tokens`, a 409 for a stale list).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import '../modules/settings.js';
import { SAFETY_MODEL_KEYS } from '../modules/consciousness_api.js';
import { applyMcpSettings, collectMcpSettings, refreshStatus } from '../modules/mcp_settings.js';
import { MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_KEY, fallbackBody, fallbackEntries, roleBody, routeWords,
    writeModelKeys } from '../modules/model_roles.js';
import { modelCatalog } from '../modules/model_catalog_store.js';
import { MODEL_PROCESSING_PREFERENCES_KEY } from '../modules/route_editor_primitives.js';
import { forgetSettingRows, outcomeLine, readSettings } from '../modules/settings_controls.js';
import { localModelStatusWords } from '../modules/settings_local_model.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { HAND_WRITTEN, ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 12) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };

const SERVERS = [
    { id: 'slack', name: 'Slack', enabled: true, transport: 'streamable_http', url: 'https://mcp.example.com/slack', auth_header: 'Authorization', auth_token: 'sk-...', auth_configured: true },
    { id: 'files', name: 'Files', enabled: false, transport: 'stdio', command: 'npx', args: ['-y', 'files'] },
];
const VALUES = {
    MCP_ENABLED: true, MCP_TOOL_TIMEOUT_SEC: 60, OUROBOROS_TOOL_TIMEOUT_SEC: 600, MCP_SERVERS: SERVERS,
    OUROBOROS_MODEL: 'claudexor::codex=gpt-main', OUROBOROS_MODEL_LIGHT: 'openai::gpt-light', OUROBOROS_MODEL_SAFETY: '',
    OUROBOROS_MODEL_VISION: '', OUROBOROS_MODEL_CONSCIOUSNESS: '', OUROBOROS_MODEL_FALLBACKS: 'anthropic::claude-a, moonshotai/kimi, openai::gpt-b',
    USE_LOCAL_MAIN: false, USE_LOCAL_LIGHT: false, USE_LOCAL_CONSCIOUSNESS: false, USE_LOCAL_FALLBACK: false,
    [MODEL_ACCOUNTS_KEY]: { main: 'studio-01', vision: 'studio-02' }, [MODEL_CONTEXT_KEY]: { light: 64000, fallback: [0, 128000, 0] },
    [MODEL_PROCESSING_PREFERENCES_KEY]: { consciousness: 'economy' },
    OUROBOROS_WEBSEARCH_BACKEND: 'auto', OUROBOROS_WEBSEARCH_MODEL: '', OUROBOROS_CONTEXT_MODE: 'max', OUROBOROS_IMAGE_INPUT_MODE: 'auto',
    OUROBOROS_PROCESSING_PREFERENCE: '', OUROBOROS_PROMPT_CACHE_TTL: '1h', OUROBOROS_MODEL_MAX_CONCURRENCY: 3,
    OPENAI_API_KEY: 'sk-...', ANTHROPIC_API_KEY: 'sk-...', OPENROUTER_API_KEY: 'sk-...',
};
const EFFECT_OF = (key) => (key.startsWith('LOCAL_MODEL_') ? 'local_model' : key.startsWith('MCP_') ? 'now'
    : key === MODEL_CONTEXT_KEY ? 'next_task' : /MODEL|USE_LOCAL|WEBSEARCH|PROCESSING/.test(key) ? 'next_call' : 'next_task');

/** The settings server: one document, a token per key that moves with every write, both write endpoints logged. */
function fakeServer(routes, values = {}) {
    const server = { posts: [], checker: [], reads: 0, version: 0, answers: [], values: structuredClone({ ...VALUES, ...values }) };
    const token = (key) => `${key}@${server.tokens[key] || 0}`;
    server.tokens = {};
    server.document = () => ({ ...structuredClone(server.values), _meta: {
        effects: Object.fromEntries(Object.keys(server.values).map((key) => [key, EFFECT_OF(key)])),
        tokens: Object.fromEntries(Object.keys(server.values).map((key) => [key, token(key)])),
        setup_contract: { scopedModelCatalogs: true, modelSlots: contract.modelSlots } } });
    const write = (settings, precondition = {}) => {
        // A connection's revision is compared with that connection as it is now (owner_settings.connection_revision).
        const stale = Object.entries(precondition || {}).find(([key, held]) => key !== 'ultrafast' && held !== (key === 'connection' ? server.revision : token(key)));
        if (stale) return response({ saved: false, code: 'settings_changed_elsewhere', error: `${stale[0]} changed` }, 409);
        const changes = {};
        for (const [key, value] of Object.entries(settings)) {
            server.tokens[key] = (server.tokens[key] || 0) + 1;
            const structure = typeof value === 'object' && value !== null;
            changes[key] = structure ? { withheld: 'structure', token: token(key) } : { old: server.values[key], new: value, token: token(key) };
            server.values[key] = structuredClone(value);
        }
        return response({ ok: true, status: 'saved', changes });
    };
    routes.set('/api/settings', async (_url, init = {}) => {
        if (init.method !== 'POST') { server.reads += 1; return response(server.document()); }
        const { if_unchanged: precondition, ...settings } = JSON.parse(init.body);
        server.posts.push({ settings, precondition });
        const answer = server.answers.shift();
        return answer ? answer(settings) : write(settings, precondition);
    });
    routes.set('/api/owner/safety-model', async (_url, init = {}) => {
        const body = JSON.parse(init.body);
        server.checker.push({ settings: body.settings, precondition: body.if_unchanged });
        return write(body.settings, body.if_unchanged);
    });
    routes.set('/api/mcp/status', async () => response({ sdk_available: true, enabled: server.values.MCP_ENABLED, weights: { map_tokens: 420, schema_tokens_from_task_start: 0 },
        servers: [{ id: 'slack', tool_count: 3, listed_at: new Date().toISOString(), schema_tokens: 9200, map_tokens: 210, tools: [], sign_in: { state: 'not_signed_in', reason: 'expired' } },
            { id: 'files', tool_count: 0, listed_at: '', schema_tokens: null, tools: [] }] }));
    routes.set('/api/model-catalog', async () => response({ items: [], errors: [], model_sources: [] }));
    return server;
}

async function open(t, path, { values = {}, meta = {}, onScreen = null } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const server = fakeServer(env.routes, values);
    const document = server.document;
    server.document = () => { const read = document(); Object.assign(read._meta, meta); return read; };
    applyMcpSettings({ MCP_SERVERS: [] });
    const opened = [];
    const asked = [];
    const ctx = { readSettings, openSettings: (to, options) => { opened.push(to); return true; }, state: { activePage: 'settings', settingsPath: path },
        ask: async (question) => { asked.push(question); return true; } };
    if (onScreen) ctx.onScreen = onScreen;  // the section's fact (section_levels.js): is this page the one on screen
    const container = new NodeStub();
    const handle = await settingsPageFor(path).mount(container, { path, row: '', ctx });
    await settle();
    t.after(() => handle?.unmount?.());
    return { ...env, server, ctx, container, opened, asked, handle,
        press: async (target) => { await container.fire('click', { target }); await settle(); },
        change: async (target) => { await container.fire('change', { target }); await settle(); } };
}

const head = (container, id) => container.querySelectorAll('[data-entry-id]').find((node) => node.dataset.entryId === id);
const inside = (node, selector) => node.querySelectorAll(selector)[0] || null;
const serversPosted = (post) => post.settings.MCP_SERVERS.map((entry) => [entry.id, entry.enabled]);

test('the MCP servers list: the client switch is a plain row, each server is a head with its state, and the fold holds the servers\' wait', async (t) => {
    const page = await open(t, 'tools');
    // Skills is the page beside it in Extensions, so its door is gone; the global tool timeout is one row of This
    // computer › Advanced, the MCP wait stays with the servers.
    assert.deepEqual(rowsOf('tools', { fold: false }).map((row) => row.id), ['MCP_ENABLED', 'tools:door:circles']);
    assert.deepEqual(rowsOf('tools', { fold: true }).map((row) => row.id), ['MCP_TOOL_TIMEOUT_SEC', 'tools:import', 'tools:refresh']);
    assert.deepEqual(rowById('OUROBOROS_TOOL_TIMEOUT_SEC').page, 'computer/advanced');
    assert.equal(inside(page.container.querySelector('[data-setting-row="MCP_SERVERS"]'), '.settings-card-title').textContent, 'Servers');
    const client = rowById('MCP_ENABLED');
    assert.deepEqual([client.control.kind, client.writer, client.codec, client.keys], ['switch', 'generic', 'bool', ['MCP_ENABLED']]);
    const row = page.container.querySelector('[data-setting-row="MCP_ENABLED"]');
    assert.equal(row.dataset.kind, 'switch');
    const input = inside(row, '[data-setting-control]');
    assert.equal(input.checked, true, 'the row shows the stored value');
    // The sizes are said as the status gives them; no sentence advises switching a server off.
    const lead = () => page.container.querySelector('[data-mcp-global]').textContent;
    assert.match(lead(), /^2 servers configured, 3 tools discovered\. Connections map ≈420 tokens in every Low\/Max request; call schemas from task start ≈0 tokens\.$/);
    assert.doesNotMatch(readFileSync(new URL('../modules/mcp_settings.js', import.meta.url), 'utf8'), /fewer tools|shorter requests|to save tokens|adds .* to every request/i);
    input.checked = false;
    await page.change(input);
    assert.deepEqual(page.server.posts.map((post) => post.settings), [{ MCP_ENABLED: false }], 'the switch writes its one key, never the servers');
    assert.equal(inside(row, '[data-setting-result]').textContent, '', 'a switch that acts now is its own confirmation');
    // Heads: the name, the status's own words, Sign in only where the status asks, the switch.
    assert.deepEqual(page.container.querySelectorAll('[data-entry-id]').map((node) => node.dataset.entryId), ['slack', 'files']);
    const slack = head(page.container, 'slack');
    assert.match(slack.textContent, /Slack/);
    assert.match(slack.textContent, /3 tools · listed/);
    assert.match(slack.textContent, /HTTP/);
    assert.equal(inside(slack, '[data-entry-sign-in]').textContent, 'Sign in');
    assert.equal(inside(head(page.container, 'files'), '[data-entry-sign-in]'), null);
    assert.equal(inside(slack, '[data-entry-switch]').hasAttribute('checked'), true);
    assert.equal(lead(), 'All MCP servers are off. Their settings are kept.', 'the client\'s switch changes what the status says');
    // Pressing a head opens that server's page; Add opens the unsaved one.
    await page.press(inside(slack, '[data-entry-open]'));
    await page.press(page.container.querySelector('[data-tools-add]'));
    assert.deepEqual(page.opened, ['tools/slack', 'tools/new']);
});

test('a server\'s switch writes one entry from a fresh read with its token, and a stale list is refused and shown again', async (t) => {
    const page = await open(t, 'tools');
    const reads = page.server.reads;
    const input = inside(head(page.container, 'files'), '[data-entry-switch]');
    input.checked = true;
    await page.change(input);
    assert.equal(page.server.posts.length, 1);
    const [post] = page.server.posts;
    assert.deepEqual(Object.keys(post.settings), ['MCP_SERVERS'], 'one key');
    assert.deepEqual(serversPosted(post), [['slack', true], ['files', true]], 'one entry changed, the other as the read gave it');
    assert.deepEqual(post.settings.MCP_SERVERS[0], SERVERS[0], 'the untouched entry is the fresh one, mask and all');
    assert.deepEqual(post.precondition, { MCP_SERVERS: 'MCP_SERVERS@0' }, 'the token of the read it changed');
    assert.ok(page.server.reads > reads, 'a fresh read came first');
    assert.equal(page.container.querySelector('[data-entry-result]').textContent, '', 'quiet when it lands');
    assert.equal(inside(head(page.container, 'files'), '[data-entry-switch]').hasAttribute('checked'), true, 'the head shows what the server holds');

    // Another device changed the list between the read and the write: refused, said, never sent again.
    page.server.answers.push(() => {
        page.server.values.MCP_SERVERS = [{ ...SERVERS[0], enabled: false }, { ...SERVERS[1], enabled: true }];
        return response({ saved: false, code: 'settings_changed_elsewhere', error: 'MCP_SERVERS changed' }, 409);
    });
    const slack = inside(head(page.container, 'slack'), '[data-entry-switch]');
    slack.checked = false;
    await page.change(slack);
    assert.equal(page.server.posts.length, 2, 'one attempt, no silent retry');
    assert.equal(page.container.querySelector('[data-entry-result]').textContent, 'Changed on another device. This is the current value.');
    assert.equal(inside(head(page.container, 'slack'), '[data-entry-switch]').hasAttribute('checked'), false, 'the list is drawn from the fresh read');
});

test('a server with an empty id writes by its slug or name without touching the other entry', async (t) => {
    const servers = [{ id: '', name: 'first', enabled: true, transport: 'stdio', command: 'python' },
        { id: '', slug: 'second', name: 'alias', enabled: true, transport: 'stdio', command: 'python' }];
    const page = await open(t, 'tools', { values: { MCP_SERVERS: servers } });
    for (const [id, index] of [['second', 1], ['first', 0]]) {
        const input = inside(head(page.container, id), '[data-entry-switch]');
        input.checked = false;
        await page.change(input);
        servers[index].enabled = false;
        assert.deepEqual(page.server.posts.at(-1).settings.MCP_SERVERS, servers);
    }
});

test('a server\'s page saves its one entry and keeps the switch out of the form; Remove asks first', async (t) => {
    const page = await open(t, 'tools/files');
    const card = page.container.querySelector('[data-mcp-card]');
    assert.equal(card.dataset.entryId, 'files');
    for (const hook of ['[data-editor-save]', '[data-mcp-test]', '[data-editor-remove]', '[data-editor-result]', '[data-entry-switch]']) assert.ok(inside(card, hook), hook);
    const command = card.querySelectorAll('[data-mcp-field]').find((node) => node.dataset.mcpField === 'command');
    command.value = 'node';
    await command.fire('input');
    // The list changed its switch meanwhile on the server: Save keeps the stored switch, not the form's.
    page.server.values.MCP_SERVERS[1].enabled = true;
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    const [post] = page.server.posts;
    assert.deepEqual(serversPosted(post), [['slack', true], ['files', true]]);
    assert.equal(post.settings.MCP_SERVERS[1].command, 'node');
    assert.deepEqual(post.settings.MCP_SERVERS[0], SERVERS[0]);
    assert.deepEqual(post.precondition, { MCP_SERVERS: 'MCP_SERVERS@0' });
    assert.equal(page.container.querySelector('[data-editor-result]').textContent, 'Saved');
    // Remove asks, then writes the list without the entry and returns to the list.
    await page.container.querySelector('[data-editor-remove]').fire('click', { target: page.container.querySelector('[data-editor-remove]') });
    await settle();
    assert.equal(page.asked.length, 1);
    assert.match(page.asked[0].title, /Remove Files\?/);
    assert.deepEqual(page.server.posts.at(-1).settings.MCP_SERVERS.map((entry) => entry.id), ['slack']);
    assert.deepEqual(page.opened, ['tools']);
});

test('a server\'s page is drawn again only while it is the page on screen: a read that lands after it was left or covered draws nothing', async (t) => {
    let top = 'tools/files';  // the page Extensions shows on top, '' while another place is shown
    const page = await open(t, 'tools/files', { onScreen: (path) => top === path });
    const card = () => page.container.querySelector('[data-mcp-card]');
    const name = () => card().querySelectorAll('[data-mcp-field]').find((node) => node.dataset.mcpField === 'name').getAttribute('value');
    assert.equal(name(), 'Files');
    const real = page.routes.get('/api/settings');
    const shown = async () => { page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/files', place: 'skills' } })); await settle(); };
    const late = async (away) => {
        const held = deferred();
        page.routes.set('/api/settings', async (url, init = {}) => { if (!init.method) await held.promise; return real(url, init); });
        await shown();  // the page's read starts while it is on screen...
        top = away;     // ...and the owner leaves the place, or a level covers the page, before it lands
        held.resolve();
        await settle();
        page.routes.set('/api/settings', real);
    };
    // Renamed elsewhere; the read lands after the owner left: the form stands as it was drawn.
    page.server.values.MCP_SERVERS[1].name = 'Files (shared)';
    const drawn = card();
    await late('');
    assert.equal(card(), drawn, 'a read that lands after the owner left draws nothing');
    assert.equal(name(), 'Files');
    // Back: the shell says "shown" for this path, and the form is read and drawn.
    top = 'tools/files';
    await shown();
    assert.equal(name(), 'Files (shared)');
    // Covered by another level: the same.
    page.server.values.MCP_SERVERS[1].name = 'Files (team)';
    const again = card();
    await late('tools/files/other');
    assert.equal(card(), again, 'a read that lands under another level draws nothing');
    top = 'tools/files';
    await shown();
    assert.equal(name(), 'Files (team)');
});

test('an open MCP editor keeps its live client ID control across a neighbouring edit and status refreshes', async t => {
    const path = 'tools/slack';
    let top = path;
    const page = await open(t, path, { onScreen: (candidate) => candidate === top });
    const markShown = async () => {
        page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path, place: 'skills' } }));
        await settle();
    };
    const list = new NodeStub();
    const handle = settingsPageFor('tools').mount(list, { path: 'tools', ctx: page.ctx });
    t.after(() => handle.unmount());
    await settle();
    const listState = () => head(list, 'slack').querySelector('[data-mcp-state]').textContent;
    const listSignIn = () => head(list, 'slack').querySelector('[data-entry-sign-in]');
    const lead = () => list.querySelector('[data-mcp-global]').textContent;
    const clientId = (container = page.container) => container.querySelector('[data-mcp-sign-in-client-id]');
    const card = page.container.querySelector('[data-mcp-card]');
    const input = clientId();
    input.closest('details').open = true;
    input.focus();
    input.value = 'fixture-app-id';
    const retained = () => {
        assert.ok(page.container.querySelector('[data-mcp-card]') === card, 'keep the mounted form');
        assert.ok(clientId() === input, 'keep the actual input, not an equivalent replacement');
        assert.ok(page.document.activeElement === input, 'the same input still has focus');
        assert.equal(input.value, 'fixture-app-id');
        assert.equal(input.closest('details').open, true);
    };

    assert.match(listState(), /^3 tools · listed/);
    assert.ok(listSignIn());
    await markShown();  // an ordinary equivalent settings/status read while actively typing
    retained();
    assert.equal(page.handle.unsaved(), false, 'not a settings edit');

    const reads = page.server.reads;
    page.server.values.MCP_SERVERS[1].name = 'Shared files';
    page.server.tokens.MCP_SERVERS = 1;
    await markShown();
    assert.ok(page.server.reads > reads);
    assert.match(head(list, 'files').textContent, /Shared files/, 'the other server really was refreshed');
    retained();
    assert.equal(page.handle.unsaved(), false, 'the ephemeral value is still not a settings edit');

    top = '';
    page.routes.set('/api/mcp/status', () => response({}, 503));
    await refreshStatus();
    assert.equal(lead(), 'MCP status unavailable');
    assert.equal(listState(), 'MCP status unavailable');
    assert.equal(listSignIn(), null);
    top = path;
    await markShown();
    retained();

    const status = (signIn) => response({ enabled: true, sdk_available: true, servers: [{ id: 'slack', enabled: true,
        tool_count: 5, listed_at: '2026-11-18T00:00:00Z', sign_in: signIn }] });
    page.routes.set('/api/mcp/status', () => status({ state: 'not_signed_in', reason: 'expired' }));
    await refreshStatus();
    assert.match(lead(), /^1 server configured, 5 tools discovered/);
    assert.match(listState(), /^5 tools · listed/);
    assert.ok(listSignIn());
    retained();

    // A detached form may still be subscribed while its mount's settings read settles.
    let connected = false;
    Object.defineProperty(page.container, 'isConnected', { get: () => connected });
    page.routes.set('/api/mcp/status', () => status({ state: 'signed_in' }));
    await refreshStatus();
    assert.equal(listSignIn(), null);
    retained();
    connected = true;
    await markShown();
    assert.equal(clientId(), null, 'a changed sign-in shape is drawn when the editor is shown again');

    // Changing this server's saved address starts a new form, even with the same sign-in status.
    page.routes.set('/api/mcp/status', () => status({ state: 'not_signed_in', reason: 'expired' }));
    await refreshStatus();
    const beforeAddressChange = clientId();
    beforeAddressChange.value = 'different-fixture-id';
    page.server.values.MCP_SERVERS[0].url = 'https://services.example.test/slack';
    page.server.tokens.MCP_SERVERS += 1;
    await markShown();
    assert.ok(clientId() !== beforeAddressChange);
    assert.equal(clientId().value || '', '');
    assert.equal(page.container.querySelector('[data-mcp-field="url"]').getAttribute('value'), 'https://services.example.test/slack');

    await typeInto(page.container.querySelector('[data-mcp-field="url"]'), 'https://example.test/edit');
    const discarded = clientId();
    discarded.value = 'discarded-fixture-id';
    await page.container.querySelector('[data-editor-discard]').fire('click');
    assert.ok(clientId() !== discarded, 'an explicit Discard rebuilds the form');
    assert.equal(clientId().value || '', '');

    clientId().value = 'closed-fixture-id';
    page.handle.unmount();
    const fresh = new NodeStub();
    const reopened = await settingsPageFor(path).mount(fresh, { path, ctx: page.ctx });
    t.after(() => reopened.unmount());
    assert.equal(clientId(fresh).value || '', '', 'a new mount does not inherit the old form input');
    assert.deepEqual(page.server.posts, []);
    assert.ok(!page.calls.includes('/api/mcp/sign-in'));
});

test('removing a neighbouring MCP row keeps the open server’s public client ID and focus', async t => {
    const path = 'tools/notes';
    const [archive, notes] = ['archive', 'notes'].map(id => ({ id, name: id, enabled: true,
        transport: 'streamable_http', url: `https://${id}.example.test/mcp` }));
    const page = await open(t, path, { values: { MCP_SERVERS: [archive, notes] } });
    page.routes.set('/api/mcp/status', () => response({ sdk_available: true, enabled: true, servers: [
        { id: 'notes', enabled: true, tool_count: 0, sign_in: { state: 'not_signed_in', reason: 'not_signed_in' } },
    ] }));
    await refreshStatus();
    const card = () => page.container.querySelector('[data-mcp-card]');
    const input = () => card().querySelector('[data-mcp-sign-in-client-id]');
    assert.equal(card().dataset.mcpIndex, '1');
    input().closest('details').open = true;
    input().focus();
    input().value = 'fixture-notes-client';

    page.server.values.MCP_SERVERS.shift();
    page.server.tokens.MCP_SERVERS = 1;
    page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path, place: 'skills' } }));
    await settle();
    assert.equal(card().dataset.mcpIndex, '0', 'the editor is rendered and bound at its current slot');
    assert.equal(card().dataset.entryId, 'notes');
    assert.equal(input().value, 'fixture-notes-client');
    assert.equal(input().closest('details').open, true);
    assert.ok(page.document.activeElement === input(), 'focus is in the current card');
    assert.equal(page.handle.unsaved(), false, 'the client ID is not a settings draft');
    assert.deepEqual(page.server.posts, []);
    assert.ok(page.calls.every(url => ['/api/settings', '/api/mcp/status'].includes(url)));
});

test('a new server is saved as one more entry with an ID, and a taken ID is refused before anything is sent', async (t) => {
    const page = await open(t, 'tools/new');
    const card = page.container.querySelector('[data-mcp-card]');
    const field = (name) => card.querySelectorAll('[data-mcp-field]').find((node) => node.dataset.mcpField === name);
    assert.equal(inside(card, '[data-entry-switch]'), null, 'its on/off is a field of the form');
    const type = async (name, value) => { const input = field(name); input.value = value; await input.fire('input'); };
    await type('name', 'Slack');
    await type('url', 'https://example.com/mcp');
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    assert.equal(page.server.posts.length, 0);
    assert.match(page.container.querySelector('[data-editor-result]').textContent, /Another server already uses the ID slack\./);
    await type('name', 'Git Hub');
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    const [post] = page.server.posts;
    assert.deepEqual(post.settings.MCP_SERVERS.map((entry) => [entry.id, entry.enabled]), [['slack', true], ['files', false], ['git_hub', false]]);
    assert.deepEqual(page.opened, ['tools'], 'the list shows the new head');
    assert.equal(collectMcpSettings().MCP_SERVERS.length, 3);
});

for (const action of ['save', 'add', 'remove']) test(`late MCP ${action} leaves the current entry and its focus alone`, async t => {
    let path = action === 'add' ? 'tools/new' : 'tools/files';
    const page = await open(t, path, { onScreen: candidate => candidate === path });
    if (action === 'save') {
        const name = page.container.querySelector('[data-mcp-field="name"]');
        name.value = 'Local files';
        await name.fire('input');
    }
    if (action === 'add') {
        for (const [field, value] of [['name', 'Inbox'], ['url', 'https://example.test/inbox']]) {
            const input = page.container.querySelector(`[data-mcp-field="${field}"]`);
            input.value = value;
            await input.fire('input');
        }
    }
    const write = deferred(), endpoint = page.routes.get('/api/settings');
    page.routes.set('/api/settings', async (url, init) => {
        if (init?.method === 'POST') await write.promise;
        return endpoint(url, init);
    });
    const button = page.container.querySelector(action === 'remove' ? '[data-editor-remove]' : '[data-editor-save]');
    const pending = button.fire('click', { target: button });
    await settle();
    path = 'tools/slack';
    const next = new NodeStub();
    const handle = await settingsPageFor(path).mount(next, { path, ctx: page.ctx });
    t.after(() => handle.unmount());
    const input = next.querySelector('[data-mcp-field="name"]');
    if (action !== 'remove') {
        input.value = 'My working draft';
        await input.fire('input');
    }
    input.focus();
    write.resolve();
    await pending;
    await settle();
    assert.equal(page.server.posts.length, 1);
    assert.deepEqual(page.opened, []);
    assert.equal(next.querySelector('[data-editor-result]').textContent, '');
    assert.equal(page.document.activeElement, input);
    assert.equal(next.querySelector('[data-mcp-field="name"]'), input);
    if (action !== 'remove') assert.equal(input.value, 'My working draft');
    if (action === 'remove') return;
    // The owner's own Discard releases this page's draft, so no typed form outlives the test.
    await next.querySelector('[data-editor-discard]').fire('click');
    await settle();
    assert.equal(next.querySelector('[data-mcp-field="name"]').getAttribute('value'), 'Slack');
});

// Two windows over one settings server: this page's Save can be held on its way, and its reads answer as the backend's
// do, a stored header value masked. Every value is an inert local fixture string.
const MASK = '***set***';
function sharedSettings(page) {
    const endpoint = page.routes.get('/api/settings');
    const gate = { held: null };
    page.routes.set('/api/settings', async (url, init = {}) => {
        if (init.method === 'POST' && gate.held) await gate.held.promise;
        const answer = await endpoint(url, init);
        if (init.method === 'POST') return answer;
        const data = await answer.json();
        for (const entry of data.MCP_SERVERS || []) {
            for (const name of Object.keys(entry?.headers || {})) if (entry.headers[name]) entry.headers[name] = MASK;
        }
        return response(data);
    });
    return gate;
}
const typeInto = async (input, text) => { input.value = text; await input.fire('input'); };

test('a header value one Save stored is kept, not sent again, when another row was typed while that Save was on its way', async (t) => {
    const docs = { id: 'docs', name: 'Docs', enabled: true, transport: 'streamable_http', url: 'https://docs.example.test/mcp',
        headers: { 'X-A': MASK, 'X-B': MASK } };
    const page = await open(t, 'tools/docs', { values: { MCP_SERVERS: [docs] } });
    const gate = sharedSettings(page);
    const value = (n) => page.container.querySelectorAll('[data-mcp-header-row]')[n].querySelector('[data-mcp-header-field="value"]');
    const result = () => page.container.querySelector('[data-editor-result]').textContent;
    await typeInto(value(0), 'synthetic-a');
    gate.held = deferred();
    const pending = page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    await typeInto(value(1), 'synthetic-b');  // the other row, while the Save is on its way
    gate.held.resolve();
    gate.held = null;
    await pending;
    await settle();
    assert.deepEqual(page.server.posts[0].settings.MCP_SERVERS[0].headers, { 'X-A': 'synthetic-a', 'X-B': MASK });
    assert.match(result(), /latest edits are unsaved/);
    assert.deepEqual([value(0).getAttribute('value'), value(1).getAttribute('value')], [MASK, 'synthetic-b'],
        'the stored row now shows its mask; the row typed meanwhile keeps its draft');
    // Another window replaces X-A; its masked read looks the same. The next Save keeps whatever X-A now holds.
    page.server.values.MCP_SERVERS[0].headers['X-A'] = 'other-window-a';
    page.server.tokens.MCP_SERVERS += 1;
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    assert.equal(page.asked.length, 1, 'the newer list cannot identify which masked header changed');
    assert.equal(page.server.posts.length, 2);
    assert.deepEqual(page.server.posts[1].settings.MCP_SERVERS[0].headers, { 'X-A': MASK, 'X-B': 'synthetic-b' });
    assert.deepEqual(page.server.posts[1].precondition, { MCP_SERVERS: 'MCP_SERVERS@2' });
    assert.equal(result(), 'Saved');
    // A Save that sent a typed value is stored, but its answer is lost: the form is held, never sent again for
    // another field, until Discard loads what the server holds.
    await typeInto(value(0), 'synthetic-a2');
    page.server.answers.push((settings) => {
        page.server.values.MCP_SERVERS = structuredClone(settings.MCP_SERVERS);
        page.server.tokens.MCP_SERVERS += 1;
        return response({ error: 'reply lost' }, 502);
    });
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    assert.match(result(), /^Not confirmed: this Save may have been stored\. Your edits are kept here/);
    await typeInto(page.container.querySelector('[data-mcp-field="url"]'), 'https://docs.example.test/v2');
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    assert.equal(page.server.posts.length, 3, 'the literal is not sent again');
    assert.match(result(), /^Not confirmed/);
    await page.container.querySelector('[data-editor-discard]').fire('click');
    await settle();
    assert.equal(value(0).getAttribute('value'), MASK);
});

for (const tokenAvailable of [true, false]) {
    test(`clearing an opaque header asks about the observed list, version available: ${tokenAvailable}`, async t => {
        const docs = { id: 'docs', name: 'Docs', url: 'https://docs.example.test', headers: { Mode: MASK } };
        const page = await open(t, 'tools/docs', { values: { MCP_SERVERS: [docs] }, meta: tokenAvailable ? {} : { tokens: {} } });
        sharedSettings(page);
        const save = () => page.container.querySelector('[data-editor-save]').fire('click');
        await page.container.querySelector('[data-mcp-header-clear]').fire('click');
        page.server.values.MCP_SERVERS[0].headers.Mode = 'external-mode';
        page.server.tokens.MCP_SERVERS = 1;
        page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/docs' } }));
        await settle();
        const questions = [];
        page.ctx.ask = async question => { questions.push(question); return false; };
        await save();
        await settle();
        assert.equal(questions.length, 1);
        assert.match(questions[0].body, tokenAvailable ? /list changed/ : /version is unavailable/);
        assert.equal(page.server.posts.length, 0);
        assert.equal(page.container.querySelector('[data-mcp-header-field="value"]').value, '');
        page.ctx.ask = async () => {
            if (tokenAvailable) page.server.tokens.MCP_SERVERS = 2;  // a third edit during confirmation
            return true;
        };
        await save();
        await settle();
        assert.equal(page.server.posts.length, 1);
        const [post] = page.server.posts;
        assert.equal(post.settings.MCP_SERVERS[0].headers.Mode, '');
        if (tokenAvailable) {
            assert.deepEqual(post.precondition, { MCP_SERVERS: 'MCP_SERVERS@1' });
            assert.equal(page.server.values.MCP_SERVERS[0].headers.Mode, 'external-mode');
            assert.match(page.container.querySelector('[data-editor-result]').textContent, /Not saved/);
            await page.container.querySelector('[data-editor-discard]').fire('click');
        } else {
            assert.equal(post.precondition, undefined);
            assert.equal(page.server.values.MCP_SERVERS[0].headers.Mode, '');
        }
    });
}

for (const newer of [true, false]) {
    test(`no_changes settles the submitted row to its mask, newer typing: ${newer}`, async t => {
        const docs = { id: 'docs', name: 'Docs', url: 'https://docs.example.test', headers: { Mode: MASK, Colour: MASK } };
        const page = await open(t, 'tools/docs', { values: { MCP_SERVERS: [docs] } });
        const gate = sharedSettings(page);
        page.server.values.MCP_SERVERS[0].headers.Mode = 'plain';
        const values = () => page.container.querySelectorAll('[data-mcp-header-field="value"]');
        const save = () => page.container.querySelector('[data-editor-save]').fire('click');
        await typeInto(values()[0], 'plain');
        page.server.answers.push(() => response({ status: 'saved', no_changes: true, changes: {} }));
        gate.held = deferred();
        await save();
        await settle();
        if (newer) await typeInto(values()[1], 'green');
        gate.held.resolve();
        gate.held = null;
        await settle();
        assert.equal(values()[0].getAttribute('value'), MASK);
        assert.equal(page.handle.unsaved(), newer);
        const result = page.container.querySelector('[data-editor-result]').textContent;
        if (newer) {
            assert.match(result, /latest edits are unsaved/);
            assert.equal(values()[1].getAttribute('value'), 'green');
            await save();
            await settle();
            assert.deepEqual(page.server.posts[1].settings.MCP_SERVERS[0].headers, { Mode: MASK, Colour: 'green' });
        } else assert.equal(result, 'Already saved.');
    });
}

test('Discard waits for the same Save after returning to the server and then loads the stored address', async t => {
    let path = 'tools/slack';
    const page = await open(t, path, { onScreen: candidate => path === candidate });
    const gate = sharedSettings(page);
    const first = 'https://docs.example.test/first';
    await typeInto(page.container.querySelector('[data-mcp-field="url"]'), first);
    gate.held = deferred();
    await page.container.querySelector('[data-editor-save]').fire('click');
    await settle();
    assert.equal(page.container.querySelector('[data-editor-discard]').disabled, true);
    path = 'tools/files';
    const away = await settingsPageFor(path).mount(new NodeStub(), { path, ctx: page.ctx });
    t.after(() => away.unmount());
    path = 'tools/slack';
    page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path } }));
    await settle();
    const field = () => page.container.querySelector('[data-mcp-field="url"]');
    const discard = () => page.container.querySelector('[data-editor-discard]');
    assert.equal(discard().disabled, true);
    await discard().fire('click');
    const second = 'https://docs.example.test/second';
    await typeInto(field(), second);
    gate.held.resolve();
    gate.held = null;
    await settle();
    assert.equal(field().getAttribute('value'), second);
    assert.equal(page.server.values.MCP_SERVERS[0].url, first);
    assert.equal(discard().disabled, false);
    await discard().fire('click');
    assert.equal(field().getAttribute('value'), first);
    assert.equal(page.handle.unsaved(), false);
    assert.equal(page.server.posts.length, 1);
});

for (const body of ['invalid JSON', 'missing receipt']) test(`an address save with ${body} stays unknown through a failed Discard`, async t => {
    const page = await open(t, 'tools/slack');
    const endpoint = page.routes.get('/api/settings');
    const save = () => page.container.querySelector('[data-editor-save]').fire('click');
    const discard = () => page.container.querySelector('[data-editor-discard]').fire('click');
    const url = () => page.container.querySelector('[data-mcp-field="url"]');
    const result = () => page.container.querySelector('[data-editor-result]').textContent;
    const address = 'https://edited.example.test';
    await typeInto(url(), address);
    page.server.answers.push(settings => {
        page.server.values.MCP_SERVERS = structuredClone(settings.MCP_SERVERS);
        return body === 'invalid JSON' ? { ok: true, status: 200, json: async () => { throw new SyntaxError(); } }
            : response({ unexpected: true });
    });
    await save();
    await settle();
    assert.match(result(), /^Not confirmed/);
    await typeInto(url(), 'https://newer.example.test');
    await save();
    await settle();
    assert.equal(page.server.posts.length, 1);
    page.routes.set('/api/settings', async () => response({ error: 'temporarily unavailable' }, 503));
    await discard();
    assert.match(result(), /Press Discard changes again/);
    assert.equal(page.handle.unsaved(), true);
    assert.equal(url().value, 'https://newer.example.test');
    page.routes.set('/api/settings', endpoint);
    await discard();
    assert.equal(url().getAttribute('value'), address);
    assert.equal(result(), 'Current settings loaded.');
    assert.equal(page.handle.unsaved(), false);
    assert.equal(page.server.posts.length, 1);
});

const HELD = /^Not saved: the server list changed elsewhere\. Your edits are kept here, unsaved; discard them/;

for (const [how, change] of [['save', 'renamed'], ['save', 'removed'], ['read', 'renamed'], ['read', 'removed']]) {
    test(`a typed form whose server was ${change} in another window is kept after a ${how} until Discard`, async (t) => {
        const page = await open(t, 'tools/files');
        sharedSettings(page);
        const name = page.container.querySelector('[data-mcp-field="name"]');
        const discard = page.container.querySelector('[data-editor-discard]');
        const result = () => page.container.querySelector('[data-editor-result]').textContent;
        await typeInto(name, 'Local files');
        if (change === 'renamed') page.server.values.MCP_SERVERS[1].name = 'Files (other window)';
        else page.server.values.MCP_SERVERS.pop();
        page.server.tokens.MCP_SERVERS = 1;
        try {
            if (how === 'save') await page.container.querySelector('[data-editor-save]').fire('click');
            else page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/files' } }));
            await settle();
            assert.equal(page.server.posts.length, 0, 'nothing is written over the other window');
            assert.equal(page.container.querySelector('[data-mcp-field="name"]'), name, 'the form is not drawn again');
            assert.equal(name.value, 'Local files');
            assert.match(result(), HELD);
            assert.equal(discard.hasAttribute('hidden'), false);
            const toggle = page.container.querySelector('[data-entry-switch]');
            const remove = page.container.querySelector('[data-editor-remove]');
            for (const control of [toggle, remove, page.container.querySelector('[data-mcp-test]'), page.container.querySelector('[data-mcp-refresh]')]) {
                assert.equal(control.disabled, true, 'a held form has no action on a saved entry');
            }
            await toggle.fire('change', { target: toggle });
            await remove.fire('click', { target: remove });
            await page.container.querySelector('[data-editor-save]').fire('click');
            await settle();
            assert.equal(page.asked.length, 0);
            assert.equal(page.server.posts.length, 0, 'Save waits for Discard; it never retries with the newer list');
            assert.match(result(), HELD);
        } finally {
            await discard.fire('click');  // the module's held identity never outlives this test
            await settle();
        }
        if (change === 'renamed') {
            assert.equal(page.container.querySelector('[data-mcp-field="name"]').getAttribute('value'), 'Files (other window)');
            assert.equal(result(), 'Current settings loaded.');
        } else assert.match(page.container.textContent, /no longer on the list/);
    });
}

for (const typedMeanwhile of [false, true]) {
    test(`a Save refused with a 409 holds the form as typed until Discard${typedMeanwhile ? ', with the typing done while it was refused' : ''}`, async (t) => {
        const page = await open(t, 'tools/files');
        const gate = sharedSettings(page);
        const name = page.container.querySelector('[data-mcp-field="name"]');
        const discard = page.container.querySelector('[data-editor-discard]');
        const result = () => page.container.querySelector('[data-editor-result]').textContent;
        const latest = typedMeanwhile ? 'Local files 2' : 'Local files';
        await typeInto(name, 'Local files');
        // Another window writes the list between this page's read and its write; this entry still reads the same.
        page.server.answers.push(() => {
            page.server.values.MCP_SERVERS[0].name = 'Slack (other window)';
            page.server.tokens.MCP_SERVERS = 1;
            return response({ saved: false, code: 'settings_changed_elsewhere', error: 'MCP_SERVERS changed' }, 409);
        });
        try {
            gate.held = deferred();
            const pending = page.container.querySelector('[data-editor-save]').fire('click');
            await settle();
            if (typedMeanwhile) await typeInto(name, latest);
            gate.held.resolve();
            gate.held = null;
            await pending;
            await settle();
            assert.equal(page.container.querySelector('[data-mcp-field="name"]'), name, 'the form is not drawn again');
            assert.equal(name.value, latest);
            assert.match(result(), HELD);
            assert.equal(discard.hasAttribute('hidden'), false);
            // Neither another Save nor a later read that looks like the old entry releases it.
            await page.container.querySelector('[data-editor-save]').fire('click');
            page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/files' } }));
            await settle();
            assert.equal(page.server.posts.length, 1, 'one refused write and no second one');
            assert.equal(page.container.querySelector('[data-mcp-field="name"]'), name);
            assert.equal(name.value, latest);
            assert.match(result(), HELD);
        } finally {
            await discard.fire('click');
            await settle();
        }
        assert.equal(page.container.querySelector('[data-mcp-field="name"]').getAttribute('value'), 'Files');
        assert.equal(result(), 'Current settings loaded.');
        assert.equal(page.server.posts.length, 1);
        // A clean form refused the same way stays clean: it is read again and says so, nothing is held.
        page.server.answers.push(() => {
            page.server.values.MCP_SERVERS[1].command = 'npx-other';
            page.server.tokens.MCP_SERVERS = 2;
            return response({ saved: false, code: 'settings_changed_elsewhere', error: 'MCP_SERVERS changed' }, 409);
        });
        await page.container.querySelector('[data-editor-save]').fire('click');
        await settle();
        assert.equal(page.server.posts.length, 2);
        assert.equal(result(), 'Changed on another device. This is the current value.');
        assert.equal(page.container.querySelector('[data-mcp-field="command"]').getAttribute('value'), 'npx-other');
        assert.equal(page.container.querySelector('[data-editor-discard]').hasAttribute('hidden'), true);
    });
}

test('a new server refused with a 409 keeps its form and its Discard; the page of its ID shows the server saved there', async (t) => {
    const page = await open(t, 'tools/new');
    const field = (name) => page.container.querySelector(`[data-mcp-field="${name}"]`);
    const result = () => page.container.querySelector('[data-editor-result]').textContent;
    await typeInto(field('name'), 'Inbox');
    await typeInto(field('url'), 'https://inbox.example.test/mcp');
    page.server.answers.push(() => {  // another window adds a server of the same ID first
        page.server.values.MCP_SERVERS.push({ id: 'inbox', name: 'Inbox (other window)', enabled: false, transport: 'stdio', command: 'inbox' });
        page.server.tokens.MCP_SERVERS = 1;
        return response({ saved: false, code: 'settings_changed_elsewhere', error: 'MCP_SERVERS changed' }, 409);
    });
    const discard = page.container.querySelector('[data-editor-remove]');
    try {
        await page.container.querySelector('[data-editor-save]').fire('click');
        await settle();
        assert.equal(field('name').value, 'Inbox');
        assert.match(result(), HELD);
        await page.container.querySelector('[data-editor-save]').fire('click');
        await settle();
        assert.equal(page.server.posts.length, 1, 'Save waits for Discard');
        assert.equal(discard.textContent, 'Discard');
        // The page of that very ID opens the server saved there, never this unsaved form.
        const saved = new NodeStub();
        const handle = await settingsPageFor('tools/inbox').mount(saved, { path: 'tools/inbox', ctx: page.ctx });
        t.after(() => handle.unmount());
        await settle();
        assert.equal(saved.querySelector('[data-mcp-field="name"]').getAttribute('value'), 'Inbox (other window)');
        assert.equal(field('name').value, 'Inbox');
    } finally {
        await discard.fire('click');
        await settle();
    }
    assert.deepEqual(page.opened, ['tools']);
});

test('a form kept for a server removed elsewhere stays reachable from the list and is not replaced by a new server of its ID', async (t) => {
    const page = await open(t, 'tools/files');
    const name = page.container.querySelector('[data-mcp-field="name"]');
    const discard = page.container.querySelector('[data-editor-discard]');
    await typeInto(name, 'Local files');
    page.server.values.MCP_SERVERS.pop();
    page.server.tokens.MCP_SERVERS = 1;
    page.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/files' } }));
    await settle();
    const mount = async (path) => {
        const container = new NodeStub();
        const handle = await settingsPageFor(path).mount(container, { path, ctx: page.ctx });
        t.after(() => handle.unmount());
        await settle();
        return container;
    };
    const fresh = await mount('tools/new');
    const newDiscard = fresh.querySelector('[data-editor-remove]');
    try {
        // The list offers one opener for the kept form: its typed name, no switch, and it opens that server's page.
        const list = await mount('tools');
        const kept = head(list, 'files');
        assert.match(kept.textContent, /Local files.*Unsaved changes · not on the saved list/s);
        assert.equal(inside(kept, '[data-entry-switch]'), null);
        await list.fire('click', { target: inside(kept, '[data-entry-open]') });
        await settle();
        assert.deepEqual(page.opened, ['tools/files']);
        // A new server under the same ID is refused before anything is sent; the kept form is untouched.
        await typeInto(fresh.querySelector('[data-mcp-field="name"]'), 'Files');
        await typeInto(fresh.querySelector('[data-mcp-field="url"]'), 'https://files.example.test/mcp');
        await fresh.querySelector('[data-editor-save]').fire('click');
        await settle();
        assert.equal(page.server.posts.length, 0);
        assert.match(fresh.querySelector('[data-editor-result]').textContent, /Unsaved changes are kept for the ID files\. Open them from the list and discard them first\./);
        assert.equal(name.value, 'Local files');
    } finally {
        await newDiscard.fire('click');
        await discard.fire('click');
        await settle();
    }
});

// Which endpoint each key went to: the guard of the checker's transaction.
const routed = (server) => ({ checker: server.checker.flatMap((post) => Object.keys(post.settings)).sort(),
    generic: server.posts.flatMap((post) => Object.keys(post.settings)).sort() });
const misrouted = ({ checker, generic }) => [...generic.filter((key) => SAFETY_MODEL_KEYS.includes(key)).map((key) => `${key} reached the generic save`),
    ...checker.filter((key) => !SAFETY_MODEL_KEYS.includes(key)).map((key) => `${key} reached the checker`)];
const slotOf = (name) => contract.modelSlots.find((slot) => slot.slot === name);
const entryOf = (slot, changes = {}) => ({ slot: slotOf(slot), value: `openai::new-${slot}`, account: 'acct-new', context: 32000, processing: 'fast', local: true, ...changes });

test('a role\'s Save sends every checker key through the checker and every other key through the generic save', async (t) => {
    for (const slot of ['main', 'light', 'safety', 'vision', 'consciousness', 'fallback']) {
        forgetSettingRows();
        const env = setup(t);
        const server = fakeServer(env.routes);
        const fresh = await readSettings();
        const body = roleBody(fresh, entryOf(slot), 0, 'anthropic::claude-a,moonshotai/kimi,openai::gpt-b');
        const outcome = await writeModelKeys(`models:${slot}`, body, fresh);
        assert.equal(outcome.state, 'saved', slot);
        const sent = routed(server);
        assert.deepEqual(misrouted(sent), [], slot);
        assert.deepEqual([...sent.checker, ...sent.generic].sort(), Object.keys(body).sort(), `${slot}: every changed key was written once`);
        assert.deepEqual(sent.checker, Object.keys(body).filter((key) => SAFETY_MODEL_KEYS.includes(key)).sort(), slot);
        // Each write carries the tokens of the read it changed, and the checker goes first.
        for (const post of [...server.checker, ...server.posts]) assert.deepEqual(Object.keys(post.precondition).sort(), Object.keys(post.settings).sort(), slot);
    }
    // Every key of the checker has a role that writes it here (the roster is the Agents page's).
    const written = ['main', 'light', 'safety'].flatMap((slot) => Object.keys(roleBody(VALUES, entryOf(slot))));
    assert.deepEqual(SAFETY_MODEL_KEYS.filter((key) => !written.includes(key)), ['OUROBOROS_SUBAGENTS']);
    // The guard fires on a checker key in the generic save, and on a plain key in the checker's.
    assert.deepEqual(misrouted({ checker: ['OUROBOROS_MODEL_VISION'], generic: ['OUROBOROS_MODEL', 'OUROBOROS_MODEL_VISION'] }),
        ['OUROBOROS_MODEL reached the generic save', 'OUROBOROS_MODEL_VISION reached the checker']);
    // The rows of those keys name their writer; the Save in the page is this function.
    for (const row of [...ROWS, ...HAND_WRITTEN].filter((each) => each.keys.some((key) => SAFETY_MODEL_KEYS.includes(key)))) {
        assert.ok(row.keys.every((key) => SAFETY_MODEL_KEYS.includes(key)), `${row.id}: checker keys stand in rows of their own`);
        assert.equal(row.writer, 'checker', row.id);
    }
    const source = readFileSync(new URL('../modules/model_roles.js', import.meta.url), 'utf8');
    assert.doesNotMatch(source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, ''), /saveSettings\(|saveCheckerSettings\(|\/api\/owner\//);
    assert.match(source, /writeModelKeys\(`models:\$\{slot\}`, body, fresh/);
});

test('a role\'s page reads the catalogue when it opens and when it is shown again; opening and its own "shown" are one read', async (t) => {
    forgetSettingRows();
    const env = setup(t);
    fakeServer(env.routes);
    const path = 'models/role/main';
    const ctx = { readSettings, openSettings: () => true, state: { activePage: 'settings', settingsPath: path }, ask: async () => true };
    const reads = () => env.calls.filter((url) => url === '/api/model-catalog?view=sources').length;
    const shown = (to) => env.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: to } });
    const handle = await settingsPageFor(path).mount(new NodeStub(), { path, row: '', ctx });
    t.after(() => handle?.unmount?.());
    shown(path);  // the shell says "shown" as it opens the page
    await settle();
    assert.equal(reads(), 1, 'opening and its "shown" are one read');
    shown('models');
    await settle();
    assert.equal(reads(), 1, 'the showing of another page reads nothing');
    shown(path);
    await settle();
    assert.equal(reads(), 2, 'shown again (a return, a reconnect), the page catches the catalogue up');
    let release;
    env.routes.set('/api/model-catalog?view=sources', () => new Promise(resolve => { release = resolve; }));
    shown(path);
    await settle();
    const settingsReads = () => env.calls.filter(url => url === '/api/settings').length;
    const before = settingsReads();
    shown(path);
    await settle();
    assert.ok(settingsReads() > before, 'a pending descriptor cannot block the new settings document');
    release(response({ items: [], model_sources: [], errors: [] }));
    assert.equal(handle.unsaved(), false, 'nothing is typed: the shell lets the page go when it is left');
});

test('one entry of a per-role map changes and every other role\'s entry stays as the fresh read has it', async (t) => {
    const fresh = structuredClone(VALUES);
    const vision = roleBody(fresh, entryOf('vision', { account: 'studio-09', context: 0, processing: '' }));
    assert.deepEqual(vision[MODEL_ACCOUNTS_KEY], { main: 'studio-01', vision: 'studio-09' });
    assert.ok(!Object.hasOwn(vision, MODEL_CONTEXT_KEY), 'a map with no entry for the role and nothing to add is not written');
    assert.ok(!Object.hasOwn(vision, MODEL_PROCESSING_PREFERENCES_KEY));
    const light = roleBody(fresh, entryOf('light', { account: '', context: 0, processing: 'fast' }));
    assert.deepEqual(light[MODEL_CONTEXT_KEY], { light: 0, fallback: [0, 128000, 0] }, 'a cleared window keeps the other roles\' windows');
    assert.deepEqual(light[MODEL_PROCESSING_PREFERENCES_KEY], { consciousness: 'economy', light: 'fast' });
    // Only what differs from the read is sent: an unchanged role writes nothing.
    forgetSettingRows();
    const env = setup(t);
    const server = fakeServer(env.routes);
    const read = await readSettings();
    const same = roleBody(read, { slot: slotOf('main'), value: VALUES.OUROBOROS_MODEL, account: 'studio-01', context: 0, processing: '', local: false });
    assert.equal((await writeModelKeys('models:main', same, read)).state, 'unchanged');
    assert.deepEqual([server.posts.length, server.checker.length], [0, 0]);
    // Another device changed the map after the read: the write is refused, not merged over.
    server.tokens[MODEL_ACCOUNTS_KEY] = 7;
    const stale = await writeModelKeys('models:vision', roleBody(read, entryOf('vision')), read);
    assert.equal(stale.state, 'changed_elsewhere');
    assert.deepEqual(server.values[MODEL_ACCOUNTS_KEY], VALUES[MODEL_ACCOUNTS_KEY]);
});

test('Web search saves only what the owner changed: the source or the model chosen elsewhere meanwhile stays', async (t) => {
    const page = await open(t, 'models/role/websearch');
    const [source, model] = ['#s-websearch-source', '#s-websearch-model'].map((selector) => page.container.querySelector(selector));
    const save = async () => page.press(page.container.querySelector('[data-editor-save]'));
    const elsewhere = (key, value) => { page.server.values[key] = value; page.server.tokens[key] = (page.server.tokens[key] || 0) + 1; };
    // The model typed; the source moves elsewhere before Save: only the model is written, and the page shows the source.
    model.value = 'gpt-search';
    await model.fire('input', { target: model });
    elsewhere('OUROBOROS_WEBSEARCH_BACKEND', 'openrouter');
    await save();
    assert.deepEqual(page.server.posts.map((post) => post.settings), [{ OUROBOROS_WEBSEARCH_MODEL: 'gpt-search' }]);
    assert.deepEqual([page.server.values.OUROBOROS_WEBSEARCH_BACKEND, source.value], ['openrouter', 'openrouter']);
    // The source chosen; the model moves elsewhere: only the source is written, and the other model stays.
    source.value = 'ddgs';
    await source.fire('change', { target: source });
    elsewhere('OUROBOROS_WEBSEARCH_MODEL', 'gpt-other');
    await save();
    assert.deepEqual(page.server.posts.at(-1).settings, { OUROBOROS_WEBSEARCH_BACKEND: 'ddgs' });
    assert.deepEqual([page.server.values.OUROBOROS_WEBSEARCH_MODEL, model.value], ['gpt-other', 'gpt-other']);
    // Nothing changed since the read: nothing is sent.
    await save();
    assert.equal(page.server.posts.length, 2);
    assert.equal(page.container.querySelector('[data-editor-result]').textContent, 'Nothing to save.');
});

test('Web search adopts a read, a return snapshot and a fallback through the same form and Key painter before unlocking', async (t) => {
    forgetSettingRows();
    const env = setup(t);
    const server = fakeServer(env.routes);
    const originalSettings = env.routes.get('/api/settings');
    const first = deferred();
    env.routes.set('/api/settings', async (url, init = {}) => init.method ? originalSettings(url, init) : first.promise);
    const state = { activePage: 'settings', settingsPath: 'models/role/websearch' };
    const mount = async () => {
        const container = new NodeStub();
        const handle = await settingsPageFor('models/role/websearch').mount(container, { path: state.settingsPath,
            row: '', ctx: { readSettings, openSettings() {}, state } });
        return { container, handle };
    };
    const firstPage = await mount();
    t.after(() => firstPage.handle?.unmount?.());
    await settle();
    const fields = (container) => [container.querySelector('#s-websearch-source'), container.querySelector('#s-websearch-model'),
        container.querySelector('#s-websearch-key'), container.querySelector('[data-editor-save]'), container.querySelector('[data-editor-discard]')];
    assert.deepEqual(fields(firstPage.container).map((control) => [control.value || '', control.disabled]),
        [['', true], ['', true], ['', true], ['', true], ['', true]], 'nothing is shown or operable before the read');
    first.resolve(await originalSettings('/api/settings', {}));
    await settle();
    assert.deepEqual(fields(firstPage.container).slice(0, 2).map((control) => [control.value || '', control.disabled]),
        [['auto', false], ['', false]], 'the read paints and unlocks the fields');
    firstPage.handle.unmount();

    const second = deferred();
    env.routes.set('/api/settings', async (url, init = {}) => init.method ? originalSettings(url, init) : second.promise);
    const secondPage = await mount();
    t.after(() => secondPage.handle?.unmount?.());
    await settle();
    assert.deepEqual(fields(secondPage.container).slice(0, 2).map((control) => [control.value || '', control.disabled]),
        [['auto', false], ['', false]], 'a return starts from the last read instead of a blank choice');
    second.resolve(await originalSettings('/api/settings', {}));
    await settle();
    assert.deepEqual(fields(secondPage.container).slice(0, 2).map((control) => [control.value || '', control.disabled]),
        [['auto', false], ['', false]], 'the returning read leaves the choice in place');
    assert.equal(server.values.OUROBOROS_WEBSEARCH_BACKEND, 'auto');
    secondPage.handle.unmount();

    // Its first read fails after somebody else's read succeeded: unlock only after adopting that answer.
    forgetSettingRows();
    Object.assign(server.values, { OUROBOROS_WEBSEARCH_BACKEND: 'openai', OUROBOROS_WEBSEARCH_MODEL: 'gpt-5.2' });
    const failed = deferred();
    env.routes.set('/api/settings', () => failed.promise);
    const raced = await mount();
    t.after(() => raced.handle.unmount());
    assert.equal(fields(raced.container)[3].disabled, true);
    env.routes.set('/api/settings', originalSettings);
    await readSettings();
    failed.resolve(response({ error: 'older read failed' }, 500));
    await settle();
    const [source, model, , save] = fields(raced.container);
    assert.deepEqual([source.value, model.value, save.disabled], ['openai', 'gpt-5.2', false]);
    assert.notEqual(raced.container.querySelector('[data-websearch-preview]').textContent, 'Reading…');
    source.value = 'auto';
    await source.fire('change', { target: source });
    await raced.container.fire('click', { target: save });
    await settle();
    assert.deepEqual(server.posts.at(-1).settings, { OUROBOROS_WEBSEARCH_BACKEND: 'auto' }, 'saving Source cannot erase the model');
    assert.equal(server.values.OUROBOROS_WEBSEARCH_MODEL, 'gpt-5.2');
    raced.handle.unmount();

    Object.assign(server.values, { OUROBOROS_WEBSEARCH_BACKEND: 'deepseek', OUROBOROS_WEBSEARCH_MODEL: 'deepseek@team::' });
    await readSettings();
    const returning = deferred();
    env.routes.set('/api/settings', () => returning.promise);
    const deepseek = await mount();
    t.after(() => deepseek.handle.unmount());
    const keyShown = () => {
        const [source, model, key, save] = fields(deepseek.container);
        return [source.value, model.hidden, key.hidden, key.value, save.disabled];
    };
    assert.deepEqual(keyShown(), ['deepseek', true, false, 'api:deepseek@team', false], 'the return paints Key, hiding its wire spelling');
    returning.resolve(response({ error: 'read failed' }, 500));
    await settle();
    assert.deepEqual(keyShown(), ['deepseek', true, false, 'api:deepseek@team', false], 'a failed own read keeps the adopted Key');
});

test('the fallbacks keep their per-role entries through an order change, a removal and one fallback\'s own Save', () => {
    const list = fallbackEntries(VALUES);
    assert.deepEqual(list.map((entry) => [entry.value, entry.context]), [['anthropic::claude-a', 0], ['moonshotai/kimi', 128000], ['openai::gpt-b', 0]]);
    const moved = fallbackBody(VALUES, [list[1], list[0], list[2]]);
    assert.equal(moved.OUROBOROS_MODEL_FALLBACKS, 'moonshotai/kimi, anthropic::claude-a, openai::gpt-b');
    assert.deepEqual(moved[MODEL_CONTEXT_KEY], { light: 64000, fallback: [128000, 0, 0] }, 'a window moves with its fallback');
    assert.ok(!Object.hasOwn(moved, MODEL_ACCOUNTS_KEY), 'a map without a fallback entry is left alone');
    assert.deepEqual(fallbackBody(VALUES, [list[0], list[2]])[MODEL_CONTEXT_KEY].fallback, [0, 0]);
    const shown = list.map((entry) => entry.value).join(',');
    const second = roleBody(VALUES, entryOf('fallback', { value: 'openai::gpt-c', account: '', context: 0, processing: '', local: false }), 1, shown);
    assert.equal(second.OUROBOROS_MODEL_FALLBACKS, 'anthropic::claude-a, openai::gpt-c, openai::gpt-b');
    assert.equal(roleBody(VALUES, entryOf('fallback'), 1, 'another,order'), null, 'a form read from another order writes nothing');
    assert.equal(roleBody(VALUES, entryOf('fallback', { value: 'x/new' }), 3, shown).OUROBOROS_MODEL_FALLBACKS.split(', ').length, 4, 'one past the end is a new fallback');
});

test('the Models list shows each role\'s route in the picture\'s order, and the Context door posts nothing', async (t) => {
    const page = await open(t, 'models');
    assert.deepEqual(page.container.querySelectorAll('[data-entry-id]').map((node) => node.dataset.entryId),
        ['main', 'fallback/1', 'fallback/2', 'fallback/3', 'light', 'vision', 'consciousness', 'safety', 'websearch']);
    // The role's name is the card's title above its one row: the model, then the source (79fac1cf4, model_roles.js).
    const mainCard = page.container.querySelector('[data-setting-row="models:main"]');
    assert.equal(inside(mainCard, '.settings-card-title').textContent, 'Main model');
    assert.match(head(page.container, 'main').textContent, /gpt-main.*Codex subscription · account studio-01/s);
    assert.match(head(page.container, 'fallback/2').textContent, /2.*kimi.*OpenRouter · by API key/s);
    const value = (id) => inside(page.container.querySelector(`[data-setting-row="${id}"]`), '[data-row-value]')?.textContent;
    assert.deepEqual(['light', 'safety', 'vision', 'consciousness', 'websearch'].map((slot) => value(`models:${slot}`)),
        ['gpt-light', 'gpt-light · as Light', 'Main model', 'Main model', 'Auto'], 'an empty Supervisor names the model it checks with');
    assert.deepEqual(rowsOf('models', { fold: false }).map((row) => row.id), ['OUROBOROS_IMAGE_INPUT_MODE', 'models:context', 'models:local', 'models:catalogue']);
    assert.deepEqual(rowsOf('models', { fold: true }).map((row) => row.id), ['OUROBOROS_PROCESSING_PREFERENCE', 'OUROBOROS_PROMPT_CACHE_TTL', 'OUROBOROS_MODEL_MAX_CONCURRENCY']);
    // A request over the limit waits in line and is never sent elsewhere for waiting: the queue names no clock of its own.
    const limit = page.container.querySelector('[data-setting-row="OUROBOROS_MODEL_MAX_CONCURRENCY"]').textContent;
    assert.match(limit, /waits in line until a slot frees.*never sent elsewhere for waiting/s);
    assert.doesNotMatch(limit, /waits up to|\{wait\}/);
    // A visit, even via keyboard or touch, does not start the all-provider read. Its explicit action still does.
    const catalogue = page.container.querySelector('[data-setting-row="models:catalogue"]');
    const refresh = inside(catalogue, '#btn-refresh-model-catalog');
    assert.equal(refresh.textContent, 'Read again');
    assert.ok(inside(catalogue, '#settings-model-catalog-status'));
    const catalogueReads = () => page.calls.filter((url) => url === '/api/model-catalog').length;
    page.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: 'models' } });
    await page.container.fire('pointerdown', { target: catalogue });
    await page.container.fire('focusin', { target: catalogue });
    await settle();
    assert.equal(catalogueReads(), 0);
    await page.container.fire('click', { target: refresh });
    await settle();
    assert.equal(catalogueReads(), 1);
    // The door shows the install's Context and leads to Main; it never writes it.
    const door = page.container.querySelector('[data-setting-row="models:context"]');
    assert.deepEqual([rowById('models:context').keys, rowById('models:context').control], [[], { kind: 'door', to: 'main:context' }]);
    assert.equal(value('models:context'), 'Max');
    await page.press(door);
    assert.deepEqual(page.opened, ['main:context']);
    assert.deepEqual([page.server.posts.length, page.server.checker.length], [0, 0]);
    assert.ok(!page.calls.some((url) => String(url).includes('context-mode')));
    // A role's line opens its page, a fallback's its own.
    await page.press(page.container.querySelector('[data-setting-row="models:vision"]'));
    await page.press(inside(head(page.container, 'fallback/2'), '[data-entry-open]'));
    await page.press(page.container.querySelector('[data-fallback-add]'));
    assert.deepEqual(page.opened.slice(1), ['models/role/vision', 'models/role/fallback/2', 'models/role/fallback/4']);
});

test('a role whose key cannot send says so on the Models list, and Main waiting for a key choice says which; a key that sends says nothing', async (t) => {
    // The Anthropic key is turned off and OpenRouter's is gone: the fallbacks on them and Light (OpenAI, still on) differ.
    const page = await open(t, 'models', { values: { OPENROUTER_API_KEY: '',
        OUROBOROS_CONNECTIONS: [{ kind: 'anthropic', name: 'default', enabled: false }] } });
    const meta = (id) => inside(head(page.container, id), '.ui-row-meta');
    assert.equal(meta('fallback/1').textContent, 'Anthropic is turned off in Accounts; nothing is sent through it.');
    assert.equal(meta('fallback/1').getAttribute('data-tone'), 'warn');
    assert.equal(meta('fallback/2').textContent, 'OpenRouter has no key in Accounts.');
    assert.equal(meta('fallback/3').textContent, 'OpenAI · by API key', 'a key that sends keeps its plain line');
    assert.equal(meta('fallback/3').getAttribute('data-tone'), null);
    const light = () => inside(page.container.querySelector('[data-setting-row="models:light"]'), '[data-setting-meta]');
    const described = light().textContent;
    assert.ok(described && light().getAttribute('data-tone') === null, 'Light on a working key keeps its description');
    // Its OpenAI key turned off: Light's line says so, also after the rows are painted again, and comes back after.
    page.server.values.OUROBOROS_CONNECTIONS = [{ kind: 'anthropic', name: 'default', enabled: false }, { kind: 'openai', name: 'default', enabled: false }];
    page.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: 'models' } });
    await settle();
    assert.equal(light().textContent, 'OpenAI is turned off in Accounts; nothing is sent through it.');
    assert.equal(light().getAttribute('data-tone'), 'warn');
    page.server.values.OUROBOROS_CONNECTIONS = [];
    page.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: 'models' } });
    await settle();
    assert.deepEqual([light().textContent, light().getAttribute('data-tone')], [described, null]);
    // Main on a Codex subscription has no key to lose; Main waiting for one of two OpenAI keys names both.
    assert.ok(!inside(page.container.querySelector('.models-main'), '.models-main-meta').dataset.tone);
    const waiting = await open(t, 'models', { values: { OUROBOROS_MODEL: 'openai::gpt-main',
        OUROBOROS_CONNECTIONS: [{ kind: 'openai', name: 'work', api_key: 'sk-w' }, { kind: 'openai', name: 'home', api_key: 'sk-h' }] },
        meta: { main_connection_choice: ['openai@work', 'openai@home'] } });
    const card = waiting.container.querySelector('.models-main');
    assert.equal(inside(card, '.models-main-name').textContent, 'Not chosen yet');
    assert.equal(inside(card, '.models-main-meta').textContent, 'Choose the key it uses: OpenAI · work or OpenAI · home.');
});

test('the fallbacks\' order and removal are written on the list: the order through the generic save, their windows through the checker; Remove asks first', async (t) => {
    const page = await open(t, 'models');
    await page.press(inside(head(page.container, 'fallback/2'), '[data-fallback-move]'));
    assert.deepEqual(routed(page.server), { checker: [MODEL_CONTEXT_KEY], generic: ['OUROBOROS_MODEL_FALLBACKS'] });
    assert.equal(page.server.values.OUROBOROS_MODEL_FALLBACKS, 'moonshotai/kimi, anthropic::claude-a, openai::gpt-b');
    assert.deepEqual(page.server.values[MODEL_CONTEXT_KEY].fallback, [128000, 0, 0]);
    assert.match(page.container.querySelector('[data-entry-result]').textContent, /^Order saved/);
    assert.match(head(page.container, 'fallback/1').textContent, /kimi/);
    await page.press(inside(head(page.container, 'fallback/3'), '[data-fallback-remove]'));
    assert.equal(page.asked.length, 1);
    assert.match(page.asked[0].title, /Remove fallback 3, gpt-b\?/);
    assert.equal(page.server.values.OUROBOROS_MODEL_FALLBACKS, 'moonshotai/kimi, anthropic::claude-a');
    assert.match(page.container.querySelector('[data-entry-result]').textContent, /^Removed/);
});

test('a route is said in the owner\'s words', () => {
    assert.deepEqual(routeWords('claudexor::codex=gpt-x'), { name: 'gpt-x', mark: 'codex', label: 'Codex',
        source: 'Codex subscription · your accounts in turn' });
    assert.equal(routeWords('anthropic@work::claude-opus').source, 'Anthropic (work) · by API key');
    assert.equal(routeWords('google/gemini-pro').name, 'gemini-pro');
    assert.deepEqual(routeWords('', { follows: 'Light' }), { name: 'Light model', source: '', mark: '', label: '', follows: true });
    assert.equal(routeWords('openai::x', { local: true }).name, 'Local model');
});

test('a subscription is named by its held descriptor, and keeps the old generic name without one', () => {
    // An opaque source id is no readable name; only the catalogue's descriptor names its owner.
    const sources = [{ id: 'claude', label: 'Claude Code', credentialHarness: 'claude' }];
    assert.deepEqual(routeWords('claudexor::claude=claude-opus-5-5', { modelSources: sources }),
        { name: 'claude-opus-5-5', mark: 'claude', label: 'Claude Code',
            source: 'Claude Code subscription · your accounts in turn' });
    assert.equal(routeWords('claudexor::future=gpt-x').label, 'future',
        'no descriptor: the old generic label stands, never a guessed product name');
    assert.equal(routeWords('claudexor::claude=claude-opus-5-5', { account: 'studio-01', modelSources: sources }).source,
        'Claude Code subscription · account studio-01');
});

// The Models list's Main row: its mark and its source line come from the catalogue a model sheet already holds.
// Nothing here reads the catalogue; a descriptor adopted later reaches only those two slots.
test('the Models Main row wears its descriptor\'s mark and name, and a later descriptor reaches only them', async (t) => {
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity work', credentialHarness: 'agy' }] });
    const value = 'claudexor::agy=claude-opus-5-5';
    const page = await open(t, 'models', { values: { OUROBOROS_MODEL: value, OUROBOROS_MODEL_ACCOUNTS: { main: 'studio-01' } } });
    const row = () => page.container.querySelector('.models-main-row');
    const meta = () => inside(row(), '.models-main-meta');
    const markNode = () => inside(meta(), '.model-source-mark');
    const catalogueReads = () => page.calls.filter((url) => String(url).startsWith('/api/model-catalog')).length;
    // Already held: the descriptor's mark and its name at the first paint, with no read of this page's own.
    const reads = catalogueReads();
    assert.equal(markNode().getAttribute('viewBox'), '0 0 180 180', 'a held descriptor draws its mark at first paint');
    assert.match(meta().textContent, /Antigravity work subscription · account studio-01/);
    assert.equal(inside(meta(), '[data-i18n-authored]').textContent, 'Antigravity work', 'the name is authored, not chrome');
    // A held list that names no harness: the neutral initial and the old generic name, never a brand guessed from the id.
    const rowNode = row();
    modelCatalog.adopt({ items: [], model_sources: [] });
    await settle();
    assert.equal(row(), rowNode, 'the card is not redrawn for a mark');
    assert.match(markNode().getAttribute('class'), /model-source-initial/, 'no descriptor: a neutral initial');
    assert.match(meta().textContent, /Antigravity subscription · account studio-01/s,
        'without a descriptor the old generic name stands, and no brand is chosen by the id alone');
    // A descriptor adopted later brands the route it names: only the mark and its source line change.
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity work', credentialHarness: 'agy' }] });
    await settle();
    assert.equal(row(), rowNode, 'the card is not redrawn for a mark');
    assert.equal(inside(meta(), '.model-source-mark').getAttribute('viewBox'), '0 0 180 180', 'the descriptor\'s Antigravity mark');
    assert.match(meta().textContent, /Antigravity work subscription · account studio-01/);
    assert.equal(catalogueReads(), reads, 'no read for a mark');
    // The same descriptor again is a no-op: the drawn nodes stay the ones the owner sees.
    const mark = inside(meta(), '.model-source-mark');
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity work', credentialHarness: 'agy' }] });
    await settle();
    assert.equal(inside(meta(), '.model-source-mark'), mark, 'an unchanged result writes nothing');
    // A descriptor that names no credential harness is no evidence of one: the neutral initial stays.
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity work', credentialHarness: '' }] });
    await settle();
    assert.match(inside(meta(), '.model-source-mark').getAttribute('class'), /model-source-initial/,
        'an unbound opaque id is never branded');
    assert.doesNotMatch(meta().innerHTML, /viewBox="0 0 180 180"/);
});

test('a saved local-model setting says it applies when the local model is restarted, never a restart of Ouroboros', async (t) => {
    const rows = rowsOf('models/local');
    assert.deepEqual(rows.map((row) => row.id), ['LOCAL_MODEL_SOURCE', 'LOCAL_MODEL_FILENAME', 'LOCAL_MODEL_CONTEXT_LENGTH',
        'LOCAL_MODEL_N_GPU_LAYERS', 'LOCAL_MODEL_PORT', 'LOCAL_MODEL_CHAT_FORMAT']);
    for (const row of rows) {
        const line = outcomeLine(row, { state: 'saved', effect: 'local_model', withheld: '', old: 'before', new: 'after', token: 't', message: '', code: '' });
        assert.match(line.text, /^Saved · applies when the local model is restarted · was before$/, row.id);
        assert.doesNotMatch(line.text, /Ouroboros|after restart/i, row.id);
        // The guard fires on the wording a restart of Ouroboros would take.
        assert.match(outcomeLine(row, { state: 'saved', effect: 'restart', withheld: '', old: 'before', token: 't', message: '', code: '' }).text, /after restart/);
    }
    const page = await open(t, 'models/local', { values: { LOCAL_MODEL_SOURCE: 'org/model-GGUF', LOCAL_MODEL_FILENAME: 'm.gguf', LOCAL_MODEL_PORT: 8766,
        LOCAL_MODEL_N_GPU_LAYERS: -1, LOCAL_MODEL_CONTEXT_LENGTH: 16384, LOCAL_MODEL_CHAT_FORMAT: '' } });
    const text = page.container.textContent;
    for (const word of ['Start', 'Stop', 'Test tool calling', 'Install the local runtime']) assert.ok(text.includes(word), word);
    assert.doesNotMatch(text, /Restart/);
    assert.doesNotMatch(readFileSync(new URL('../modules/settings_local_model.js', import.meta.url), 'utf8'), /Restart (now|Ouroboros)|restart_required|['"]restart['"]/);
    assert.deepEqual(localModelStatusWords({ status: 'ready', context_length: 16384 }), { text: 'Ready · context 16384', tone: 'ok' });
    assert.deepEqual(localModelStatusWords({ status: 'error', error: 'no such file' }), { text: 'Error · no such file', tone: 'danger' });
});

test('an empty Safety Supervisor shows what the runtime gives it along Safety → Light → Main: the model, the account, the tier', async (t) => {
    const page = await open(t, 'models', { values: { OUROBOROS_MODEL_LIGHT: '',
        [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'fast', consciousness: 'economy' } } });
    const value = (id) => inside(page.container.querySelector(`[data-setting-row="${id}"]`), '[data-row-value]')?.textContent;
    assert.equal(value('models:safety'), 'gpt-main · account studio-01 · Fast · as Main, via Light');
    // Light's own model ends the chain: Light's options, and none of Main's.
    const light = await open(t, 'models', { values: { [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'fast' } } });
    const lightValue = inside(light.container.querySelector('[data-setting-row="models:safety"]'), '[data-row-value]')?.textContent;
    assert.equal(lightValue, 'gpt-light · as Light');
});

/* Two windows on Main's page: this one typed a new model; the other saved something meanwhile. */
async function mainPage(t, values = {}, { revision = '' } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const server = fakeServer(env.routes, values);
    server.revision = revision;  // the named key's revision, as the server words it now
    const read = server.document;
    if (revision) server.document = () => { const doc = read(); doc._meta.connections = [{ id: 'openai@work', revision: server.revision }]; return doc; };
    const path = 'models/role/main';
    const asked = [];
    const ctx = { readSettings, openSettings: () => true, state: { activePage: 'settings', settingsPath: path },
        ask: async (question) => { asked.push(question); return true; } };
    const container = new NodeStub();
    env.document.body.appendChild(container);  // the role editor finds its host by id in the document
    const handle = await settingsPageFor(path).mount(container, { path, row: '', ctx });
    await settle();
    t.after(() => handle?.unmount?.());
    const page = { ...env, server, ctx, container, asked, handle,
        press: async (target) => { await container.fire('click', { target }); await settle(); } };
    const model = page.container.querySelector('[data-model-role-model]');
    const elsewhere = (key, value) => { page.server.values[key] = value; page.server.tokens[key] = (page.server.tokens[key] || 0) + 1; };
    const result = () => page.container.querySelector('[data-editor-result]').textContent;
    return { ...page, elsewhere, result,
        // The chooser's popup is layout work this DOM stub has none of; a disabled field never opens it, and the
        // editor's own `input` listener, which is what is typed into, does not read that flag.
        type: async (text) => { model.disabled = true; model.value = text; await model.fire('input', { target: model }); await settle(); },
        save: () => page.press(page.container.querySelector('[data-editor-save]')) };
}

test('two windows: a new Main model saved here never takes a tier set elsewhere meanwhile; an unrelated save elsewhere keeps it saving', async (t) => {
    const page = await mainPage(t);
    await page.type('gpt-next');
    page.elsewhere(MODEL_PROCESSING_PREFERENCES_KEY, { consciousness: 'economy', main: 'ultrafast' });
    await page.save();
    assert.deepEqual([page.server.posts, page.server.checker], [[], []], 'nothing is sent on a binding this page did not show');
    assert.match(page.result(), /^Not saved\. Main's processing is Ultrafast now, set on another device\./);
    assert.equal(page.server.values.OUROBOROS_MODEL, 'claudexor::codex=gpt-main');
    // Another window saved an unrelated setting: the model is written, standing on Main's whole binding.
    const other = await mainPage(t);
    await other.type('gpt-next');
    other.elsewhere('OUROBOROS_MODEL_LIGHT', 'openai::gpt-other');
    await other.save();
    const sent = [...other.server.checker, ...other.server.posts].find((post) => 'OUROBOROS_MODEL' in post.settings);
    assert.deepEqual(Object.keys(sent.precondition).sort(), ['OUROBOROS_MODEL', MODEL_ACCOUNTS_KEY, MODEL_PROCESSING_PREFERENCES_KEY, 'USE_LOCAL_MAIN'].sort());
    assert.equal(other.server.values.OUROBOROS_MODEL, 'claudexor::codex=gpt-next');
});

test('carrying stored Ultrafast to an unlisted model asks before the write', async (t) => {
    const page = await mainPage(t, { [MODEL_ACCOUNTS_KEY]: {}, [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'ultrafast' } });
    await page.type('gpt-next');
    await page.save();
    assert.equal(page.asked.length, 1);
    assert.equal(page.asked[0].title, 'Ultrafast for Main');
    assert.equal(page.server.values.OUROBOROS_MODEL, 'claudexor::codex=gpt-next');
    assert.match(page.result(), /^Saved/);
    const sent = [...page.server.checker, ...page.server.posts].find((post) => 'OUROBOROS_MODEL' in post.settings);
    assert.ok(!('connection' in sent.precondition), 'a subscription has no key to bind');
    assert.deepEqual(JSON.parse(sent.precondition.ultrafast), ['claudexor::codex=gpt-next', false, '', '', '']);
});

test('two windows: Main\'s Ultrafast stands on its key as this page showed it; a key rotated elsewhere refuses, an unrelated edit does not', async (t) => {
    const keyed = { OUROBOROS_MODEL: 'openai@work::gpt-a', [MODEL_ACCOUNTS_KEY]: {}, [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'ultrafast' } };
    const page = await mainPage(t, keyed, { revision: 'openai@work#r1' });
    await page.type('gpt-b');
    page.server.revision = 'openai@work#r2';  // the other window rotated the named key (or its API surface)
    await page.save();
    assert.deepEqual([page.server.posts, page.server.checker], [[], []], 'nothing is sent on a key the page did not show');
    assert.match(page.result(), /^Not saved\. The key this model goes through changed on another device\./);
    assert.equal(page.server.values.OUROBOROS_MODEL, 'openai@work::gpt-a');
    const other = await mainPage(t, keyed, { revision: 'openai@work#r1' });
    await other.type('gpt-b');
    other.elsewhere('OUROBOROS_MODEL_LIGHT', 'openai::gpt-other');
    await other.save();
    const sent = [...other.server.checker, ...other.server.posts].find((post) => 'OUROBOROS_MODEL' in post.settings);
    assert.equal(sent.precondition.connection, 'openai@work#r1', 'the revision shown, sent as shown');
    assert.equal(other.server.values.OUROBOROS_MODEL, 'openai@work::gpt-b');
});
