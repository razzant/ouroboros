// Settings › Tools and Settings › Models (docs/DESIGN.md "Settings"), against a fake settings server that answers
// like the real one (`changes` with a token per key, `_meta.effects`, `_meta.tokens`, a 409 for a stale list).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import '../modules/settings.js';
import { SAFETY_MODEL_KEYS } from '../modules/consciousness_api.js';
import { applyMcpSettings, collectMcpSettings } from '../modules/mcp_settings.js';
import { MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_KEY, fallbackBody, fallbackEntries, roleBody, routeWords, slotWaitSeconds,
    writeModelKeys } from '../modules/model_roles.js';
import { MODEL_PROCESSING_PREFERENCES_KEY } from '../modules/route_editor_primitives.js';
import { forgetSettingRows, outcomeLine, readSettings } from '../modules/settings_controls.js';
import { localModelStatusWords } from '../modules/settings_local_model.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { HAND_WRITTEN, ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 12) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

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
    OUROBOROS_PROCESSING_PREFERENCE: '', OUROBOROS_PROMPT_CACHE_TTL: '1h', OUROBOROS_MODEL_MAX_CONCURRENCY: 3, OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC: 45,
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
        setup_contract: { modelSlots: contract.modelSlots } } });
    const write = (settings, precondition = {}) => {
        const stale = Object.entries(precondition || {}).find(([key, held]) => held !== token(key));
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

async function open(t, path, { values = {}, meta = {} } = {}) {
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

test('the Tools list: the client switch is a plain row, each server is a head with its state, and the fold holds the two waits', async (t) => {
    const page = await open(t, 'tools');
    assert.deepEqual(rowsOf('tools', { fold: false }).map((row) => row.id), ['MCP_ENABLED', 'tools:door:skills', 'tools:door:circles']);
    assert.deepEqual(rowsOf('tools', { fold: true }).map((row) => row.id), ['MCP_TOOL_TIMEOUT_SEC', 'OUROBOROS_TOOL_TIMEOUT_SEC', 'tools:import', 'tools:refresh']);
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
    assert.equal(lead(), 'All tool servers are off. Their settings are kept.', 'the client\'s switch changes what the status says');
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
    const reads = () => env.calls.filter((url) => url === '/api/model-catalog').length;
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
    assert.match(head(page.container, 'main').textContent, /Main model.*gpt-main.*Codex subscription · account studio-01/s);
    assert.match(head(page.container, 'fallback/2').textContent, /2.*kimi.*OpenRouter · by API key/s);
    const value = (id) => inside(page.container.querySelector(`[data-setting-row="${id}"]`), '[data-row-value]')?.textContent;
    assert.deepEqual(['light', 'safety', 'vision', 'consciousness', 'websearch'].map((slot) => value(`models:${slot}`)),
        ['gpt-light', 'Light model', 'Main model', 'Main model', 'Auto']);
    assert.deepEqual(rowsOf('models', { fold: false }).map((row) => row.id), ['OUROBOROS_IMAGE_INPUT_MODE', 'models:context', 'models:local', 'models:catalogue']);
    assert.deepEqual(rowsOf('models', { fold: true }).map((row) => row.id), ['OUROBOROS_PROCESSING_PREFERENCE', 'OUROBOROS_PROMPT_CACHE_TTL', 'OUROBOROS_MODEL_MAX_CONCURRENCY']);
    assert.match(page.container.querySelector('[data-setting-row="OUROBOROS_MODEL_MAX_CONCURRENCY"]').textContent, /waits up to 45 s/);
    // The catalogue is an action of its row, with its own result beside it; it is read when the page is shown.
    const catalogue = page.container.querySelector('[data-setting-row="models:catalogue"]');
    assert.equal(inside(catalogue, '#btn-refresh-model-catalog').textContent, 'Read again');
    assert.ok(inside(catalogue, '#settings-model-catalog-status'));
    assert.equal(page.calls.filter((url) => url === '/api/model-catalog').length, 0, 'not read before the page is shown');
    page.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: 'tools' } });
    page.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path: 'models' } });
    await settle();
    assert.equal(page.calls.filter((url) => url === '/api/model-catalog').length, 1, 'read once, for its own path');
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

test('a route is said in the owner\'s words, and the slot wait keeps its bounds', () => {
    assert.deepEqual(routeWords('claudexor::codex=gpt-x'), { name: 'gpt-x', mark: 'codex', source: 'Codex subscription · your accounts in turn' });
    assert.equal(routeWords('anthropic@work::claude-opus').source, 'Anthropic (work) · by API key');
    assert.equal(routeWords('google/gemini-pro').name, 'gemini-pro');
    assert.deepEqual(routeWords('', { follows: 'Light' }), { name: 'Light model', source: '', mark: '', follows: true });
    assert.equal(routeWords('openai::x', { local: true }).name, 'Local model');
    assert.deepEqual([slotWaitSeconds('45'), slotWaitSeconds(-3), slotWaitSeconds(''), slotWaitSeconds('soon')], [45, 0, 180, 180]);
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
