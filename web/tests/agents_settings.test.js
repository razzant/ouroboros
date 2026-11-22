// Settings › Agents (subagents_settings.js): the list of heads and one agent's page, against a fake settings server.
// Every change is one write of the roster through writeEntry and the checker's endpoint; the server's half answers
// here as PINS section 6 pins it (`_meta.tokens`, `if_unchanged`, a receipt in `changes`, its refusals).
import assert from 'node:assert/strict';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN } from '../modules/api_client.js';
import { forgetSettingRows, readSettings } from '../modules/settings_controls.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { HAND_WRITTEN, ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { agentName, rosterResult } from '../modules/subagents_settings.js';
import { createAgentModelCatalog } from '../modules/agent_model_catalog.js';
import { modelCatalog } from '../modules/model_catalog_store.js';
import { claudexorStatus } from '../modules/claudexor_status_store.js';
import { recheckDraft, recheckSaved } from '../modules/safety_recheck.js';
import { SAFETY_MODEL_KEYS } from '../modules/consciousness_api.js';
import '../modules/settings_ui.js';  // the shared ownerCap codec, registered before any page as settings.js does
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const KEY = 'OUROBOROS_SUBAGENTS';
const CHECKER = '/api/owner/safety-model';
const EMPTY_POOL = 'Every row marked Reviewer is switched off, so the main model would review alone.';
const TWINS = 'Subagent 3 runs the same engine as Subagent 2 — change its model, effort, access, account or processing.';
// The fixture's nodes keep attributes and properties apart: a drawn switch is on by its attribute, a pressed or
// painted one by its property.
NodeStub.prototype.toggleAttribute ??= function toggleAttribute(name, force) {
    if (force ?? !this.hasAttribute(name)) this.setAttribute(name, ''); else this.removeAttribute(name);
};
Object.defineProperty(NodeStub.prototype, 'attributes', { configurable: true,
    get() { return Object.entries(this.attrs).map(([name, value]) => ({ name, value })); } });
// A field's text: what was typed into it, else what it was drawn with.
const words = (field) => field.value ?? field.textContent;
const isOn = (input) => input.checked ?? input.hasAttribute('checked');
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 12) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

const session = (id, target, extra = {}) => ({ subagent_id: id, recommended_use: `${id} notes`, route: { kind: 'agent_session', target_id: target }, ...extra });
const api = (id, target, extra = {}) => ({ subagent_id: id, recommended_use: `${id} notes`, route: { kind: 'api_model', target_id: target }, ...extra });
const ROSTER = () => [
    session('reviewer', 'codex=gpt-test', { review_eligible: true }),
    api('recheck', 'deepseek/deepseek-v5', { delivery: 'packet', rechecks_flags: true }),
    api('scout', 'openai::gpt-6-luna'),
];

// The settings server's roster half: the stored list, its token, and one queue of scripted answers.
function fakeServer(env, { items = ROSTER(), saved = true, meta = {}, delay = 0 } = {}) {
    const server = { roster: { enabled: true, items }, saved, version: 0, posts: [], generic: [], answers: [], reads: 0, delay,
        meta, plain: { OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: '', OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT: 6, OUROBOROS_MAX_SUBAGENT_DEPTH: 3,
            OUROBOROS_SUBAGENT_WORKTREE_ROOT: '', OUROBOROS_SUBAGENT_PROJECTS_ROOT: '' } };
    server.token = () => `roster@${server.version}`;
    server.agent = (id) => server.roster.items.find((item) => item.subagent_id === id);
    server.document = () => ({
        [KEY]: server.saved ? JSON.stringify(server.roster) : '', ...server.plain,
        OUROBOROS_RUNTIME_MODE: 'pro',
        _meta: { tokens: { [KEY]: server.token() }, effects: { [KEY]: 'next_task', OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: 'next_task' },
            available_subagents: { source: server.saved ? 'configured' : 'undecided', candidate: server.saved ? null : server.roster }, ...server.meta },
    });
    server.accept = (body) => {
        const roster = body.settings[KEY];
        const pool = roster.items.filter((item) => item.review_eligible === true && item.enabled !== false);
        if (roster.items.length && !pool.length && body.allow_empty_review_pool !== true) {
            return response({ saved: false, code: 'empty_review_pool', error: EMPTY_POOL }, 400);
        }
        Object.assign(server, { roster: structuredClone(roster), saved: true, version: server.version + 1 });
        return response({ ok: true, changes: { [KEY]: { withheld: 'structure', token: server.token() } } });
    };
    env.routes.set('/api/settings', async (_url, init = {}) => {
        if (server.delay) await new Promise((resolve) => setTimeout(resolve, server.delay));
        if (init.method !== 'POST') { server.reads += 1; return response(server.document()); }
        const body = JSON.parse(init.body);
        server.generic.push(body);
        const changes = Object.fromEntries(Object.keys(body).filter((key) => key in server.plain)
            .map((key) => [key, { old: server.plain[key], new: body[key], token: `${key}@${server.generic.length}` }]));
        Object.assign(server.plain, Object.fromEntries(Object.entries(changes).map(([key, change]) => [key, change.new])));
        return response({ status: 'saved', changes });
    });
    env.routes.set(CHECKER, async (_url, init) => {
        const body = JSON.parse(init.body);
        server.posts.push({ body, headers: init.headers });
        if (body.if_unchanged?.[KEY] !== server.token()) {
            return response({ saved: false, code: 'settings_changed_elsewhere', error: 'The agents list changed elsewhere.' }, 409);
        }
        return (server.answers.shift() || server.accept)(body);
    });
    env.routes.set('/api/review-pool', response({ pool: [], last_executions: {}, row_costs: {} }));
    return server;
}

// The app's shell in one function: mount the registered page for a path into a body, as settings.js does.
async function open(t, path = 'agents', options = {}) {
    forgetSettingRows();
    const env = options.env || setup(t);
    const server = options.server || fakeServer(env, options);
    if (!options.env) {
        // The shared status store and the model catalogue read by themselves; neither is this page's subject.
        const fetch = globalThis.fetch;
        globalThis.fetch = async (url, init) => (options.catalog && /^\/api\/model-catalog/.test(url) ? options.catalog(url)
            : options.status && /^\/api\/claudexor\/status/.test(url) ? response(options.status)
            : /^\/api\/(claudexor|model-catalog)/.test(url) ? response({}, 503) : fetch(url, init));
        const createElement = env.document.createElement;
        env.document.createElement = (tag) => {
            const node = createElement(tag);
            if (tag === 'template') node.content = node;  // a template's content is where its markup is parsed
            return node;
        };
    }
    const container = new NodeStub();
    env.mount.appendChild(container);
    const asked = [];
    const answers = options.answers || [];
    const opened = [];
    const ctx = { readSettings, openSettings: (to) => { opened.push(to); return true; },
        ask: async (question) => { asked.push(question); return answers.length ? answers.shift() : true; } };
    const handle = settingsPageFor(path).mount(container, { path, row: '', view: options.view || '', ctx });
    await settle();
    const head = (id) => container.querySelector(`[data-entry-id="${id}"]`);
    return { env, server, container, asked, opened, handle, head,
        heads: () => container.querySelectorAll('[data-entry-id]').map((node) => node.dataset.entryId),
        toggle: (id) => head(id).querySelector('[data-entry-switch]'),
        result: () => (container.querySelector('[data-entry-result]') || container.querySelector('[data-editor-result]')).textContent,
        async press(input, checked) {
            input.checked = checked;
            await input.fire('change', { target: input });
            await settle();
        },
        async shown() {
            await env.window.fire('ouro:settings-page-shown', { detail: { path } });
            await settle();
        } };
}

test('the Agents page is one registered family: the list, an agent\'s page, a new agent', () => {
    const page = settingsPageFor('agents');
    assert.equal(page, settingsPageFor('agents/reviewer'));
    assert.equal(page, settingsPageFor('agents/new'));
    assert.equal(page.title('agents'), 'Agents');
    assert.equal(page.title('agents/new'), 'New agent');
    // The roster is one of the checker's keys: its row names the checker's writer, and its page draws it by hand.
    const roster = ROWS.find((row) => row.id === KEY);
    assert.ok(SAFETY_MODEL_KEYS.includes(KEY));
    assert.deepEqual([roster.writer, roster.control.kind, roster.page, roster.fold], ['checker', 'custom', 'agents', false]);
    assert.ok(roster.aliases.includes('Delegation'));
    assert.deepEqual(rowsOf('agents', { fold: true }).map((row) => row.keys[0]), ['OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS',
        'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 'OUROBOROS_SUBAGENT_WORKTREE_ROOT',
        'OUROBOROS_SUBAGENT_PROJECTS_ROOT']);
    assert.equal(HAND_WRITTEN.some((row) => row.keys.includes(KEY)), false, 'one key, one control: the roster is described once');
});

test('an agent is named by its model in short and the facets its handle spells', () => {
    assert.equal(agentName(api('a', 'deepseek/deepseek-v5')), 'deepseek-v5');
    assert.equal(agentName(api('a', 'openai::gpt-6-luna', { effort: 'high' })), 'gpt-6-luna/high');
    assert.equal(agentName(session('a', 'codex=gpt-test', { access: 'readonly', route: { kind: 'agent_session', target_id: 'codex=gpt-test', credential_profile_id: 'work' } })),
        'gpt-test/readonly/@work');
    assert.equal(agentName(session('a', 'codex')), 'codex', 'a session on its engine\'s default model is named by the engine');
    assert.equal(agentName(api('a', '')), 'New agent');
});

test('the list draws one head per agent with its switch, and the roster\'s own switch above it', async (t) => {
    const page = await open(t);
    assert.deepEqual(page.heads(), ['reviewer', 'recheck', 'scout']);
    assert.deepEqual(page.heads().map((id) => isOn(page.toggle(id))), [true, true, true]);
    assert.match(page.head('reviewer').querySelector('.ui-row-title').textContent, /^gpt-testReviewer/);
    assert.equal(page.head('scout').querySelector('.ui-row-meta').textContent, 'API · OpenAI');
    assert.equal(page.head('scout').querySelector('.agents-head-use').textContent, 'scout notes');
    assert.equal(page.head('scout').querySelectorAll('textarea').length, 0, 'a head holds no form');
    assert.equal(page.container.querySelectorAll('[data-subagent-field="recommended_use"]').length, 0);
    const delegation = page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-control]');
    assert.equal(isOn(delegation), true);
    assert.equal(page.container.querySelector('.agents-card-title').textContent, '3 agents');
    assert.equal(page.server.posts.length, 0, 'showing the list writes nothing');
    // A head opens its agent's page; Add opens an unsaved one.
    await page.head('scout').querySelector('[data-subagent-field="open"]').fire('click', {});
    await page.container.querySelector('[data-subagent-add]').fire('click', {});
    assert.deepEqual(page.opened, ['agents/scout', 'agents/new']);
});

test('the list\'s first open draws an OpenRouter key\'s own rates and each program\'s mark from its scoped reads', async (t) => {
    const asked = [];
    const priced = (items) => response({ read_state: 'ok', items, errors: [], model_sources: [] });
    const answers = {
        'provider_id=openrouter': priced([{ value: 'vendor/model-a', price: { input: 3, output: 15 } },
            { value: 'vendor/model-b', price: { input: 1, output: 2 } }, { value: 'vendor/model-c', price_condition: 'off_peak_usd' }]),
        // The same model id on a named key is that key's entry, at that key's price.
        'provider_id=openrouter%40work': priced([{ value: 'openrouter@work::vendor/model-a', price: { input: 8, output: 9 } }]),
        'harness_id=agy': response({ read_state: 'ok', items: [], errors: [], session_catalog: { models: [{ id: 'claude-x' }] } }),
        'view=sources': response({ read_state: 'ok', sources_read_state: 'ok', items: [], errors: [],
            model_sources: [{ id: 'agy-models', label: 'Antigravity models', credentialHarness: 'agy' }] }),
    };
    const items = [api('a', 'vendor/model-a'), api('b', 'openrouter::vendor/model-b', { review_eligible: true }),
        api('c', 'vendor/model-c'), api('w', 'openrouter@work::vendor/model-a'), api('direct', 'openai::gpt-x'),
        session('s', 'agy=claude-x'), api('sub', 'claudexor::agy-models=claude-x')];
    const meta = { setup_contract: { scopedModelCatalogs: true }, connections: [{ id: 'openrouter', configured: true },
        { id: 'openrouter@work', kind: 'openrouter', name: 'work', configured: true, display: 'OpenRouter · work' },
        { id: 'openai', configured: true }] };
    const page = await open(t, 'agents', { items, meta, catalog: (url) => {
        const query = url.split('?')[1] || '';
        asked.push(query);
        return answers[query] || response({}, 503);
    } });
    // One read per key or program a head needs, never a model list per row, another provider's or a subscription's.
    const counts = () => asked.reduce((all, query) => ({ ...all, [query]: (all[query] || 0) + 1 }), {});
    assert.deepEqual(counts(), { 'harness_id=agy': 1, 'provider_id=openrouter': 1, 'provider_id=openrouter%40work': 1, 'view=sources': 1 });
    const rate = (id) => page.head(id).querySelector('[data-subagent-rate]');
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens');
    assert.equal(rate('b').textContent, '$1 in / $2 out per 1M tokens', 'two agents of one key, each its own model\'s price');
    assert.equal(rate('w').textContent, '$8 in / $9 out per 1M tokens', 'an equal model id on another key keeps that key\'s price');
    assert.equal(rate('a').dataset.i18nFmt, 'model price');
    assert.equal(page.head('a').querySelector('.ui-row-meta').textContent, 'API · OpenRouter · $3 in / $15 out per 1M tokens');
    for (const id of ['c', 'direct', 's', 'sub']) assert.equal(rate(id), null, `${id}: a condition, an unread key or a seat says no figure`);
    // The description is the owner's own words, never interface text; the program behind a seat is its mark.
    assert.equal(page.head('a').querySelector('.agents-head-use').hasAttribute('data-i18n-authored'), true);
    // The source label (product text + the user connection name) is authored, never a translation key; the rate stays
    // one `model price` template beside it.
    const metaLabel = page.head('a').querySelector('[data-i18n-authored]');
    assert.match(metaLabel.textContent, /OpenRouter/, 'the connection name lives in the authored span, out of translation');
    assert.equal(page.head('a').querySelector('[data-subagent-rate]').hasAttribute('data-i18n-fmt'), true);
    const mark = (id) => page.head(id).querySelector('[data-harness-identity]').dataset.harnessIdentity;
    assert.deepEqual(['s', 'sub', 'a'].map(mark), ['agy', 'agy', 'api']);
    const box = (id) => page.head(id).querySelector('.harness-identity-mark').getAttribute('viewBox');
    assert.deepEqual(['s', 'sub', 'a'].map(box), ['0 0 180 180', '0 0 180 180', '0 0 24 24'], 'Antigravity\'s own mark; an API key the neutral one');
    // A failed refresh keeps the key's rows for the chooser, never as the price now; a successful one brings it back.
    const good = answers['provider_id=openrouter'];
    answers['provider_id=openrouter'] = response({}, 503);
    await page.shown();
    assert.deepEqual([rate('a'), rate('b')], [null, null]);
    assert.equal(rate('w').textContent, '$8 in / $9 out per 1M tokens', 'another key\'s read is its own');
    answers['provider_id=openrouter'] = good;
    await page.shown();
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens');
    assert.deepEqual(counts(), { 'harness_id=agy': 3, 'provider_id=openrouter': 3, 'provider_id=openrouter%40work': 3, 'view=sources': 3 },
        'each showing reads each once');
    // An answer of the named key that lists the built-in spelling prices neither row: its read is the named key's only.
    const work = answers['provider_id=openrouter%40work'];
    answers['provider_id=openrouter%40work'] = priced([{ value: 'vendor/model-a', price: { input: 99, output: 99 } }]);
    await page.shown();
    answers['provider_id=openrouter%40work'] = work;
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens');
    assert.equal(rate('w'), null);
    await page.shown();
    assert.equal(rate('w').textContent, '$8 in / $9 out per 1M tokens');
    // A key removed meanwhile keeps no price, even while its earlier read is still held.
    page.server.meta.connections = meta.connections.filter((entry) => entry.id !== 'openrouter@work');
    await page.shown();
    assert.equal(rate('w'), null);
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens');
    // Another reader's global catalogue update is not this core's key read: it costs no scoped query and moves no price.
    const before = counts();
    modelCatalog.adopt({ read_state: 'ok', errors: [], items: [{ value: 'vendor/model-a', price: { input: 99, output: 99 } }] });
    await settle();
    assert.deepEqual(counts(), before, 'a global update asks nothing of this core');
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens', 'the key\'s own read is still the price');
});

test('a sign-in renews the list\'s session verdicts, never its source descriptors or its keys\' rates', async (t) => {
    const asked = [];
    const fetch = globalThis.fetch;
    globalThis.fetch = async (url) => { asked.push(String(url).split('?')[1]); return response({ read_state: 'ok', items: [], errors: [], model_sources: [] }); };
    t.after(() => { globalThis.fetch = fetch; });
    const rows = [api('a', 'vendor/model-a'), session('s', 'agy=claude-x'), api('sub', 'claudexor::agy-models=claude-x')];
    const catalog = createAgentModelCatalog({ state: { providers: [{ id: 'openrouter' }], modelSources: [] }, saved: true,
        rows: () => rows, apiCatalogAllowed: () => true, supportsScopes: () => true, changed() {}, adoptSources() {} });
    catalog.refresh({ list: true, accounts: true });
    await settle();
    assert.deepEqual(asked, ['harness_id=agy']);
    catalog.refresh({ list: true });
    await settle();
    assert.deepEqual(asked.slice(1).sort(), ['harness_id=agy', 'provider_id=openrouter', 'view=sources']);
});

test('a legacy core: the list follows the one catalogue read — a failed refresh hides the price, a new one brings it back', async (t) => {
    const asked = [];
    const items = [api('a', 'vendor/model-a'), api('w', 'openrouter@work::vendor/model-b')];
    // A core without scoped catalogues: one full read answers every head.
    const meta = { connections: [{ id: 'openrouter', configured: true },
        { id: 'openrouter@work', kind: 'openrouter', name: 'work', configured: true, display: 'OpenRouter · work' }] };
    // What a refresh event hands the window's catalogue store: the read outcome, with or without rows.
    const figures = [{ value: 'vendor/model-a', price: { input: 3, output: 15 } },
        { value: 'openrouter@work::vendor/model-b', price: { input: 8, output: 9 } }];
    const full = (read_state, errors = [], rows = figures) => modelCatalog.adopt({ read_state, errors,
        items: read_state === 'ok' ? rows : undefined });
    full('ok');
    t.after(() => modelCatalog.adopt({ read_state: 'ok', errors: [], items: [] }));
    const page = await open(t, 'agents', { items, meta, catalog: (url) => { asked.push(String(url)); return response({}, 503); } });
    const rate = (id) => page.head(id).querySelector('[data-subagent-rate]');
    const label = (id) => page.head(id).querySelector('.ui-row-meta').textContent;
    assert.equal(rate('a').textContent, '$3 in / $15 out per 1M tokens', 'the head shows the rate the one full read published');
    assert.equal(rate('w').textContent, '$8 in / $9 out per 1M tokens', 'a named key\'s own entry, at that key\'s own rate');
    assert.equal(label('w'), 'API · OpenRouter · work · $8 in / $9 out per 1M tokens', 'the key\'s own name is the label');
    assert.deepEqual(asked, [], 'the list reads no catalogue of its own');
    // The refresh fails: the price goes at once; the head, its label and the rows the last read listed stay.
    full('transport', [{ provider_id: 'catalog', error: 'offline' }]);
    await settle();
    assert.equal(rate('a'), null, 'a failed refresh says no price');
    assert.equal(rate('w'), null);
    assert.equal(label('a'), 'API · OpenRouter', 'the label stays; only the figure goes');
    assert.deepEqual(page.heads(), ['a', 'w'], 'the agents stay on the list');
    assert.deepEqual(modelCatalog.catalog().items.map((item) => item.value), ['vendor/model-a', 'openrouter@work::vendor/model-b'],
        'the rows the last good read listed stay for a chooser, never as the price now');
    // A new success brings the price back, at the figures it now publishes.
    full('ok', [], [{ value: 'vendor/model-a', price: { input: 4, output: 16 } },
        { value: 'openrouter@work::vendor/model-b', price: { input: 1, output: 2 } }]);
    await settle();
    assert.equal(rate('a').textContent, '$4 in / $16 out per 1M tokens');
    assert.equal(rate('w').textContent, '$1 in / $2 out per 1M tokens');
    assert.deepEqual(asked, [], 'still no catalogue read from the list');
});

test('a background rate arrival patches the mounted list in place: the price appears, the row, the control and the focus stay', async (t) => {
    const items = [api('a', 'vendor/model-a'), api('w', 'openrouter@work::vendor/model-b')];
    const meta = { connections: [{ id: 'openrouter', configured: true },
        { id: 'openrouter@work', kind: 'openrouter', name: 'work', configured: true, display: 'OpenRouter · work' }] };
    modelCatalog.adopt({ read_state: 'ok', errors: [], items: [] });
    t.after(() => modelCatalog.adopt({ read_state: 'ok', errors: [], items: [], model_sources: [] }));
    const page = await open(t, 'agents', { items, meta, catalog: () => response({}, 503) });
    const head = page.head('a');
    const keep = page.container.querySelector('[data-review-pool-keep-one]');
    const origin = page.container.querySelector('[data-review-pool-history]');
    keep.focus();
    origin.open = true;
    assert.equal(page.head('a').querySelector('[data-subagent-rate]'), null, 'no price before the read lands');
    // The window's one catalogue read lands: the price arrives in the background.
    modelCatalog.adopt({ read_state: 'ok', errors: [], items: [{ value: 'vendor/model-a', price: { input: 3, output: 15 } }] });
    await settle();
    assert.equal(page.head('a').querySelector('[data-subagent-rate]').textContent, '$3 in / $15 out per 1M tokens');
    assert.equal(page.head('a'), head, 'the row that was drawn is the row that is shown');
    assert.equal(page.container.querySelector('[data-review-pool-keep-one]'), keep, 'the control keeps its place');
    assert.equal(page.container.querySelector('[data-review-pool-history]'), origin, 'the disclosure is the same one');
    assert.equal(origin.open, true, 'and it stays open');
    assert.equal(globalThis.document.activeElement, keep, 'the caret is not taken away');
    // A tick that changes nothing keeps all of it: the price, the row, the control, the disclosure and the caret.
    await page.shown();
    assert.equal(page.head('a').querySelector('[data-subagent-rate]').textContent, '$3 in / $15 out per 1M tokens');
    assert.equal(page.head('a'), head, 'the row that was drawn is the row that is shown');
    assert.equal(page.container.querySelector('[data-review-pool-keep-one]'), keep, 'the control keeps its place');
    assert.equal(page.container.querySelector('[data-review-pool-history]'), origin, 'the disclosure is the same one');
    assert.equal(origin.open, true, 'and it stays open');
    assert.equal(globalThis.document.activeElement, keep, 'the caret is not taken away');
    // A key removed meanwhile takes its price away, and the list around it stays where it is.
    page.server.meta.connections = meta.connections.filter((entry) => entry.id !== 'openrouter');
    await page.shown();
    assert.equal(page.head('a').querySelector('[data-subagent-rate]'), null, 'a key that is gone says no price');
    assert.equal(page.container.querySelector('[data-review-pool-keep-one]'), keep, 'the control keeps its place');
    assert.equal(page.container.querySelector('[data-review-pool-history]'), origin, 'the disclosure is the same one');
    assert.equal(origin.open, true, 'and it stays open');
});

test('a late session verdict reaches the mounted list: the warning, the Unavailable count and the row itself', async (t) => {
    // A late session verdict reaches the mounted list: the warning, the Unavailable count and the row itself
    // — including negative -> positive Unavailable filter existence and exact roster-ordered filtered insertion.
    const status = {
        daemon: { state: 'running' },
        harnesses: [{ id: 'agy', display_name: 'Antigravity' }, { id: 'cursor', display_name: 'Cursor' }],
        profiles: { profiles: [
            { profile: { harness_id: 'agy', profile_id: 'agy-1', enabled: true }, status: { verification: 'passed' } },
            { profile: { harness_id: 'cursor', profile_id: 'cur-1', enabled: true }, status: { verification: 'passed' } },
        ] },
        quota: [],
        reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
    };
    const fetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = fetch; void claudexorStatus.refresh(); });

    // Negative -> positive Unavailable filter witness from All view (inactive zero-count chip omitted until verdict arrives).
    const fromAll = await open(t, 'agents', {
        items: [api('a', 'vendor/model-a'), session('s', 'agy=claude-x')], status,
        meta: { setup_contract: { scopedModelCatalogs: true }, connections: [] },
        catalog: async (url) => {
            if (url.includes('harness_id=agy')) await new Promise((resolve) => setTimeout(resolve, 40));
            return response({ read_state: 'ok', items: [], errors: [], session_catalog: { models: [{ id: 'claude-other' }] } });
        },
    });
    assert.equal(fromAll.container.querySelector('[data-agents-filter="unavailable"]'), null, 'inactive zero-count Unavailable chip is omitted');
    const allChip = fromAll.container.querySelector('[data-agents-filter="all"]');
    allChip.focus();
    await new Promise((resolve) => setTimeout(resolve, 140));
    await settle();
    assert.equal(fromAll.container.querySelector('[data-agents-filter="all"]'), allChip, 'surviving filter retains its element');
    assert.equal(globalThis.document.activeElement, allChip, 'surviving filter retains focus');
    const appearedChip = fromAll.container.querySelector('[data-agents-filter="unavailable"]');
    assert.ok(appearedChip, 'Unavailable filter chip appears once verdict turns positive');
    assert.equal(appearedChip.textContent, 'Unavailable 1');
    await appearedChip.fire('click', {});
    assert.deepEqual(fromAll.heads(), ['s']);
    fromAll.handle.unmount();

    // Initial view=unavailable: empty before agy settles, then ['s'] with Unavailable 1.
    const single = await open(t, 'agents', {
        view: 'unavailable', items: [session('s', 'agy=claude-x')], status,
        meta: { setup_contract: { scopedModelCatalogs: true }, connections: [] },
        catalog: async (url) => {
            if (url.includes('harness_id=agy')) await new Promise((resolve) => setTimeout(resolve, 40));
            return response({ read_state: 'ok', items: [], errors: [], session_catalog: { models: [{ id: 'claude-other' }] } });
        },
    });
    assert.deepEqual(single.heads(), [], 'the model list is not read yet, so nothing is unavailable yet');
    await new Promise((resolve) => setTimeout(resolve, 140));
    await settle();
    assert.deepEqual(single.heads(), ['s'], 'the row that became unavailable joins the list it is filtered by');
    assert.equal(single.container.querySelector('[data-agents-filter="unavailable"]').textContent, 'Unavailable 1');
    single.handle.unmount();

    // Selected zero-count filter is kept, and a late Antigravity verdict inserts in roster order without re-entry.
    const page = await open(t, 'agents', {
        view: 'unavailable',
        items: [session('s', 'agy=claude-x'), session('last', 'cursor=grok-missing')], status,
        meta: { setup_contract: { scopedModelCatalogs: true }, connections: [] },
        catalog: async (url) => {
            if (url.includes('harness_id=agy')) await new Promise((resolve) => setTimeout(resolve, 40));
            return response({ read_state: 'ok', items: [], errors: [], session_catalog: { models: [{ id: 'claude-other' }] } });
        },
    });
    const survivingLast = page.head('last');
    assert.deepEqual(page.heads(), ['last'], 'only the already-unavailable trailing row shows before agy settles');
    await new Promise((resolve) => setTimeout(resolve, 140));
    await settle();
    assert.deepEqual(page.heads(), ['s', 'last'], 'late unavailable row inserts in roster order');
    assert.equal(page.head('last'), survivingLast, 'surviving filtered row keeps its element');
    assert.match(page.head('s').querySelector('.ui-row-title').textContent, /Unavailable/, 'the head says it');
    const chip = page.container.querySelector('[data-agents-filter="unavailable"]');
    assert.equal(chip.textContent, 'Unavailable 2', 'the filter says how many');
    assert.equal(chip.getAttribute('aria-pressed'), 'true', 'and it is the one shown');
    page.handle.unmount();

    // Selected zero-count filter chip stays rendered with count 0 when its view is active.
    const emptySelected = await open(t, 'agents', { view: 'unavailable', items: [api('a', 'vendor/model-a')] });
    const zeroChip = emptySelected.container.querySelector('[data-agents-filter="unavailable"]');
    assert.ok(zeroChip, 'selected zero-count filter stays present');
    assert.equal(zeroChip.textContent, 'Unavailable 0');
    assert.equal(zeroChip.getAttribute('aria-pressed'), 'true');
    assert.equal(emptySelected.container.querySelector('[data-agents-filter="reviewers"]'), null, 'inactive zero-count filter is omitted');
});

test('no-op pool tick and review-pool chrome arrival preserve Add, Keep one per model, filter and open Reviewer origin', async (t) => {
    const env = setup(t);
    const items = [api('r1', 'vendor/model-a', { review_eligible: true }), api('r2', 'vendor/model-a', { review_eligible: true })];
    const server = fakeServer(env, { items });
    let poolPayload = { pool: [], last_executions: {}, row_costs: {}, migration: { outcome: 'converted', source: 'file', snapshot: 'snap-1' } };
    env.routes.set('/api/review-pool', () => response(poolPayload));
    const page = await open(t, 'agents', { env, server });
    const origin = page.container.querySelector('[data-review-pool-history]');
    const addBtn = page.container.querySelector('[data-subagent-add]');
    const keepBtn = page.container.querySelector('[data-review-pool-keep-one]');
    const filterBtn = page.container.querySelector('[data-agents-filter="all"]');
    assert.equal(origin.hidden, false);
    origin.open = true;
    addBtn.focus();
    await page.shown();
    assert.equal(page.container.querySelector('[data-review-pool-history]'), origin);
    assert.equal(origin.open, true, 'open Reviewer origin stays open on no-op pool tick');
    assert.equal(page.container.querySelector('[data-subagent-add]'), addBtn);
    assert.equal(globalThis.document.activeElement, addBtn, 'Add keeps its element and focus');
    assert.equal(page.container.querySelector('[data-review-pool-keep-one]'), keepBtn);
    assert.equal(page.container.querySelector('[data-agents-filter="all"]'), filterBtn);
    filterBtn.focus();
    poolPayload = { ...poolPayload, pool_without_credentials: ['r1'] };
    await page.shown();
    assert.match(page.container.querySelector('[data-review-pool-note]').textContent, /r1/);
    assert.equal(page.container.querySelector('[data-agents-filter="all"]'), filterBtn);
    assert.equal(globalThis.document.activeElement, filterBtn, 'surviving filter keeps focus on chrome update');
    assert.equal(origin.open, true, 'open Reviewer origin stays open on non-row chrome update');
});

test('translated chrome and live in-flight switch differences do not force full repaint or clobber translations', async (t) => {
    const env = setup(t);
    const items = [api('r1', 'vendor/model-a', { review_eligible: true }), api('r2', 'vendor/model-a', { review_eligible: true })];
    const server = fakeServer(env, { items });
    let poolPayload = { pool: [], last_executions: {}, row_costs: {}, migration: { outcome: 'converted', source: 'file', snapshot: 'snap-1' } };
    env.routes.set('/api/review-pool', () => response(poolPayload));
    const page = await open(t, 'agents', { env, server });
    const origin = page.container.querySelector('[data-review-pool-history]');
    const keepBtn = page.container.querySelector('[data-review-pool-keep-one]');
    const addBtn = page.container.querySelector('[data-subagent-add]');
    const filterBtn = page.container.querySelector('[data-agents-filter="all"]');
    const titleEl = page.container.querySelector('.agents-card-title');
    const copiesText = page.container.querySelector('[data-review-pool-copies-text]');
    const rowTitle = page.head('r1').querySelector('.ui-row-title');
    const rowSwitch = page.toggle('r1');
    // Simulate i18n MutationObserver localization and an in-flight live checkbox toggle.
    titleEl.textContent = '2 агента';
    filterBtn.textContent = 'Все 2';
    copiesText.textContent = 'Переведённый текст копий';
    rowTitle.textContent = 'переведённый-заголовок';
    rowSwitch.setAttribute('aria-label', 'Доступен: model-a');
    rowSwitch.checked = false;
    origin.open = true;
    keepBtn.focus();

    poolPayload = { ...poolPayload, pool_without_credentials: ['r1'] };
    await page.shown();
    assert.match(page.container.querySelector('[data-review-pool-note]').textContent, /r1/);
    assert.equal(page.container.querySelector('[data-review-pool-keep-one]'), keepBtn);
    assert.equal(globalThis.document.activeElement, keepBtn, 'Keep one per model keeps element and focus');
    assert.equal(page.container.querySelector('[data-subagent-add]'), addBtn);
    assert.equal(page.container.querySelector('[data-agents-filter="all"]'), filterBtn);
    assert.equal(page.container.querySelector('[data-review-pool-history]'), origin);
    assert.equal(origin.open, true, 'open Reviewer origin stays open');
    assert.equal(titleEl.textContent, '2 агента', 'translated card title stays intact');
    assert.equal(filterBtn.textContent, 'Все 2', 'translated unchanged filter text stays intact');
    assert.equal(copiesText.textContent, 'Переведённый текст копий', 'translated unchanged copies text stays intact');
    assert.equal(rowTitle.textContent, 'переведённый-заголовок', 'translated unchanged row title stays intact');
    assert.equal(rowSwitch.getAttribute('aria-label'), 'Доступен: model-a', 'translated unchanged switch aria-label stays intact');
});

test('discovery-only automatic chrome change falls back to full paint without stale state.html cache', async (t) => {
    const { createAvailableSubagentsEditor } = await import('../modules/subagents_settings.js');
    const env = setup(t);
    const createElement = env.document.createElement;
    env.document.createElement = (tag) => {
        const node = createElement(tag);
        if (tag === 'template') node.content = node;
        return node;
    };
    const host = new NodeStub();
    host.id = 'agents-chrome-test';
    env.mount.appendChild(host);
    const editor = createAvailableSubagentsEditor({
        hostId: 'agents-chrome-test',
        page: { entry: '', ctx: { ask: async () => true, openSettings: () => true }, reload: async () => {}, shown: () => true, filter: 'all' },
    });
    editor.mount({ bindStatus: false });
    const setting = { enabled: true, items: [api('a', 'vendor/model-a')] };
    editor.load(setting, { source: 'undecided', automatic: false });
    assert.ok(host.querySelector('[data-agents-automatic]'), 'automatic notice shown initially');
    // Same items with source changing to configured triggers discoveryOnly=true; outer chrome must update.
    editor.applyGeneratedPreview({ available_subagents: setting, source: 'configured', diagnostics: [] });
    assert.equal(host.querySelector('[data-agents-automatic]'), null, 'automatic notice removed on chrome fallback');
    editor.destroy();
});

test('an agent\'s page names its source from a descriptor that arrives late, and the form keeps what the owner typed', async (t) => {
    const items = [api('sub', 'claudexor::agy-models=claude-x')];
    const meta = { setup_contract: { scopedModelCatalogs: true }, connections: [] };
    const page = await open(t, 'agents/sub', { items, meta, catalog: async (url) => {
        if (url.includes('view=sources')) await new Promise((resolve) => setTimeout(resolve, 40));
        return response({ read_state: 'ok', items: [], errors: [], model_sources: url.includes('view=sources')
            ? [{ id: 'agy-models', label: 'Antigravity models', credentialHarness: 'agy' }] : [] });
    } });
    const lead = () => page.container.querySelector('.settings-page-lead').textContent;
    assert.match(lead(), /agy-models/, 'before the descriptor the source names its own id');
    const purpose = page.container.querySelector('[data-subagent-field="recommended_use"]');
    purpose.value = 'my own words';
    await purpose.fire('input', { target: purpose });
    await new Promise((resolve) => setTimeout(resolve, 140));
    await settle();
    assert.match(lead(), /Antigravity models/, 'the descriptor that arrived late names the program');
    assert.equal(page.container.querySelector('[data-subagent-field="recommended_use"]').value, 'my own words',
        'the form keeps what the owner typed');
});

test('a legacy core takes the source descriptors from the one catalogue read: the program\'s own mark, with no second read', async (t) => {
    const asked = [];
    const items = [api('sub', 'claudexor::agy-models=claude-x')];
    modelCatalog.adopt({ read_state: 'ok', errors: [], items: [] });
    t.after(() => modelCatalog.adopt({ read_state: 'ok', errors: [], items: [], model_sources: [] }));
    const page = await open(t, 'agents', { items, meta: { connections: [] },
        catalog: (url) => { asked.push(String(url)); return response({}, 503); } });
    const mark = () => page.head('sub').querySelector('[data-harness-identity]').dataset.harnessIdentity;
    assert.equal(mark(), '', 'without a descriptor the opaque source id names no program');
    assert.match(page.head('sub').querySelector('[data-subagent-identity]').textContent, /agy-models/);
    // The one full read lands and names the source: the program's own mark follows it, with no second read.
    modelCatalog.adopt({ read_state: 'ok', errors: [], items: [],
        model_sources: [{ id: 'agy-models', label: 'Antigravity models', credentialHarness: 'agy' }] });
    await settle();
    assert.equal(mark(), 'agy', 'the descriptor earns the program\'s own mark');
    assert.match(page.head('sub').querySelector('[data-subagent-identity]').textContent, /Antigravity models/);
    assert.deepEqual(asked, [], 'the list reads no catalogue of its own');
});

test('a page left while its settings read is out starts no catalogue read of its own', async (t) => {
    const asked = [];
    const meta = { setup_contract: { scopedModelCatalogs: true }, connections: [{ id: 'openrouter', configured: true }] };
    const page = await open(t, 'agents', { items: [api('a', 'vendor/model-a')], meta, delay: 60,
        catalog: (url) => { asked.push(String(url)); return response({}, 503); } });
    page.handle.unmount();
    await new Promise((resolve) => setTimeout(resolve, 140));
    assert.deepEqual(asked, [], 'the page that was left asks for no catalogue');
});

test('a door\'s view is the list\'s first filter: Reviewers opens the list on its reviewers, any other on all', async (t) => {
    const chosen = (page) => page.container.querySelectorAll('[data-agents-filter]').filter((chip) => chip.getAttribute('aria-pressed') === 'true')
        .map((chip) => chip.dataset.agentsFilter);
    const reviewers = await open(t, 'agents', { view: 'reviewers' });
    assert.deepEqual(chosen(reviewers), ['reviewers']);
    assert.deepEqual(reviewers.heads(), ['reviewer'], 'only the agents that read finished work');
    assert.equal(ROWS.find((row) => row.id === 'permissions:reviewers').control.view, 'reviewers', 'the door names its view');
    reviewers.handle.unmount();
    const plain = await open(t, 'agents', { view: 'nothing-such' });
    assert.deepEqual(chosen(plain), ['all']);
    assert.deepEqual(plain.heads(), ['reviewer', 'recheck', 'scout']);
});

test('a plain row\'s switch writes that one entry at once through the checker\'s endpoint, with the read\'s token', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.posts.length, 1);
    const { body, headers } = page.server.posts[0];
    assert.deepEqual(Object.keys(body.settings), [KEY]);
    assert.deepEqual(body.if_unchanged, { [KEY]: 'roster@0' });
    assert.equal(headers['X-Ouroboros-Surface'], APP_SETTINGS_ORIGIN['X-Ouroboros-Surface']);
    assert.deepEqual(body.settings[KEY].items, [...ROSTER().slice(0, 2), { ...ROSTER()[2], enabled: false }], 'only its entry changed');
    assert.equal(page.server.generic.length, 0, 'never the generic save: outside Cyber Pro it refuses a change of the checker');
    assert.equal(isOn(page.toggle('scout')), false);
    assert.equal(page.result(), 'Saved · applies to new tasks');
    assert.equal(page.asked.length, 0);
    // On again: the stored entry loses the flag it only carries while off.
    await page.press(page.toggle('scout'), true);
    assert.equal(page.server.posts[1].body.settings[KEY].items[2].enabled, true);
    assert.equal(isOn(page.toggle('scout')), true);
});

test('a row with the re-check mark and the roster\'s own switch take the same owner transaction', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('recheck'), false);
    assert.equal(page.server.posts.length, 1);
    assert.equal(page.server.generic.length, 0);
    assert.deepEqual(page.server.agent('recheck'), { ...ROSTER()[1], enabled: false }, 'the mark stays with its row');
    const delegation = page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-control]');
    await page.press(delegation, false);
    assert.equal(page.server.posts.length, 2);
    assert.equal(page.server.roster.enabled, false);
    assert.equal(page.server.roster.items.length, 3, 'the roster\'s own switch changes no entry');
    assert.equal(page.server.generic.length, 0);
    assert.equal(isOn(delegation), false);
    assert.equal(page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-result]').textContent,
        'Saved · applies to new tasks');
    assert.equal(page.container.querySelector('[data-review-pool-stays]').textContent, 'Subagents are off; agents marked Reviewer still review.');
});

test('the agent page owns the second-check binding and releases it when left', async (t) => {
    const page = await open(t, 'agents/recheck');
    const mark = page.container.querySelector('[data-recheck-mark="recheck"]');
    assert.ok(mark, 'the saved second-check switch participates in the shared binding');
    assert.equal(recheckSaved(), 'recheck');
    mark.checked = false;
    // The fixture dispatches capture and target separately, as the browser does in that order.
    await page.container.fire('change', { target: mark });
    assert.equal(recheckDraft(), '', 'the page itself binds the shared one-mark rule');
    await page.press(mark, false);
    assert.equal(page.server.agent('recheck').rechecks_flags, false);
    assert.equal(recheckSaved(), '', 'the completed read resets the mark from the stored roster');
    assert.equal(recheckDraft(), null);
    page.handle.unmount();
    mark.checked = true;
    await page.container.fire('change', { target: mark });
    assert.equal(recheckDraft(), null, 'a detached page cannot change the shared mark');
});

test('a toggle on a roster that already holds a twin pair is sent, and the server\'s answer is shown as it is', async (t) => {
    const items = [...ROSTER(), api('scout-twin', 'openai::gpt-6-luna')];
    const page = await open(t, 'agents', { items });
    // The server judges twins only when THIS save changes the roster: here it refuses for a pair the press never touched.
    page.server.answers.push(() => response({ saved: false, error: TWINS }, 400));
    await page.press(page.toggle('reviewer'), false);
    assert.equal(page.server.posts.length, 1, 'sent once, never retried');
    assert.equal(page.result(), `Not saved. ${TWINS}`);
    assert.equal(isOn(page.toggle('reviewer')), true, 'the switch shows what is stored');
    assert.equal(page.asked.length, 0);
    // Accepted, it is saved like any other.
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.agent('scout').enabled, false);
    assert.equal(page.result(), 'Saved · applies to new tasks');
});

test('switching the last reviewer off asks, then sends the SAME body again with the flag', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('reviewer'), false);
    assert.equal(page.asked.length, 1);
    assert.equal(page.asked[0].body, 'No agent marked Reviewer would be on, so the main model reviews alone: one model, no second opinion.');
    assert.equal(page.asked[0].title, 'Save without reviewers?');
    assert.equal(page.server.posts.length, 2);
    assert.deepEqual(page.server.posts[1].body, { ...page.server.posts[0].body, allow_empty_review_pool: true });
    assert.equal(page.server.agent('reviewer').enabled, false);
    assert.equal(isOn(page.toggle('reviewer')), false);

    // No reviewer is on now: the next changed roster is judged again and asks for ITSELF; a No writes nothing more.
    const declined = await open(t, 'agents', { env: page.env, server: page.server, answers: [false] });
    await declined.press(declined.toggle('scout'), false);
    assert.equal(declined.asked.length, 1);
    assert.equal(page.server.posts.length, 3, 'the refused body, and no second one');
    assert.equal(page.server.posts[2].body.allow_empty_review_pool, undefined, 'a confirmation is never carried to another roster');
    assert.equal(page.server.agent('scout').enabled, undefined);
    assert.equal(isOn(declined.toggle('scout')), true, 'the switch returns to what is stored');
    assert.equal(declined.result(), '');
});

test('a roster that was never saved is written whole on the first press', async (t) => {
    const page = await open(t, 'agents', { saved: false });
    assert.deepEqual(page.heads(), ['reviewer', 'recheck', 'scout'], 'the list shows the candidate the server proposes');
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.posts.length, 1);
    // The list as shown, in its canonical spelling (a session's access is written out), with the one entry changed.
    assert.deepEqual(page.server.posts[0].body.settings[KEY], { enabled: true,
        items: [{ ...ROSTER()[0], access: 'full' }, ROSTER()[1], { ...ROSTER()[2], enabled: false }] });
    assert.deepEqual(page.server.posts[0].body.if_unchanged, { [KEY]: 'roster@0' }, 'guarded by the token of the unsaved value');
    assert.equal(page.server.saved, true);
    assert.deepEqual(page.heads().map((id) => isOn(page.toggle(id))), [true, true, false]);
});

test('a list that changed elsewhere is refused, repainted, and nothing is retried', async (t) => {
    const page = await open(t);
    // Another device removes an agent and switches one off between this page's read and the write's own read...
    page.server.answers.push((body) => {
        Object.assign(page.server, { roster: { enabled: true, items: [{ ...ROSTER()[0] }, { ...ROSTER()[2], enabled: false }] }, version: 7 });
        return response({ saved: false, code: 'settings_changed_elsewhere', error: 'The agents list changed elsewhere.', sent: body.if_unchanged }, 409);
    });
    await page.press(page.toggle('recheck'), false);
    assert.equal(page.server.posts.length, 1, 'one write, no silent retry');
    assert.equal(page.result(), 'The list changed elsewhere. This is how it stands now.');
    assert.deepEqual(page.heads(), ['reviewer', 'scout'], 'the page shows the list the server holds');
    assert.equal(isOn(page.toggle('scout')), false);
    // ...and a press for an agent that is gone from the fresh read sends nothing at all.
    const stale = await open(t, 'agents', { env: page.env, server: page.server });
    page.server.roster.items.pop();
    page.server.version += 1;
    await stale.press(stale.toggle('scout'), true);
    assert.equal(page.server.posts.length, 1);
    assert.equal(stale.result(), 'The list changed elsewhere. This is how it stands now.');
    assert.deepEqual(stale.heads(), ['reviewer']);
});

test('an agent\'s page: a switch never publishes its unsaved form, and Save writes the form into its own entry only', async (t) => {
    const list = await open(t);
    const page = await open(t, 'agents/scout', { env: list.env, server: list.server });
    const field = (name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    assert.deepEqual(page.container.querySelectorAll('.agents-card-title').map((node) => node.textContent), ['In use', 'What it is good at', 'Runs on']);
    assert.equal(field('enabled').getAttribute('role'), 'switch', 'a saved agent\'s on/off acts at once');
    assert.equal(field('rechecks_flags'), null, 'only an API row with Packet delivery, or the marked one, may take the re-check');
    // The Reviewer row says what the mark means under the saved Access.
    assert.match(page.container.querySelector('.agent-page').textContent, /Counts when chosen: for each check Ouroboros asks one, several or all reviewers/);
    field('recommended_use').value = 'UNSAVED WORDS';
    await field('recommended_use').fire('input', { target: field('recommended_use') });

    // The list's switch on another agent, then this page's own Reviewer mark: neither carries the unsaved words.
    await list.press(list.toggle('recheck'), false);
    await page.press(field('review_eligible'), true);
    assert.equal(list.server.posts.length, 2);
    assert.equal(JSON.stringify(list.server.posts).includes('UNSAVED WORDS'), false);
    assert.equal(list.server.agent('scout').review_eligible, true);
    assert.equal(list.server.agent('scout').recommended_use, 'scout notes');
    assert.equal(words(field('recommended_use')), 'UNSAVED WORDS', 'the form keeps its text through the repaint');

    // Elsewhere the agent is switched off; this page's Save then keeps the stored on/off and marks, and its own text.
    Object.assign(list.server.agent('scout'), { enabled: false });
    list.server.version += 1;
    await page.container.querySelector('[data-editor-save]').fire('click', { target: page.container.querySelector('[data-editor-save]') });
    await settle();
    assert.equal(list.server.posts.length, 3);
    assert.deepEqual(list.server.agent('scout'), { ...ROSTER()[2], recommended_use: 'UNSAVED WORDS', review_eligible: true, enabled: false });
    assert.deepEqual(list.server.agent('recheck'), { ...ROSTER()[1], enabled: false }, 'no other entry is rewritten');
    assert.equal(page.result(), 'Saved · applies to new tasks');
    assert.equal(isOn(field('enabled')), false, 'the page shows the stored value after its Save');
});

test('a page that is left keeps its unsaved form for the next opening; Remove asks first and leaves the page', async (t) => {
    const first = await open(t, 'agents/scout');
    const field = (page, name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    field(first, 'recommended_use').value = 'KEPT WORDS';
    await field(first, 'recommended_use').fire('input', { target: field(first, 'recommended_use') });
    first.handle.unmount();
    const again = await open(t, 'agents/scout', { env: first.env, server: first.server, answers: [false, true] });
    assert.equal(words(field(again, 'recommended_use')), 'KEPT WORDS');
    assert.equal(first.server.posts.length, 0);

    const remove = again.container.querySelector('[data-editor-remove]');
    await remove.fire('click', { target: remove });
    await settle();
    assert.equal(again.asked[0].title, 'Remove gpt-6-luna?');
    assert.equal(first.server.posts.length, 0, 'No removes nothing');
    await remove.fire('click', { target: remove });
    await settle();
    assert.deepEqual(first.server.roster.items.map((item) => item.subagent_id), ['reviewer', 'recheck']);
    assert.deepEqual(again.opened, ['agents']);
    again.handle.unmount();
    const fresh = await open(t, 'agents', { env: first.env, server: first.server });
    assert.deepEqual(fresh.heads(), ['reviewer', 'recheck']);
});

test('a new agent is an unsaved form whose on/off is a field of it; its Save adds it and returns to the list', async (t) => {
    const page = await open(t, 'agents/new');
    const field = (name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    assert.equal(field('enabled').getAttribute('role'), null, 'a checkbox of the form, not a switch that writes');
    assert.equal(field('enabled').classList.contains('ui-checkbox'), true);
    await page.press(field('enabled'), false);
    assert.equal(page.server.posts.length, 0, 'the on/off of an unsaved agent writes nothing');
    const save = page.container.querySelector('[data-editor-save]');
    await save.fire('click', { target: save });
    await settle();
    assert.equal(page.server.posts.length, 0, 'an agent without a model is not sent');
    assert.match(page.container.querySelector('[data-subagents-validation]').textContent, /New agent needs a model or agent-session route\./);
    // The form's own listener takes the typed model (the chooser's suggestion list is a browser matter).
    field('model').value = 'qwen/qwen3.8-max';
    field('model').events.get('input')[0]({ target: field('model') });
    await save.fire('click', { target: save });
    await settle();
    assert.equal(page.server.posts.length, 1);
    const added = page.server.roster.items.at(-1);
    assert.deepEqual({ ...added, subagent_id: '' }, { subagent_id: '', recommended_use: '', route: { kind: 'api_model', target_id: 'qwen/qwen3.8-max' }, enabled: false });
    assert.equal(page.server.roster.items.length, 4);
    assert.deepEqual(page.opened, ['agents']);
});

test('a list of 50, switched-off agents included, holds Add and Duplicate, and their one hint says why; at 49 both act', async (t) => {
    const said = ['The list holds at most 50 agents, including those switched off.', 'Remove one to add another.'];
    const items = Array.from({ length: 50 }, (_, index) => api(`a${index}`, `openai/model-${index}`, index % 3 ? {} : { enabled: false }));
    // A held control: reachable (not disabled), described by its own hint, which a mouse, a finger or the keyboard shows.
    const heldWithHint = async (control) => {
        const host = control.closest('[data-full-hint]');
        const hint = host.querySelector('.ui-hint');
        assert.deepEqual([control.disabled, control.getAttribute('aria-disabled'), control.getAttribute('aria-describedby')], [false, 'true', hint.id]);
        assert.deepEqual([hint.hidden, hint.querySelector('.ui-hint-text').textContent, hint.querySelector('.ui-hint-rule').textContent], [true, ...said]);
        for (const [show, hide] of [['pointerenter', 'pointerleave'], ['focusin', 'focusout'], ['click', 'focusout']]) {
            await host.fire(show, {});
            assert.equal(hint.hidden, false, `${show} shows why`);
            await host.fire(hide, {});
            assert.equal(hint.hidden, true, `${hide} hides it`);
        }
    };
    const list = await open(t, 'agents', { items });
    const add = list.container.querySelector('[data-subagent-add]');
    assert.equal(list.heads().length, 50);
    await heldWithHint(add);
    assert.equal(list.container.querySelectorAll('.ui-hint').length, 1, 'one fact, in the hint of Add only');
    assert.equal(list.container.textContent.split('holds at most').length, 2, 'said once: in that hint, by no line of the page');
    await add.fire('click', { target: add });
    assert.deepEqual(list.opened, [], 'a held Add opens no new agent');
    list.handle.unmount();
    const full = await open(t, 'agents/a49', { env: list.env, server: list.server });
    const duplicate = full.container.querySelector('[data-subagent-duplicate]');
    await heldWithHint(duplicate);
    assert.equal(full.container.querySelectorAll('.ui-hint').length, 1, 'the page holds Duplicate only');
    await duplicate.fire('click', { target: duplicate });
    assert.deepEqual(full.opened, [], 'a full list opens no copy');
    full.handle.unmount();
    list.server.roster.items.pop();
    const room = await open(t, 'agents/a0', { env: list.env, server: list.server });
    const copy = room.container.querySelector('[data-subagent-duplicate]');
    assert.deepEqual([copy.getAttribute('aria-disabled'), room.container.querySelector('.ui-hint')], [null, null]);
    await copy.fire('click', { target: copy });
    assert.deepEqual(room.opened, ['agents/new'], 'the 50th agent is a copy away');
    room.handle.unmount();
    const fewer = await open(t, 'agents', { env: list.env, server: list.server });
    const free = fewer.container.querySelector('[data-subagent-add]');
    assert.deepEqual([free.disabled, free.getAttribute('aria-disabled'), fewer.container.querySelector('.ui-hint')], [false, null, null]);
    assert.equal(list.server.posts.length, 0, 'looking writes nothing');
});

test('Subagents may change files is a strip over the stored spelling: Auto is the unset value, and a receipt reads back in its words', async (t) => {
    const page = await open(t);
    const fold = page.container.querySelector('[data-agents-fold]');
    const row = fold.querySelector('[data-setting-row="OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS"]');
    const option = (label) => row.querySelectorAll('[data-setting-value]').find((button) => button.textContent === label);
    const pressed = () => row.querySelectorAll('[data-setting-value]').filter((button) => button.getAttribute('aria-pressed') === 'true')
        .map((button) => button.textContent);
    assert.deepEqual(row.querySelectorAll('[data-setting-value]').map((button) => button.textContent), ['Off', 'Auto', 'On']);
    assert.deepEqual(pressed(), ['Auto']);
    await fold.fire('click', { target: option('On') });
    await settle();
    assert.deepEqual(page.server.generic, [{ OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: 'true' }], 'its one key, by the generic save');
    assert.deepEqual(pressed(), ['On']);
    assert.equal(row.querySelector('[data-setting-result]').textContent, 'Saved · applies to new tasks · was Auto · Undo');
    await fold.fire('click', { target: option('Auto') });
    await settle();
    assert.deepEqual(page.server.generic[1], { OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: '' }, 'Auto is stored as nothing');
    assert.equal(row.querySelector('[data-setting-result]').textContent, 'Saved · applies to new tasks · was On · Undo');
    // A value written long ago in another spelling reads as the same choice.
    page.server.plain.OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS = 'off';
    await page.shown();
    assert.deepEqual(pressed(), ['Off']);
    assert.equal(page.server.posts.length, 0, 'the roster is no part of these rows');
});

test('At once, for one task is the owner\'s optional number: the server\'s blank is No limit, both ways', async (t) => {
    const page = await open(t);
    const key = 'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT';
    const fold = page.container.querySelector('[data-agents-fold]');
    const row = fold.querySelector(`[data-setting-row="${key}"]`);
    const field = row.querySelector('[data-setting-control]');
    const noLimit = row.querySelector('[data-setting-no-limit]');
    assert.equal(row.querySelector('[data-setting-meta]').textContent,
        'A number caps the subagents one task runs at once. No limit: Ouroboros decides; this computer\u2019s memory still limits.');
    assert.deepEqual([field.value, noLimit.getAttribute('aria-pressed')], ['6', 'false'], 'a saved number is shown as itself');
    await fold.fire('click', { target: noLimit });
    await settle();
    assert.deepEqual(page.server.generic.at(-1), { [key]: '' }, 'No limit is sent as the server\'s blank');
    assert.deepEqual([field.value, noLimit.getAttribute('aria-pressed')], ['', 'true']);
    const type = async (text) => {
        field.value = text;
        await fold.fire('input', { target: field });
        await fold.fire('change', { target: field });
        await settle();
    };
    for (const typo of ['0', '-2', '1000', 'six']) {  // a number outside 1..500 is refused here, as before No limit
        const writes = page.server.generic.length;
        await type(typo);
        assert.equal(page.server.generic.length, writes, `${typo} is not sent`);
        assert.match(row.querySelector('[data-setting-result]').textContent, /Enter a number from 1 to 500\./, typo);
    }
    await type('4');
    assert.deepEqual(page.server.generic.at(-1), { [key]: 4 });
    assert.equal(noLimit.getAttribute('aria-pressed'), 'false');
    page.server.plain[key] = '';  // a fresh install: nothing chosen reads as No limit
    await page.shown();
    assert.deepEqual([field.value, noLimit.getAttribute('aria-pressed')], ['', 'true']);
});

test('a roster write says Saved with when it applies, a refusal with the server\'s sentence, and nothing for no change', () => {
    assert.deepEqual(rosterResult({ state: 'saved', effect: 'next_task', withheld: 'structure', token: 't', message: '' }),
        { text: 'Saved · applies to new tasks', tone: 'ok', details: '' });
    assert.equal(rosterResult({ state: 'saved', effect: 'next_task', withheld: '', message: '' }).text, 'Saved', 'no receipt: no claim about when');
    assert.equal(rosterResult({ state: 'refused', effect: 'next_task', message: TWINS }).text, `Not saved. ${TWINS}`);
    assert.equal(rosterResult({ state: 'unknown', effect: 'next_task', message: 'timeout' }).text, 'Not confirmed yet');
    assert.deepEqual(rosterResult({ state: 'unchanged' }), {});
});

test('the roster\'s switch is never off before a read: inert and reading, then the stored value; opened again, the last read at once', async (t) => {
    const first = await open(t);  // the fixture, the server and the status reads answered by themselves, as for every page here
    first.handle.unmount();
    forgetSettingRows();
    const answer = first.env.routes.get('/api/settings');
    let gate = null;
    first.env.routes.set('/api/settings', async (url, init = {}) => { if (init.method !== 'POST') await gate.promise; return answer(url, init); });
    const hold = () => { let resolve; gate = { promise: new Promise((done) => { resolve = done; }) }; gate.resolve = resolve; };
    const mount = () => {
        const container = new NodeStub();
        first.env.mount.appendChild(container);
        const handle = settingsPageFor('agents').mount(container, { path: 'agents', row: '', view: '', ctx: { readSettings, openSettings: () => true, ask: async () => true } });
        t.after(() => handle.unmount());
        return container;
    };
    const facts = (container) => {
        const row = container.querySelector(`[data-setting-row="${KEY}"]`);
        const control = row.querySelector('[data-setting-control]');
        return [control.disabled, isOn(control), row.querySelector('[data-setting-meta]').textContent, container.querySelectorAll('[data-entry-id]').length];
    };
    hold();
    const cold = mount();
    assert.deepEqual(facts(cold), [true, false, 'Reading…', 0], 'nothing read: the switch says so, and no list stands');
    gate.resolve();
    await settle();
    assert.deepEqual(facts(cold), [false, true, rowById(KEY).meta, 3]);
    hold();
    const warm = mount();
    assert.deepEqual(facts(warm), [false, true, rowById(KEY).meta, 3], 'the last read, before this page\'s own read answers');
    gate.resolve();
    await settle();
    assert.deepEqual(facts(warm), [false, true, rowById(KEY).meta, 3]);
});

test('the roster\'s switch follows the reads made anywhere: its own fails while a newer one runs, then what that one and the next find', async (t) => {
    const first = await open(t);
    first.handle.unmount();
    forgetSettingRows();
    const answer = first.env.routes.get('/api/settings');
    const gates = [];
    first.env.routes.set('/api/settings', async (url, init = {}) => {
        if (init.method === 'POST') return answer(url, init);
        const gate = {};
        gate.promise = new Promise((done) => { gate.resolve = done; });
        gates.push(gate);
        return (await gate.promise) === 200 ? answer(url, init) : response({ error: 'offline' }, 500);
    });
    const container = new NodeStub();
    first.env.mount.appendChild(container);
    const handle = settingsPageFor('agents').mount(container, { path: 'agents', row: '', view: '', ctx: { readSettings, openSettings: () => true, ask: async () => true } });
    t.after(() => handle.unmount());
    const facts = () => {
        const row = container.querySelector(`[data-setting-row="${KEY}"]`);
        const control = row.querySelector('[data-setting-control]');
        return [control.disabled, isOn(control), row.querySelector('[data-setting-meta]').textContent, container.querySelectorAll('[data-entry-id]').length];
    };
    const newer = readSettings().catch(() => 'failed');
    await settle();
    gates[0].resolve(500);
    await settle();
    assert.deepEqual(facts(), [true, false, 'Reading…', 0], 'its own read failed, a newer one still runs');
    gates[1].resolve(500);
    await newer;
    assert.deepEqual(facts(), [true, false, 'Could not be read', 0]);
    assert.match(container.textContent, /could not be loaded/, 'the editor keeps its own words for a failed read');
    const elsewhere = readSettings();
    await settle();
    gates[2].resolve(200);
    await elsewhere;
    assert.deepEqual(facts(), [false, true, rowById(KEY).meta, 3], 'the read that lands is the page\'s too');
});
