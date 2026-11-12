// Settings › Accounts and keys (settings_accounts.js composes harness_accounts.js, settings_connections.js and
// settings_secrets.js; docs/DESIGN.md "Settings"): the list, a provider's page, a key's page and the stored keys, each
// mounted the way the shell mounts it, against a server that answers like the settings server with its receipt.
// What must never happen by itself is proved in both directions: showing the list reads no status, starts no daemon
// and calls no provider, while showing a provider's page reads once.
import assert from 'node:assert/strict';
import test from 'node:test';

import { claudexorStatus } from '../modules/claudexor_status_store.js';
import { accountGroups, destroyHarnessAccounts, harnessFamilyMarkup } from '../modules/harness_accounts.js';
import { otherKeyRows, providerRowFacts, quotaBars, quotaLeft } from '../modules/settings_accounts.js';
import { editorLine, keyHeads, keyPath, keyTitle, parseKeyPath, referencesLine, setKeyEnabled } from '../modules/settings_connections.js';
import { forgetSettingRows, readSettings } from '../modules/settings_controls.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { PROVIDER_CARDS, providerSettingsCard } from '../modules/settings_provider_cards.js';
import { HAND_WRITTEN, ROWS, rowById } from '../modules/settings_rows.js';
import { secretRowHtml, storedKeyRow } from '../modules/settings_secrets.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';
import { Stub } from './helpers/settings_dom.js';

// A field's value starts as its `value` attribute, as a browser's does.
Object.defineProperty(NodeStub.prototype, 'value', { configurable: true,
    get() { return this.typed ?? this.attrs.value ?? ''; }, set(text) { this.typed = String(text); } });
Object.defineProperty(NodeStub.prototype, 'lastElementChild', { configurable: true, get() { return this.children.at(-1) || null; } });
NodeStub.prototype.insertAdjacentHTML ??= function insertAdjacentHTML(_where, html) {
    const parsed = new Stub('div');
    parsed.innerHTML = html;
    [...parsed.children].forEach((child) => this.appendChild(child));
};

const settle = async (rounds = 14) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const MASK = 'sk-…mask';
const LABELS = { openrouter: 'OpenRouter', openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek', zai: 'Z.ai (GLM)', minimax: 'MiniMax',
    cloudru: 'Cloud.ru Foundation Models', gigachat: 'GigaChat', 'openai-compatible': 'OpenAI-compatible endpoint' };
const STARTS = ['/api/claudexor/status', '/api/claudexor/wake', '/api/claudexor/maintenance/harnesses', '/api/providers/test',
    '/api/model-catalog', '/api/claudexor/quota/refresh', '/api/claudexor/login', '/api/review-pool'];

const profile = (harness, id, { enabled = true, verification = 'passed' } = {}) => ({
    profile: { harness_id: harness, profile_id: id, display_name: `${id}@example.com`, enabled },
    status: { verification, verification_source: 'vendor', availability: verification === 'passed' ? 'available' : 'unavailable' },
    identity: { email: `${id}@example.com`, plan: 'pro' } });
/** A status answer with eighteen Codex accounts (one needs sign-in, one is out of rotation) and two of Claude Code. */
function statusAnswer({ daemon = { state: 'running', engine_version: '3.8.0', runtime: { state: 'ready' } }, reads = 'ok' } = {}) {
    const profiles = [
        ...Array.from({ length: 18 }, (_, i) => profile('codex', `studio-${String(i + 1).padStart(2, '0')}`,
            { enabled: i !== 4, verification: i === 2 ? 'failed' : 'passed' })),
        profile('claude', 'home'), profile('claude', 'work'),
    ];
    return { daemon, config_dir: '/tmp/home', unified_accounts: true, reads: { catalog: reads, accounts: reads, quota: reads },
        harnesses: [{ id: 'codex', display_name: 'Codex' }, { id: 'claude', display_name: 'Claude Code' }],
        profiles: { profiles: reads === 'ok' ? profiles : [] },
        quota: reads === 'ok' ? profiles.filter((row) => row.profile.harness_id === 'codex').map((row, i) => ({
            subject: { harness: 'codex', subject_id: row.profile.profile_id }, freshness: 'fresh',
            constraints: [{ id: 'window', used_ratio: i === 7 ? 1 : i / 20, resets_at: new Date(Date.now() + 3600e3).toISOString() }] })) : [],
        quota_absences: [] };
}

/** The settings server: the document as a read masks it, tokens, a receipt for every write, and a 409 for a list that
 *  changed since the token was read. Every request is recorded. */
function server(t, { values = {}, status = statusAnswer(), skills = [] } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const state = {
        calls: [], status, wakes: 0, revision: 1, askedTests: [], refuse: null, env,
        values: { OPENROUTER_API_KEY: 'sk-or-real', OPENAI_API_KEY: '', ANTHROPIC_API_KEY: '', OPENAI_BASE_URL: '', GITHUB_TOKEN: 'ghp_real',
            GITHUB_REPO: 'studio/home', TEAM_HOOK: 'https://hook', OUROBOROS_MODEL: 'openrouter::z-ai/glm-5.3',
            OUROBOROS_CONNECTIONS: [{ kind: 'openrouter', name: 'research-credits', label: 'Research credits', api_key: 'sk-or-second' }], ...values },
    };
    const secret = (key) => /KEY|TOKEN|HOOK|PASSWORD|CREDENTIALS/.test(key);
    state.document = () => {
        const now = state.values;
        const rows = now.OUROBOROS_CONNECTIONS;
        const connections = Object.keys(LABELS).flatMap((kind) => {
            const off = rows.some((row) => row.kind === kind && row.name === 'default' && row.enabled === false);
            const configured = Boolean(now[{ openrouter: 'OPENROUTER_API_KEY', openai: 'OPENAI_API_KEY', anthropic: 'ANTHROPIC_API_KEY' }[kind]]);
            return [{ kind, name: '', builtin: true, enabled: !off, configured, source: configured ? 'settings' : 'none', id: kind, display: LABELS[kind] },
                ...rows.filter((row) => row.kind === kind && row.name !== 'default').map((row) => ({ kind, name: row.name, label: row.label || '', builtin: false,
                    enabled: row.enabled !== false, configured: Boolean(row.api_key), source: 'settings', id: `${kind}@${row.name}`,
                    display: `${LABELS[kind]} · ${row.label || row.name}` }))];
        });
        return { ...Object.fromEntries(Object.entries(now).map(([key, value]) => [key, secret(key) && value ? MASK : structuredClone(value)])),
            OUROBOROS_CONNECTIONS: rows.map((row) => ({ ...row, ...(row.api_key ? { api_key: MASK } : {}) })),
            _meta: { connections, effects: { GITHUB_REPO: 'next_task', GITHUB_TOKEN: 'next_task', TEAM_HOOK: 'next_task', OUROBOROS_CONNECTIONS: 'next_call' },
                tokens: { OUROBOROS_CONNECTIONS: `list-${state.revision}`, GITHUB_REPO: 'repo-1' },
                custom_secret_keys: Object.keys(now).filter((key) => /^(TEAM|SLACK|MY)_/.test(key) && now[key]),
                setup_contract: { modelSlots: [{ slot: 'main', settingKey: 'OUROBOROS_MODEL', label: 'Main Model' }] } } };
    };
    globalThis.fetch = async (url, init = {}) => {
        const path = String(url).split('?')[0];
        const method = init.method || 'GET';
        const body = init.body ? JSON.parse(init.body) : null;
        state.calls.push({ method, path, url: String(url), body });
        if (path === '/api/settings' && method === 'GET') return response(state.document());
        if (path === '/api/settings') {
            if (state.refuse) { const refusal = state.refuse; state.refuse = null; return response(...refusal); }
            const { if_unchanged: wanted, ...changed } = body;
            if (wanted?.OUROBOROS_CONNECTIONS && wanted.OUROBOROS_CONNECTIONS !== `list-${state.revision}`) {
                return response({ saved: false, code: 'settings_changed_elsewhere', error: 'Changed on another device. Reload the current value before saving.' }, 409);
            }
            const changes = {};
            for (const [key, value] of Object.entries(changed)) {
                if (key === 'OUROBOROS_CONNECTIONS') {
                    // A mask a row sends back keeps the stored secret of the row with that identity.
                    state.values[key] = value.map((row) => ({ ...row, ...(row.api_key === MASK
                        ? { api_key: state.values[key].find((old) => old.kind === row.kind && old.name === row.name)?.api_key || '' } : {}) }));
                    state.revision += 1;
                    changes[key] = { withheld: 'structure', token: `list-${state.revision}` };
                } else {
                    changes[key] = secret(key) ? { withheld: 'secret' } : { old: state.values[key], new: value, token: 'repo-2' };
                    state.values[key] = value;
                }
            }
            // The rows that still name a key which cannot send ride on the answer, as the gateway sends them.
            return response({ status: 'saved', saved: true, changes, ...(state.references ? { connection_references: state.references } : {}) });
        }
        if (path === '/api/extensions') return response({ skills, live: { settings_sections: [] } });
        if (path === '/api/claudexor/status') return response(state.status);
        if (path === '/api/claudexor/wake') { state.wakes += 1; return response(statusAnswer()); }
        if (path === '/api/claudexor/maintenance/harnesses') return response({ harnesses: [] });
        if (path === '/api/providers/test') { state.askedTests.push(body); return response({ ok: false, error: 'No credits' }); }
        if (path === '/api/settings/secret') return response({ value: state.values[body.key] });
        throw new Error(`Unexpected request ${method} ${url}`);
    };
    env.document.createElement = (tag) => new Stub(tag);
    const container = new Stub('div');
    env.document.body.appendChild(container);
    const asked = [];
    const opened = [];
    const ctx = { readSettings, openSettings: (path) => { opened.push(path); return true; },
        ask: async (question) => { asked.push(question); return state.answer ?? true; } };
    const show = (path) => globalThis.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path } });
    const mount = async (path) => {
        const handle = await settingsPageFor(path).mount(container, { path, row: '', ctx });
        show(path);
        await settle();
        return handle;
    };
    const posts = () => state.calls.filter((call) => call.method === 'POST' && call.path === '/api/settings').map((call) => call.body);
    return { ...state, state, container, ctx, asked, opened, mount, show, posts,
        find: (selector) => container.querySelector(selector), all: (selector) => container.querySelectorAll(selector) };
}
const starts = (world) => world.state.calls.filter((call) => STARTS.includes(call.path)).map((call) => `${call.method} ${call.path}`);

test('the family is registered, and every path of it names its page', () => {
    assert.equal(settingsPageFor('accounts')?.prefix, 'accounts');
    const title = settingsPageFor('accounts').title;
    assert.deepEqual(['accounts', 'accounts/codex', 'accounts/key/openai', 'accounts/key/new', 'accounts/key/custom'].map(title),
        ['Accounts and keys', 'Codex', 'OpenAI · Default', 'New key', 'New key']);
    assert.deepEqual(parseKeyPath('accounts/key/openrouter/research-credits'), { kind: 'openrouter', name: 'research-credits', fresh: false });
    assert.deepEqual(parseKeyPath('accounts/key/new'), { kind: '', name: '', fresh: true });
    assert.equal(parseKeyPath('accounts/key/nobody'), null, 'a word that names no provider is no key');
    assert.equal(parseKeyPath('accounts/codex'), null, 'a provider\'s page is not a key\'s');
    assert.equal(keyPath('openai-compatible', 'my server'), 'accounts/key/openai-compatible/my%20server');
    assert.equal(keyTitle(keyPath('openrouter', 'second')), 'OpenRouter · second', 'an added key is named by its stored name until a read names its label');
});

test('the rows of the page are data with one control per key, and search finds a key by its provider', () => {
    assert.deepEqual([rowById('GITHUB_TOKEN').control.kind, rowById('GITHUB_REPO').control.kind], ['secret', 'text']);
    assert.deepEqual([rowById('GITHUB_TOKEN').page, rowById('GITHUB_REPO').page, rowById('OPENAI_BASE_URL').page, rowById('OPENAI_BASE_URL').fold],
        ['accounts', 'accounts', 'accounts/key/openai', true]);
    // Every field of every provider card has exactly one home: its key's page (or, for the legacy address, the fold).
    const keys = [...ROWS, ...HAND_WRITTEN].flatMap((row) => row.keys);
    for (const card of PROVIDER_CARDS) {
        for (const field of card.fields) assert.equal(keys.filter((key) => key === field.settingKey).length, 1, field.settingKey);
        assert.ok(HAND_WRITTEN.some((row) => row.page === keyPath(card.kind) && settingsPageFor(row.page)), card.kind);
    }
    assert.equal(HAND_WRITTEN.find((row) => row.keys.includes('OUROBOROS_CONNECTIONS')).page, 'accounts');
});

test('a provider\'s head says how many take turns and who needs the owner, from what was read; nothing read is "Not checked"', () => {
    const payload = statusAnswer();
    const [codex, claude, cursor] = accountGroups(payload, { accountsRead: 'ok' });
    assert.deepEqual(providerRowFacts(codex), { meta: '16 of 18 in rotation', chip: { text: '1 needs sign-in', tone: 'warn' } });
    assert.deepEqual(providerRowFacts(claude), { meta: '2 of 2 in rotation', chip: null });
    assert.deepEqual(providerRowFacts(cursor), { meta: 'No account connected', chip: null }, 'a provider with no account says so once it was read');
    // A store nobody read claims nothing, and a remembered row is dated and raises no alarm.
    assert.deepEqual(providerRowFacts(codex, { asked: false }), { meta: 'Not checked', chip: null });
    assert.deepEqual(providerRowFacts(codex, { accountsRead: 'failed' }), { meta: '17 of 18 in rotation · last known', chip: null });
    // The program's one fact rides only when this session already read it, and only when it asks for the owner.
    assert.equal(providerRowFacts(claude, { program: { updateAvailable: 'yes', newest: { status: 'available', version: '2.5.0' } } }).meta,
        '2 of 2 in rotation · program update: 2.5.0 available');
    assert.equal(providerRowFacts(claude, { program: { updateAvailable: 'no' } }).meta, '2 of 2 in rotation');
});

test('the glance is one bar per account from windows that were read; the accounts that need sign-in stand first on the page', () => {
    const payload = statusAnswer();
    const rows = accountGroups(payload, { accountsRead: 'ok' })[0].rows;
    assert.equal(quotaLeft(payload, rows[0]), 1);
    assert.equal(quotaLeft(payload, rows[10]), 0.5);
    assert.equal(quotaLeft({ quota: [] }, rows[0]), null, 'no reading is no number');
    const glance = quotaBars(rows, payload);
    assert.equal(glance.bars.length, 18);
    assert.equal(glance.words, '16 in rotation · 1 spent · 1 needs sign-in · 1 out of rotation');
    assert.deepEqual(glance.bars.map((bar) => bar.state).slice(-3), ['spent', 'need', 'off']);
    const lefts = glance.bars.filter((bar) => bar.state === 'ok').map((bar) => bar.left);
    assert.deepEqual(lefts, [...lefts].sort((a, b) => b - a), 'the ones that take turns, by what they have left');
    assert.ok(quotaBars(rows, payload, { quotaRead: 'failed' }).bars.every((bar) => bar.left === null), 'a window that was not read draws no height');

    const html = harnessFamilyMarkup(accountGroups(payload, { accountsRead: 'ok' })[0], payload, { accountsRead: 'ok', quotaRead: 'ok' });
    const order = [...html.matchAll(/data-profile="([^"]+)"/g)].map((hit) => hit[1]);
    assert.equal(order[0], 'studio-03', 'the account whose login failed leads');
    assert.deepEqual(order.slice(1, 4), ['studio-01', 'studio-02', 'studio-04'], 'the rest keep the engine\'s order');
    // Kept unknown, the same rows raise no alarm and keep their order.
    const stale = harnessFamilyMarkup(accountGroups(payload, { accountsRead: 'failed' })[0], payload, { accountsRead: 'failed', quotaRead: 'failed' });
    assert.equal([...stale.matchAll(/data-profile="([^"]+)"/g)][0][1], 'studio-01');
    // A row's place in rotation is a switch; what is rare waits behind More; the program's row stands above.
    assert.equal((html.match(/role="switch" data-harness-toggle/g) || []).length, 18);
    assert.match(html, /data-harness-login>Sign in<\/button>\s*<label class="ui-switch">/, 'Sign in stands in the row of the account that needs it');
    assert.equal((html.match(/data-harness-login data-rare>Sign in again</g) || []).length, 17);
    assert.equal((html.match(/data-harness-remove data-rare>Remove</g) || []).length, 18);
    assert.doesNotMatch(html, /title="/, 'no fact lives in a hover title');
});

test('showing the list reads no status, starts no daemon and calls no provider; showing a provider\'s page reads once', async (t) => {
    const world = server(t, { status: statusAnswer({ daemon: { state: 'stale', runtime: { state: 'ready' } }, reads: 'not_read' }) });
    t.after(() => { destroyHarnessAccounts(); });
    const before = { subscribers: claudexorStatus.subscriberCount, polling: claudexorStatus.polling };
    const list = await world.mount('accounts');
    world.show('accounts');
    world.show('');
    await settle();
    assert.deepEqual(starts(world), [], 'the list is drawn from what the shared store already holds');
    assert.equal(claudexorStatus.polling, before.polling, 'and its subscriber arms no poll');
    // One head per provider, never a row per account; an unread store says so in words and offers one Check.
    assert.deepEqual(world.all('[data-provider-rows]')[0].children.map((head) => head.dataset.entryId), ['codex', 'claude', 'cursor']);
    assert.match(world.find('[data-provider-rows]').textContent, /Not checked/);
    assert.equal(world.find('[data-accounts-check]').hidden, false);
    assert.deepEqual([...new Set(world.state.calls.map((call) => `${call.method} ${call.path}`))].sort(), ['GET /api/extensions', 'GET /api/settings'],
        'only the settings document and the skills are read');
    list.unmount();

    // The provider's page: its showing is the owner's act. One status read, one program read, and the idle daemon it
    // found provisioned is started once.
    world.state.calls.length = 0;
    const page = await world.mount('accounts/codex');
    assert.deepEqual(starts(world).filter((line) => line.includes('status')), ['GET /api/claudexor/status']);
    assert.deepEqual(starts(world).filter((line) => line.includes('wake')), ['POST /api/claudexor/wake']);
    const programs = world.state.calls.filter((call) => call.path === '/api/claudexor/maintenance/harnesses' && /checkLatest=true/.test(call.url));
    assert.equal(programs.length, 1, 'the program is read once, with its latest version');
    assert.equal(world.all('.harness-account-row').length, 18, 'this provider\'s accounts, and no other\'s');
    assert.ok(world.all('.harness-account-row').every((row) => row.dataset.harness === 'codex'));
    // The frame: Refresh, then Restart Claudexor (hidden until an owned engine was ever started), in words.
    assert.equal(world.find('#btn-harness-refresh').textContent, 'Refresh');
    assert.equal(world.find('#btn-claudexor-restart').textContent, 'Restart Claudexor');
    assert.ok(world.find('[data-family-maintenance]'), 'the program\'s row has its host above the accounts');
    // Another page of Settings shown while this one stands reads nothing more.
    world.state.calls.length = 0;
    world.show('accounts');
    world.show('');
    await settle();
    assert.deepEqual(starts(world), []);
    page.unmount();
    assert.equal(claudexorStatus.subscriberCount, before.subscribers, 'both pages let the store go');
});

test('the list draws a head per key that exists and a row per provider that has none; a head\'s switch writes one entry at once', async (t) => {
    const world = server(t);
    const list = await world.mount('accounts');
    const heads = () => world.find('[data-key-rows]').children;
    assert.deepEqual(heads().map((head) => head.dataset.entryId), ['openrouter', 'openrouter@research-credits']);
    assert.match(heads()[0].textContent, /OpenRouter · Default\s*Main/);
    assert.match(heads()[1].textContent, /OpenRouter · Research credits\s*No role uses it/);
    assert.equal(world.find('[data-more-rows]').children.length, 8, 'the providers that have no key yet wait under More providers');
    assert.equal(world.find('[data-settings-fold]').hidden, false);
    assert.doesNotMatch(world.container.textContent, /sk-or-real|sk-or-second|ghp_real/, 'no key is ever drawn');

    // Off for an added key: its one entry changes, with the token of the fresh read.
    const second = heads()[1].querySelector('[data-entry-switch]');
    second.checked = false;
    await world.container.fire('change', { target: second });
    await settle();
    assert.equal(world.posts().length, 1);
    assert.deepEqual(world.posts()[0], { OUROBOROS_CONNECTIONS: [{ kind: 'openrouter', name: 'research-credits', label: 'Research credits', api_key: MASK, enabled: false }],
        if_unchanged: { OUROBOROS_CONNECTIONS: 'list-1' } });
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].api_key, 'sk-or-second', 'the kept mask kept the stored secret');
    assert.equal(heads()[1].querySelector('[data-entry-switch]').hasAttribute('checked'), false);
    // Off for a built-in key: the secret-free row that names it; on again takes that row away.
    assert.equal((await setKeyEnabled({ kind: 'openrouter' }, false)).state, 'saved');
    assert.deepEqual(world.posts().at(-1).OUROBOROS_CONNECTIONS.at(-1), { kind: 'openrouter', name: 'default', enabled: false });
    assert.equal((await setKeyEnabled({ kind: 'openrouter' }, true)).state, 'saved');
    assert.ok(!world.posts().at(-1).OUROBOROS_CONNECTIONS.some((row) => row.name === 'default'));

    // A list that changed between the read and the write is refused and never retried: one request, and the head
    // shows what is stored.
    const before = world.posts().length;
    const realFetch = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        const answer = await realFetch(url, init);
        if ((init.method || 'GET') === 'GET' && String(url) === '/api/settings') world.state.revision += 1;  // another device wrote meanwhile
        return answer;
    };
    const outcome = await setKeyEnabled({ kind: 'openrouter', name: 'research-credits' }, true);
    globalThis.fetch = realFetch;
    assert.equal(outcome.state, 'changed_elsewhere');
    assert.equal(world.posts().length, before + 1, 'refused once, not retried');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].enabled, false, 'nothing was written');
    assert.equal(editorLine(outcome).text, 'Changed on another device. This is the current value.');
    // A key that is gone is no entry to switch: nothing is sent.
    assert.equal((await setKeyEnabled({ kind: 'openrouter', name: 'nobody' }, false)).state, 'changed_elsewhere');
    assert.equal(world.posts().length, before + 1);
    list.unmount();
});

test('a key\'s page: Save sends only what changed, Test says what the provider said, Remove asks first', async (t) => {
    const world = server(t);
    // The built-in key: its field shows the mask; an untouched Save sends nothing, a replaced key travels alone.
    let page = await world.mount('accounts/key/openrouter');
    const field = () => world.find('[data-connection-field="api_key"]');
    assert.equal(field().value, MASK);
    assert.equal(world.find('.settings-page-lead').textContent, 'Main.', 'the lead says who names the key');
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.equal(world.posts().length, 0);
    assert.equal(world.find('[data-editor-result]').textContent, 'Nothing changed.');
    field().value = 'sk-or-new';
    await world.container.fire('input', { target: field() });
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.deepEqual(world.posts(), [{ OPENROUTER_API_KEY: 'sk-or-new' }]);
    assert.equal(world.find('[data-editor-result]').textContent, 'Saved · applies to new tasks', 'a secret has no "was" and no Undo');
    assert.equal(world.find('[data-setting-undo]'), null);
    assert.equal(field().value, MASK, 'and is shown back only as its mask');
    // Test: the provider's own word, on the page and then in the list's chip. Nothing is said about credit before it.
    assert.equal(keyHeads(world.state.document()).keys[0].chip, null);
    await world.container.fire('click', { target: world.find('[data-connection-test]') });
    await settle();
    assert.deepEqual(world.state.askedTests, [{ provider_id: 'openrouter', overrides: {} }]);
    assert.equal(world.find('[data-editor-result]').textContent, 'Not ready — No credits');
    assert.deepEqual(keyHeads(world.state.document()).keys[0].chip, { text: 'Not ready — No credits', tone: 'warn' });
    // Remove asks first; declined, nothing is sent. Confirmed, the built-in key's credential is cleared.
    world.state.answer = false;
    await world.container.fire('click', { target: world.find('[data-editor-remove]') });
    await settle();
    assert.equal(world.asked.length, 1);
    assert.match(world.asked[0].title, /^Remove OpenRouter · Default\?$/);
    assert.equal(world.posts().length, 1);
    world.state.answer = true;
    await world.container.fire('click', { target: world.find('[data-editor-remove]') });
    await settle();
    assert.deepEqual(world.posts().at(-1), { OPENROUTER_API_KEY: '' });
    assert.deepEqual(world.opened, ['accounts'], 'and the page returns to the list');
    assert.equal(keyHeads(world.state.document()).keys.some((head) => head.id === 'openrouter'), false, 'a removed key has no head; its provider waits under More providers');
    page.unmount();

    // A key added beside it: its row is written from a fresh read, and a row that changed since the form read it is
    // shown, never written over.
    page = await world.mount('accounts/key/openrouter/research-credits');
    assert.equal(world.find('[data-connection-label]').value, 'Research credits');
    world.find('[data-connection-label]').value = 'Research';
    world.state.values.OUROBOROS_CONNECTIONS[0].label = 'Renamed elsewhere';
    const sent = world.posts().length;
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.equal(world.posts().length, sent, 'a form read from an older row writes nothing');
    assert.equal(world.find('[data-editor-result]').textContent, 'Changed on another device. This is the current value.');
    assert.equal(world.find('[data-connection-label]').value, 'Renamed elsewhere');
    world.find('[data-connection-label]').value = 'Research';
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.deepEqual(world.posts().at(-1), { OUROBOROS_CONNECTIONS: [{ kind: 'openrouter', name: 'research-credits', label: 'Research', api_key: MASK }],
        if_unchanged: { OUROBOROS_CONNECTIONS: `list-${world.state.revision - 1}` } });
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].api_key, 'sk-or-second');
    // The server's refusal is said on the page, and the typed text stays.
    world.state.refuse = [{ saved: false, error: 'OUROBOROS_CONNECTIONS: a key needs its API key.' }, 400];
    world.find('[data-connection-label]').value = 'Typed';
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.match(world.find('[data-editor-result]').textContent, /^Not saved\. OUROBOROS_CONNECTIONS: a key needs its API key\./);
    assert.equal(world.find('[data-connection-label]').value, 'Typed');
    page.unmount();
});

const TIMED_OUT = [{ saved: null, error: 'The save did not answer in time.' }, 504];
/** The next settings write lands, then its answer is lost: the server holds the new value and answers 504. */
function landsThenTimesOut() {
    const real = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        if ((init.method || 'GET') !== 'POST' || String(url) !== '/api/settings') return real(url, init);
        globalThis.fetch = real;
        await real(url, init);
        return response(...TIMED_OUT);
    };
}

test('a key\'s write whose outcome is unknown is judged by what a read shows, field by field; the typed draft stays until then', async (t) => {
    const LAB = { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'https://old.example/v1', api_key: 'sk-lab' };
    const world = server(t, { values: { OUROBOROS_CONNECTIONS: [LAB] } });
    const page = await world.mount(keyPath('openai-compatible', 'lab'));
    const field = (name) => world.find(`[data-connection-field="${name}"]`);
    const said = () => world.find('[data-editor-result]').textContent;
    const save = async () => { await world.container.fire('click', { target: world.find('[data-editor-save]') }); await settle(); };
    assert.equal(field('base_url').value, 'https://old.example/v1');
    field('base_url').value = 'https://new.example/v1';
    await world.container.fire('input', { target: field('base_url') });
    // Not landed: the read shows the old entry. The typed address stays and nothing says Saved.
    world.state.refuse = TIMED_OUT;
    await save();
    assert.equal(world.posts().length, 1);
    assert.equal(said(), 'Not confirmed yet');
    assert.equal(field('base_url').value, 'https://new.example/v1', 'the typed address stays');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].base_url, 'https://old.example/v1');
    // Landed: the read shows the new entry, with the kept secret's mask as sent, so it is saved.
    landsThenTimesOut();
    await save();
    assert.equal(world.posts().length, 2);
    assert.equal(said(), 'Saved · applies from the next model call');
    assert.equal(field('base_url').value, 'https://new.example/v1');
    // A new secret in the entry: the read can only show its mask, so a lost answer stays unknown and the draft stays.
    field('api_key').value = 'sk-lab-2';
    await world.container.fire('input', { target: field('api_key') });
    landsThenTimesOut();
    await save();
    assert.equal(said(), 'Not confirmed yet');
    assert.equal(field('api_key').value, 'sk-lab-2');
    page.unmount();
});

test('a key\'s switch whose outcome is unknown is not called saved until a read shows it', async (t) => {
    const world = server(t);
    world.state.refuse = TIMED_OUT;
    const lost = await setKeyEnabled({ kind: 'openrouter', name: 'research-credits' }, false);
    assert.equal(lost.state, 'unknown');
    assert.equal(editorLine(lost).text, 'Not confirmed yet');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].enabled, undefined);
    landsThenTimesOut();
    assert.equal((await setKeyEnabled({ kind: 'openrouter', name: 'research-credits' }, false)).state, 'saved');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].enabled, false);
});

test('a new key needs its name; saved, it takes a stored name from it and the list says so', async (t) => {
    const world = server(t);
    const page = await world.mount('accounts/key/new');
    assert.equal(world.find('[data-key-kind]').children.length, PROVIDER_CARDS.length, 'the provider is chosen first');
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.equal(world.posts().length, 0);
    assert.equal(world.find('[data-editor-result]').textContent, 'Not saved. OpenRouter: name the new key.');
    world.find('[data-connection-label]').value = 'Research credits';
    world.find('[data-connection-field="api_key"]').value = 'sk-or-third';
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.equal(world.posts().length, 0, 'a name another key of the provider has is refused here');
    assert.match(world.find('[data-editor-result]').textContent, /two keys are named Research credits/);
    world.find('[data-connection-label]').value = 'Night batch';
    await world.container.fire('click', { target: world.find('[data-editor-save]') });
    await settle();
    assert.deepEqual(world.posts().at(-1).OUROBOROS_CONNECTIONS.at(-1), { kind: 'openrouter', name: 'night-batch', label: 'Night batch', api_key: 'sk-or-third' });
    assert.deepEqual(world.opened, ['accounts']);
    page.unmount();
});

test('a stored key\'s row says only whether a value is saved: no echo, no "was", no Undo; Remove asks first', async (t) => {
    const world = server(t, { skills: [{ name: 'slack', title: 'Slack connector', grants: { requested_keys: ['SLACK_BOT_TOKEN', 'OPENROUTER_API_KEY'] } }] });
    assert.deepEqual(otherKeyRows(world.state.document(), [{ name: 'slack', grants: { requested_keys: ['SLACK_BOT_TOKEN'] } }]).map((row) => [row.id, row.meta]),
        [['GITHUB_TOKEN', rowById('GITHUB_TOKEN').meta], ['GITHUB_REPO', rowById('GITHUB_REPO').meta], ['SLACK_BOT_TOKEN', 'Asked by slack'], ['TEAM_HOOK', 'Your own key']]);
    assert.doesNotMatch(secretRowHtml(storedKeyRow('TEAM_HOOK', 'Your own key')), /value="/, 'the row is drawn with no value');
    const list = await world.mount('accounts');
    const row = (id) => world.all('[data-setting-row]').find((node) => node.dataset.settingRow === id);
    assert.deepEqual(world.find('[data-other-rows]').children.map((node) => node.children[0].dataset.settingRow), ['GITHUB_TOKEN', 'GITHUB_REPO', 'SLACK_BOT_TOKEN', 'TEAM_HOOK']);
    const token = row('GITHUB_TOKEN');
    const input = token.querySelector('[data-setting-control]');
    assert.equal(token.querySelector('[data-secret-state]').textContent, 'Saved');
    assert.equal(token.querySelector('[data-secret-open]').textContent, 'Replace');
    assert.equal(row('SLACK_BOT_TOKEN').querySelector('[data-secret-open]').textContent, 'Add', 'a key nobody stored offers Add');
    assert.equal(row('SLACK_BOT_TOKEN').querySelector('[data-secret-state]').textContent, '');
    assert.equal(input.value ?? '', '', 'the stored value is never put into the field');
    assert.equal(token.querySelector('[data-secret-field]').hidden, true);
    assert.doesNotMatch(world.container.textContent, /ghp_real|https:\/\/hook/);
    // Replace opens the field; a typed value is written when the field is left, like every typed row.
    await token.parentElement.fire('click', { target: token.querySelector('[data-secret-open]') });
    assert.equal(token.querySelector('[data-secret-field]').hidden, false);
    input.value = 'ghp_new';
    await token.parentElement.fire('change', { target: input });
    await settle();
    assert.deepEqual(world.posts(), [{ GITHUB_TOKEN: 'ghp_new' }]);
    const result = row('GITHUB_TOKEN').querySelector('[data-setting-result]');
    assert.equal(result.textContent, 'Saved · applies to new tasks');
    assert.doesNotMatch(result.textContent, /was|ghp_/);
    assert.equal(row('GITHUB_TOKEN').querySelector('[data-setting-undo]'), null);
    assert.equal(row('GITHUB_TOKEN').querySelector('[data-setting-control]').value, '', 'the field is emptied');
    assert.equal(row('GITHUB_TOKEN').querySelector('[data-secret-field]').hidden, true, 'and put away once the write landed');
    // The plain row beside it keeps the row helper's words: what it was, and Undo.
    const repo = row('GITHUB_REPO').querySelector('[data-setting-control]');
    repo.value = 'studio/other';
    await row('GITHUB_REPO').parentElement.fire('change', { target: repo });
    await settle();
    assert.equal(row('GITHUB_REPO').querySelector('[data-setting-result]').textContent, 'Saved · applies to new tasks · was studio/home · Undo');
    // Remove asks first; declined, nothing is sent; confirmed, the key is cleared and the row offers Add.
    const sent = world.posts().length;
    world.state.answer = false;
    await row('TEAM_HOOK').parentElement.fire('click', { target: row('TEAM_HOOK').querySelector('[data-secret-remove]') });
    await settle();
    assert.equal(world.asked.length, 1);
    assert.equal(world.posts().length, sent);
    world.state.answer = true;
    await row('TEAM_HOOK').parentElement.fire('click', { target: row('TEAM_HOOK').querySelector('[data-secret-remove]') });
    await settle();
    assert.deepEqual(world.posts().at(-1), { TEAM_HOOK: '' });
    assert.equal(row('TEAM_HOOK'), undefined, 'a key of the owner\'s own that holds nothing is no row');
    list.unmount();
});

test('a key of the owner\'s own: its name is judged before anything is sent, and a name already stored is refused', async (t) => {
    const world = server(t);
    const page = await world.mount('accounts/key/custom');
    const [name, value, save] = ['[data-custom-secret-key]', '[data-custom-secret-value]', '[data-editor-save]'].map(world.find);
    const press = async () => { await save.fire('click'); await settle(); };
    const said = () => world.find('[data-editor-result]').textContent;
    for (const [typed, secret, refusal] of [['', 'x', /Enter a key name/], ['ab', 'x', /at least three letters/], ['OUROBOROS_MODEL', 'x', /built-in setting/],
        ['TEAM_HOOK', 'x', /Duplicate key TEAM_HOOK/], ['MY_KEY', '', /Enter a value for this key/]]) {
        name.value = typed;
        value.value = secret;
        await press();
        assert.match(said(), refusal, typed);
    }
    assert.equal(world.posts().length, 0, 'none of them reached the server');
    name.value = 'my_key';
    value.value = 'v-1';
    await press();
    assert.deepEqual(world.posts(), [{ MY_KEY: 'v-1' }]);
    assert.deepEqual(world.opened, ['accounts']);
    page.unmount();
});

test('the wizard\'s provider step renders and collects as before', async (t) => {
    const { createConnectionsEditor } = await import('../modules/settings_connections.js');
    const world = server(t);
    const element = globalThis.Element;
    globalThis.Element = NodeStub;  // the editor asks whether a press landed on an element
    t.after(() => { globalThis.Element = element; });
    const root = new Stub('div');
    root.innerHTML = `<div data-settings-panel="providers">${PROVIDER_CARDS.map(providerSettingsCard).join('')}</div>`;
    assert.equal(root.querySelectorAll('[data-connection-row]').length, PROVIDER_CARDS.length, 'one Default row per provider card');
    assert.equal(root.querySelectorAll('[data-connection-add]').length, PROVIDER_CARDS.length);
    let edits = 0;
    const editor = createConnectionsEditor({ root, onChange: () => { edits += 1; } });
    const document = world.state.document();
    editor.load(document, { modelSlots: document._meta.setup_contract.modelSlots });
    assert.deepEqual(editor.collect(), {}, 'an untouched draft collects nothing');
    assert.deepEqual(editor.validate(), []);
    const added = root.querySelectorAll('[data-connection-row]').find((row) => row.dataset.name === 'research-credits');
    assert.ok(added, 'a saved added key is a row of its provider\'s card');
    assert.equal(added.querySelector('[data-connection-status]').textContent, 'Ready');
    // Disable in the wizard stays a draft: it is collected with the wizard's other answers, never written at once.
    await root.querySelector('[data-settings-panel="providers"]').fire('click', { target: added.querySelector('[data-connection-toggle]') });
    assert.equal(edits, 1);
    assert.deepEqual(editor.collect(), { OUROBOROS_CONNECTIONS: [{ kind: 'openrouter', name: 'research-credits', label: 'Research credits', api_key: MASK, enabled: false }] });
    assert.equal(world.posts().length, 0);
    editor.destroy();
});

test('a key saved, turned off or removed names the rows that still point at a key which cannot send; none, nothing is added', () => {
    const references = [{ setting: 'OUROBOROS_MODEL_LIGHT', route: 'openai@work::gpt-light', connection: 'openai@work' },
        { setting: 'OUROBOROS_MODEL_FALLBACKS', entry: '2', route: 'anthropic::claude-a', connection: 'anthropic' }];
    const saved = { state: 'saved', effect: 'next_call', withheld: 'structure', message: '', code: '', references };
    assert.deepEqual(referencesLine(saved), { tone: 'warn',
        text: 'Still naming a key that was removed or turned off, so they cannot send: Light (OpenAI · work), Fallback 2 (Anthropic)' });
    assert.match(editorLine(saved).text, /^Saved.*Still naming a key that was removed or turned off, so they cannot send: Light \(OpenAI · work\)/);
    // The read's keys give a key its own label.
    assert.match(referencesLine(saved, [{ id: 'openai@work', label: 'Work account' }]).text, /Light \(Work account\)/);
    assert.equal(editorLine(saved).tone, 'warn');
    const clean = { ...saved, references: [] };
    assert.deepEqual(referencesLine(clean), {});
    assert.doesNotMatch(editorLine(clean).text, /Still naming/);
});

test('a key turned off, added or removed on its page while a role names a key that cannot send: the line names it by its label, and the list says it after the page', async (t) => {
    const world = server(t, { values: { OUROBOROS_CONNECTIONS: [{ kind: 'openai', name: 'work', label: 'Work account', api_key: 'sk-work' },
        { kind: 'openrouter', name: 'research-credits', label: 'Research credits', api_key: 'sk-or-second' }] } });
    const list = await world.mount('accounts');
    // A key's page opens as its own level above the list, which stays underneath to be told what the page did.
    const onPage = async (path, act) => {
        const box = new Stub('div');
        const page = await settingsPageFor(path).mount(box, { path, row: '', ctx: world.ctx });
        await settle();
        await act(box);
        await settle();
        page.unmount();
    };
    const press = (selector) => async (box) => box.fire('click', { target: box.querySelector(selector) });
    const line = () => world.find('[data-entry-result]');
    const naming = 'Still naming a key that was removed or turned off, so they cannot send: Light (OpenAI · Work account)';
    world.state.references = [{ setting: 'OUROBOROS_MODEL_LIGHT', route: 'openai@work::gpt-light', connection: 'openai@work' }];
    // Turned off on its page: the page's line names the key by the label the owner gave it, not its stored name.
    await onPage('accounts/key/openai/work', async (box) => {
        await press('[data-key-toggle]')(box);
        await settle();
        assert.equal(box.querySelector('[data-editor-result]').textContent, naming);
        assert.equal(box.querySelector('[data-editor-result]').dataset.tone, 'warn');
    });
    // A key added meanwhile: the list says what was added and the roles still unable to send.
    await onPage('accounts/key/new', async (box) => {
        box.querySelector('[data-connection-label]').value = 'Night batch';
        box.querySelector('[data-connection-field="api_key"]').value = 'sk-or-third';
        await press('[data-editor-save]')(box);
    });
    assert.equal(line().textContent, `Added OpenRouter · Night batch. ${naming}`);
    assert.equal(line().dataset.tone, 'warn');
    await onPage('accounts/key/openai/work', press('[data-editor-remove]'));
    assert.equal(line().textContent, `Removed OpenAI · Work account. ${naming}`);
    assert.equal(line().dataset.tone, 'warn');
    // Nothing names the next one: the line says only what was removed, in the plain colour.
    world.state.references = null;
    await onPage('accounts/key/openrouter/research-credits', press('[data-editor-remove]'));
    assert.equal(line().textContent, 'Removed OpenRouter · Research credits.');
    assert.notEqual(line().dataset.tone, 'warn');
    list.unmount();
});

