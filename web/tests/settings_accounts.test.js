// Settings › Accounts and keys (settings_accounts.js composes harness_accounts.js, settings_connections.js and
// settings_secrets.js; docs/DESIGN.md "Settings"): the list, a provider's page, a key's page and the stored keys, each
// mounted the way the shell mounts it, against a server that answers like the settings server with its receipt.
// What must never happen by itself is proved in both directions: showing the list reads no status, starts no daemon
// and calls no provider, while a provider's page observes accounts/quota independently of full discovery and wake.
import assert from 'node:assert/strict';
import test from 'node:test';

import { claudexorStatus } from '../modules/claudexor_status_store.js';
import { accountGroups, destroyHarnessAccounts, harnessFamilyMarkup } from '../modules/harness_accounts.js';
import { otherKeyRows, providerRowFacts, quotaBars, quotaLeft } from '../modules/settings_accounts.js';
import { editorLine, keyHeads, keyPath, keyTitle, parseKeyPath, referencesLine, setKeyEnabled } from '../modules/settings_connections.js';
import { forgetSettingRows, readSettings } from '../modules/settings_controls.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { PROVIDER_CARDS, createFlatKeyChoices, providerSettingsCard } from '../modules/settings_provider_cards.js';
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
                    display: `${LABELS[kind]} · ${row.label || row.name}`,
                    // The server's own reading of a private compatible key: its consent stands for the address it has.
                    ...(state.unidentified ? { unidentified: { ...state.unidentified, admission: row.continue_unconfirmed && row.continue_unconfirmed === row.base_url ? 'consent' : '' } } : {}) }))];
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
        if (path === '/api/claudexor/status') {
            if (new URL(String(url), 'http://fixture').searchParams.get('view') === 'quota') {
                const { profiles, quota, quota_absences, unified_accounts, reads } = state.status;
                return response({ view: 'quota', profiles, quota, quota_absences, unified_accounts,
                    reads: { catalog: 'not_read', accounts: reads.accounts, quota: reads.quota }, read_errors: {}, timings_ms: {} });
            }
            return response(state.status);
        }
        if (path === '/api/claudexor/wake') { state.wakes += 1; return response(statusAnswer()); }
        if (path === '/api/claudexor/maintenance/harnesses') return response({ harnesses: [] });
        if (path === '/api/providers/test') { state.askedTests.push(body); return response(state.testAnswer?.(body) ?? { ok: false, error: 'No credits' }); }
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
    let mountedPath = '';
    const show = (path) => {
        // The shell makes the mounted provider visible; the text-only DOM has no layout engine.
        const groups = container.querySelector('#harness-accounts-groups');
        if (groups) groups.offsetParent = path === mountedPath ? container : null;
        return globalThis.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path } });
    };
    const mount = async (path) => {
        mountedPath = path;
        const handle = await settingsPageFor(path).mount(container, { path, row: '', ctx });
        show(path);
        await settle();
        return handle;
    };
    const posts = () => state.calls.filter((call) => call.method === 'POST' && call.path === '/api/settings').map((call) => call.body);
    return { ...state, state, container, ctx, asked, opened, mount, show, posts,
        find: (selector) => container.querySelector(selector), all: (selector) => container.querySelectorAll(selector) };
}
const starts = (world) => world.state.calls.filter((call) => STARTS.includes(call.path)).map((call) => `${call.method} ${call.url}`);

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
    assert.match(html, /data-harness-login>Sign in<\/button><button[^>]*data-harness-remove data-rare>/, 'Sign in stands on the line under the words of the account that needs it, the switch keeps the row\'s end');
    assert.equal((html.match(/data-harness-login data-rare>Sign in again</g) || []).length, 17);
    const actionsOf = [...html.matchAll(/<div class="harness-account-actions">([\s\S]*?)<\/div>/g)].map((hit) => hit[1]);
    assert.equal(actionsOf.length, 18);
    assert.ok(actionsOf.every((part) => /data-harness-toggle/.test(part) && /data-resources/.test(part) && !/data-harness-login|data-harness-remove/.test(part)),
        'every row keeps only its switch and More at its end, whether or not it needs sign-in');
    assert.equal((html.match(/data-harness-remove data-rare>Remove</g) || []).length, 18);
    assert.doesNotMatch(html, /title="/, 'no fact lives in a hover title');
});

test('showing the list reads no status, starts no daemon and calls no provider; a provider observes independently and wakes once', async (t) => {
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

    // The provider's page starts one full discovery and one independent passive observation.
    // The passive envelope carries no lifecycle: activation joins that full read and wakes the stale home once.
    world.state.calls.length = 0;
    const page = await world.mount('accounts/codex');
    assert.deepEqual(starts(world).filter((line) => line.includes('status')), [
        'GET /api/claudexor/status', 'GET /api/claudexor/status?view=quota']);
    assert.deepEqual(starts(world).filter((line) => line.includes('wake')), ['POST /api/claudexor/wake']);
    assert.deepEqual(starts(world).filter((line) => !/\/claudexor\/(status|wake|maintenance\/harnesses)(\?|$)/.test(line)), [],
        'opening the provider starts no provider refresh, login, model catalog or review-pool read');
    const programs = world.state.calls.filter((call) => call.path === '/api/claudexor/maintenance/harnesses' && /checkLatest=true/.test(call.url));
    assert.equal(programs.length, 1, 'the program is read once, with its latest version');
    assert.equal(world.all('.harness-account-row').length, 18, 'this provider\'s accounts, and no other\'s');
    assert.ok(world.all('.harness-account-row').every((row) => row.dataset.harness === 'codex'));
    // The frame: Refresh, then Restart Claudexor (hidden until an owned engine was ever started), in words.
    assert.equal(world.find('#btn-harness-refresh').textContent, 'Refresh');
    assert.equal(world.find('#btn-claudexor-restart').textContent, 'Restart Claudexor');
    assert.ok(world.find('[data-family-maintenance]'), 'the program\'s row has its host above the accounts');
    const family = world.find('[data-family="codex"]');
    const login = family.querySelector('[data-family-login]');
    const program = family.querySelector('[data-family-maintenance]');
    await claudexorStatus.observe({ afterCurrent: true });
    assert.equal(world.find('[data-family="codex"]'), family);
    assert.equal(world.find('[data-family-login]'), login);
    assert.equal(world.find('[data-family-maintenance]'), program);
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

test('a compatible key\'s consent is its own box: it stores the address shown, a new address clears it, Test checks the draft', async (t) => {
    const LAB = { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', api_key: 'sk-lab' };
    const world = server(t, { values: { OUROBOROS_CONNECTIONS: [LAB] } });
    world.state.unidentified = { checked: { type: 'unknown', observed_at: '2026-11-19T10:00:00Z' } };
    await world.mount(keyPath('openai-compatible', 'lab'));
    const box = () => world.find('[data-connection-consent]');
    const field = (name) => world.find(`[data-connection-field="${name}"]`);
    const press = async (selector) => { await world.container.fire('click', { target: world.find(selector) }); await settle(); };
    const tick = async (on) => { box().checked = on; await world.container.fire('input', { target: box() }); };
    const stored = () => world.state.values.OUROBOROS_CONNECTIONS[0].continue_unconfirmed;
    assert.equal(box().checked, false);
    assert.match(world.container.textContent, /Continue via the compatible API without confirmation that history is preserved/);
    assert.match(world.find('[data-connection-unidentified]').textContent, /^Server type not confirmed at the last check.*A request goes to one only with your consent\.$/);
    // Ticked and not saved: Test checks what a save would allow, and nothing is stored.
    await tick(true);
    await press('[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai-compatible', connection: 'lab', overrides: { continue_unconfirmed: 'http://lab.lan/v1' } });
    assert.equal(stored(), undefined);
    await press('[data-editor-save]');
    assert.equal(stored(), 'http://lab.lan/v1', 'the consent names the address shown');
    assert.equal(box().checked, true, 'read back, it stands');
    assert.match(world.find('[data-connection-unidentified]').textContent, /may shorten a long conversation without saying so\.$/, 'the ticked box says the rest');
    // Another address typed: the consent was not given for it, and saving so withdraws it.
    field('base_url').value = 'http://other.lan/v1';
    await world.container.fire('input', { target: field('base_url') });
    assert.equal(box().checked, false, 'a new address asks again');
    await press('[data-editor-save]');
    assert.deepEqual([world.state.values.OUROBOROS_CONNECTIONS[0].base_url, stored()], ['http://other.lan/v1', ''], 'no consent is left behind');
    await tick(true);
    await press('[data-editor-save]');
    assert.deepEqual([world.state.values.OUROBOROS_CONNECTIONS[0].base_url, stored()], ['http://other.lan/v1', 'http://other.lan/v1']);
    // Revoked by clearing the box.
    await tick(false);
    await press('[data-editor-save]');
    assert.equal(stored(), '');
    assert.equal(box().checked, false);
    // Only the kind whose server can be a private one of unknown program has such a box.
    assert.deepEqual(PROVIDER_CARDS.filter((card) => card.consent).map((card) => card.kind), ['openai-compatible']);
});

test('the consent box is offered only where a consent can matter', async (t) => {
    const LAB = { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', api_key: 'sk-lab' };
    const page = async (unidentified, row = LAB) => {
        const world = server(t, { values: { OUROBOROS_CONNECTIONS: [row] } });
        world.state.unidentified = unidentified;
        await world.mount(keyPath('openai-compatible', 'lab'));
        return world;
    };
    const offered = (world) => !world.find('.accounts-key-consent').hidden;
    const seen = { type: 'ollama', version: '0.40.16', observed_at: '2026-11-19T10:00:00Z' };
    // A private server that named no program, at its last check or before any.
    assert.equal(offered(await page({ checked: { type: 'unknown' } })), true);
    assert.equal(offered(await page({})), true);
    // An identified server and an address where Ollama was seen: no consent changes what they send.
    assert.equal(offered(await page({ checked: { type: 'llama.cpp' } })), false);
    assert.equal(offered(await page({ ollama: seen })), false);
    // A public address keeps no such fact, and typing another offers nothing: the page cannot tell a private server from a
    // public one, so only a Test that was refused for want of the consent (or a consent that stands) opens the box.
    const open = await page(null);
    assert.equal(offered(open), false);
    const address = open.find('[data-connection-field="base_url"]');
    address.value = 'http://127.0.0.1:8000/v1';
    await open.container.fire('input', { target: address });
    assert.equal(offered(open), false);
    // A consent that stands stays shown even where it decides nothing, so that it can be withdrawn.
    const held = await page({ ollama: seen }, { ...LAB, continue_unconfirmed: LAB.base_url });
    assert.deepEqual([offered(held), held.find('[data-connection-consent]').checked], [true, true]);
    // A key with no address yet offers none either: its first Test says whether the server needs one.
    const empty = server(t);
    await empty.mount(keyPath('openai-compatible'));
    assert.equal(offered(empty), false);
});


/** A compatible key's page and what it needs to be driven: the box, the note, the Test and its line. */
async function compatiblePage(t, { unidentified, row = { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', api_key: 'sk-lab' } } = {}) {
    const world = server(t, { values: { OUROBOROS_CONNECTIONS: [row] } });
    world.state.unidentified = unidentified;
    await world.mount(keyPath('openai-compatible', 'lab'));
    const press = async (selector) => { await world.container.fire('click', { target: world.find(selector) }); await settle(); };
    const type = async (name, value) => { const input = world.find(`[data-connection-field="${name}"]`); input.value = value; await world.container.fire('input', { target: input }); };
    const tick = async (on) => { world.find('[data-connection-consent]').checked = on; await world.container.fire('input', { target: world.find('[data-connection-consent]') }); };
    const view = () => ({ offered: !world.find('.accounts-key-consent').hidden, why: world.find('.accounts-key-consent').dataset.consentOffer,
        ticked: world.find('[data-connection-consent]').checked, note: world.find('[data-connection-unidentified]'),
        line: world.find('[data-editor-result]') });
    return { world, press, type, tick, view };
}
const REFUSAL = (code, error) => ({ ok: false, code, error });
const NO_PROGRAM = 'The server at http://lab.lan does not show which program it runs, so Ouroboros cannot confirm that it keeps the whole history; nothing was sent.';

test('a Test that is refused for want of the consent opens its box unticked, even where the saved key knew a program; Test and Save then carry it', async (t) => {
    // The key was identified at its last check; now the server names nothing.
    const page = await compatiblePage(t, { unidentified: { checked: { type: 'llama.cpp', observed_at: '2026-11-18T10:00:00Z' } } });
    const { world, press, tick, view } = page;
    assert.deepEqual([view().offered, view().note.dataset.serverState], [false, 'identified'], 'a named program needs no consent, so no box yet');
    world.state.testAnswer = () => REFUSAL('ollama_guard_consent', NO_PROGRAM);
    await press('[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai-compatible', connection: 'lab', overrides: {} }, 'a Test of an untouched form sends nothing of the consent');
    assert.deepEqual([view().offered, view().why, view().ticked], [true, 'test', false], 'actionable and unticked');
    assert.equal(view().line.textContent, `Not ready — ${NO_PROGRAM}`);
    assert.deepEqual([view().line.dataset.tone, view().line.dataset.testCode], ['warn', 'ollama_guard_consent']);
    // Ticking is an edit of the form: the answer to the old request goes, the box stays where the owner put the cursor.
    await tick(true);
    assert.deepEqual([view().offered, view().ticked, view().line.textContent, view().line.dataset.testCode], [true, true, '', undefined]);
    world.state.testAnswer = () => ({ ok: true, caution: 'Server type not confirmed: Ouroboros cannot tell whether it keeps the whole history.' });
    await press('[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai-compatible', connection: 'lab', overrides: { continue_unconfirmed: 'http://lab.lan/v1' } },
        'the Test checks what a save would allow');
    assert.equal(view().line.dataset.tone, 'warn', 'a pass on a server that named no program is a warning');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].continue_unconfirmed, undefined, 'a Test saves no consent');
    assert.equal(world.posts().length, 0, 'nor anything else');
    await tick(false);
    assert.equal(view().offered, true, 'unticked again it is still there: the control that was asked for does not leave');
    await tick(true);
    await press('[data-editor-save]');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].continue_unconfirmed, 'http://lab.lan/v1');
});

test('a hold or an old Ollama found by a Test is told in the server\'s words and offers no consent; the head says it short', async (t) => {
    for (const [code, error, chip] of [['ollama_guard_held', 'Ollama was seen at http://lab.lan before, but this connection\'s program was not identified for this request, so nothing was sent.', 'Ollama was seen at this address'],
        ['ollama_guard_unavailable', 'Ollama 0.12.5 can silently truncate history. Upgrade to 0.12.6 or later.', 'Ollama needs an update']]) {
        const { world, press, view } = await compatiblePage(t, { unidentified: { admission: '', checked: { type: 'unknown', observed_at: '2026-11-18T10:00:00Z' } } });
        assert.deepEqual([view().offered, view().why], [true, 'facts'], 'before the Test, the saved facts offer it');
        world.state.testAnswer = () => REFUSAL(code, error);
        await press('[data-connection-test]');
        assert.deepEqual([view().offered, view().line.textContent, view().line.dataset.tone, view().line.dataset.testCode], [false, `Not ready — ${error}`, 'warn', code], code);
        const head = keyHeads(await readSettings()).keys.find((key) => key.id === 'openai-compatible@lab');
        assert.deepEqual(head.chip, { text: chip, tone: 'warn' }, 'a long sentence is no chip');
    }
});

test('a held address shows no box to open it, and a consent that stands is shown only to be withdrawn', async (t) => {
    const seen = { type: 'ollama', version: '0.40.16', observed_at: '2026-11-19T10:00:00Z' };
    const { world, press, tick, view } = await compatiblePage(t, { unidentified: { ollama: seen, checked: { type: 'unknown', observed_at: '2026-11-19T10:00:00Z' } },
        row: { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', api_key: 'sk-lab', continue_unconfirmed: 'http://lab.lan/v1' } });
    assert.deepEqual([view().offered, view().why, view().ticked, view().note.dataset.serverState, view().note.dataset.admission], [true, 'ticked', true, 'held', 'consent']);
    assert.doesNotMatch(view().note.textContent, /consent/, 'the hold says nothing a consent could change');
    await tick(false);
    await press('[data-editor-save]');
    assert.equal(world.state.values.OUROBOROS_CONNECTIONS[0].continue_unconfirmed, '', 'withdrawn');
    assert.deepEqual([view().offered, view().ticked], [false, false], 'withdrawn and held: nothing to offer');
});

test('a draft whose address is not the saved one leaves the saved facts out, starts unticked and carries no old answer', async (t) => {
    const LAB = { kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', api_key: 'sk-lab', continue_unconfirmed: 'http://lab.lan/v1' };
    const { world, press, type, view } = await compatiblePage(t, { row: LAB, unidentified: { checked: { type: 'unknown', observed_at: '2026-11-19T10:00:00Z' } } });
    assert.deepEqual([view().note.dataset.serverState, view().note.dataset.admission, view().ticked], ['unconfirmed', 'consent', true]);
    world.state.testAnswer = () => REFUSAL('ollama_guard_consent', NO_PROGRAM);
    await type('base_url', 'http://other.lan/v1');
    assert.deepEqual([view().note.dataset.serverState, view().note.dataset.admission, view().ticked, view().offered], ['draft', undefined, false, false],
        'the earlier note is labelled as of the saved address; the consent was not given for this one');
    assert.match(view().note.textContent, /^This address has not been checked/);
    await press('[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1).overrides, { base_url: 'http://other.lan/v1', continue_unconfirmed: '' }, 'the draft is tested without the consent the saved address had');
    assert.deepEqual([view().offered, view().why, view().ticked], [true, 'test', false]);
    assert.equal(keyHeads(await readSettings()).keys.find((key) => key.id === 'openai-compatible@lab').chip, null, 'a Test of another address is not the saved key\'s word');
    // Another address typed: what the Test found at the last is gone, and so is the offer.
    await type('base_url', 'https://api.example.com/v1');
    assert.deepEqual([view().offered, view().line.dataset.testCode], [false, undefined]);
    // The saved address typed back: its facts and its consent stand again.
    await type('base_url', 'http://lab.lan/v1');
    assert.deepEqual([view().note.dataset.serverState, view().ticked, view().offered], ['unconfirmed', true, true]);
    assert.equal(world.posts().length, 0, 'none of it was saved');
});

test('Default and named keys, fresh or saved, start without a consent and say nothing until something asks', async (t) => {
    const world = server(t, { values: { OPENAI_COMPATIBLE_BASE_URL: 'http://lab.lan/v1', OPENAI_COMPATIBLE_API_KEY: 'sk-lab' } });
    for (const path of [keyPath('openai-compatible'), 'accounts/key/new']) {
        await world.mount(path);
        if (path.endsWith('/new')) {  // the provider is chosen first
            Object.assign(world.find('[data-key-kind]'), { value: 'openai-compatible' });
            await world.container.fire('change', { target: world.find('[data-key-kind]') });
        }
        assert.equal(world.find('[data-connection-consent]').checked, false, path);
        assert.equal(world.find('.accounts-key-consent').hidden, true, path);
        assert.equal(world.find('[data-connection-unidentified]').hidden, true, path);
    }
});

test('Remove on the built-in compatible key clears its address, key and consent; a consent the organization\'s checker reads stays', async (t) => {
    const saved = { OPENAI_COMPATIBLE_BASE_URL: 'http://lab.lan/v1', OPENAI_COMPATIBLE_API_KEY: 'sk-lab', OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED: 'http://lab.lan/v1' };
    for (const [checker, cleared] of [[null, Object.keys(saved)], [{ fields: ['OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED'] }, Object.keys(saved).slice(0, 2)]]) {
        const world = server(t, { values: { ...saved } });
        const read = world.state.document;
        world.state.document = () => {  // the built-in key is saved; a managed deployment names what its checker reads
            const document = read();
            Object.assign(document._meta.connections.find((row) => row.id === 'openai-compatible'), { configured: true, source: 'settings' });
            if (checker) document._meta.policy_state = { deployment: { checker } };
            return document;
        };
        const page = await world.mount(keyPath('openai-compatible'));
        await world.container.fire('click', { target: world.find('[data-editor-remove]') });
        await settle();
        assert.deepEqual(world.posts().at(-1), Object.fromEntries(cleared.map((key) => [key, ''])), JSON.stringify(checker));
        assert.deepEqual(world.opened, ['accounts'], 'and the page returns to the list');
        page.unmount();
    }
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

test('the wizard\'s compatible rows carry the same consent box: it opens by a Test, rides the Test and the draft, and a new address starts it false', async (t) => {
    const { createConnectionsEditor } = await import('../modules/settings_connections.js');
    server(t);
    const element = globalThis.Element;
    globalThis.Element = NodeStub;
    t.after(() => { globalThis.Element = element; });
    const mount = (settings, answer = () => ({ ok: true })) => {
        const root = new Stub('div');
        root.innerHTML = `<div data-settings-panel="providers">${PROVIDER_CARDS.map(providerSettingsCard).join('')}</div>`;
        const sent = [];
        const editor = createConnectionsEditor({ root, test: async (body) => { sent.push(body); return answer(body); } });
        const draft = { OPENAI_COMPATIBLE_BASE_URL: '', OPENAI_COMPATIBLE_API_KEY: '', OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED: '', OUROBOROS_CONNECTIONS: [], ...settings };
        for (const field of PROVIDER_CARDS.find((card) => card.kind === 'openai-compatible').fields) {  // the wizard puts the draft in the built-in fields first
            root.querySelector(`#${field.id}`).value = draft[field.settingKey];
        }
        editor.load(draft, { modelSlots: [] });
        const panel = root.querySelector('[data-settings-panel="providers"]');
        const row = (named) => root.querySelectorAll('[data-connection-row]').find((each) => each.dataset.kind === 'openai-compatible'
            && (named ? each.dataset.name === named || each.dataset.draft === '1' : each.dataset.builtin === '1'));
        const view = (each) => ({ offered: !each.querySelector('.accounts-key-consent').hidden, why: each.querySelector('.accounts-key-consent').dataset.consentOffer,
            ticked: each.querySelector('[data-connection-consent]').checked, line: each.querySelector('[data-connection-test-status]') });
        const type = async (each, field, value) => { const input = each.querySelector(`[data-connection-field="${field}"]`); input.value = value; await panel.fire('input', { target: input }); };
        const tick = async (each, on) => { each.querySelector('[data-connection-consent]').checked = on; await panel.fire('input', { target: each.querySelector('[data-connection-consent]') }); };
        const test = async (each) => { await panel.fire('click', { target: each.querySelector('[data-connection-test]') }); await settle(); };
        return { root, editor, sent, panel, row, view, type, tick, test };
    };
    const NONAME = REFUSAL('ollama_guard_consent', NO_PROGRAM);

    // The built-in key: nothing is offered, ticked or sent before something asks.
    const wizard = mount({}, () => NONAME);
    const key = wizard.row();
    assert.deepEqual([wizard.view(key).offered, wizard.view(key).ticked, wizard.editor.consent('openai-compatible')], [false, false, null]);
    await wizard.type(key, 'base_url', 'http://lab.lan/v1');
    assert.equal(wizard.view(key).offered, false, 'typing an address offers nothing');
    await wizard.test(key);
    assert.equal(wizard.sent.at(-1).overrides.OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED, undefined, 'an unticked draft sends no consent');
    assert.deepEqual([wizard.view(key).offered, wizard.view(key).why, wizard.view(key).ticked], [true, 'test', false], 'the refusal opens the box, unticked');
    assert.deepEqual([wizard.view(key).line.textContent, wizard.view(key).line.dataset.tone, wizard.view(key).line.dataset.testCode], [`Not ready — ${NO_PROGRAM}`, 'warn', 'ollama_guard_consent']);
    await wizard.tick(key, true);
    assert.equal(wizard.view(key).line.textContent, '', 'ticking is an edit: the answer to the old request goes');
    assert.equal(wizard.editor.consent('openai-compatible'), 'http://lab.lan/v1', 'the draft names the address shown');
    await wizard.test(key);
    assert.equal(wizard.sent.at(-1).overrides.OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED, 'http://lab.lan/v1', 'a Test checks what finishing would allow');
    assert.equal(wizard.sent.at(-1).api_key_setting, 'OPENAI_COMPATIBLE_API_KEY');
    await wizard.type(key, 'base_url', 'http://other.lan/v1');
    assert.deepEqual([wizard.view(key).ticked, wizard.view(key).offered, wizard.view(key).line.dataset.testCode, wizard.editor.consent('openai-compatible')], [false, false, undefined, null],
        'a consent given for one address is not given for another, and nothing is left behind');

    // A named key: the same box, its consent one field of the row the draft collects.
    const named = mount({}, () => NONAME);
    await named.panel.fire('click', { target: named.root.querySelector('[data-connection-add="openai-compatible"]') });
    const fresh = named.row('new');
    fresh.querySelector('[data-connection-label]').value = 'Lab';
    await named.type(fresh, 'base_url', 'http://lab.lan/v1');
    await named.test(fresh);
    assert.deepEqual([named.view(fresh).offered, named.view(fresh).why], [true, 'test']);
    await named.tick(fresh, true);
    assert.deepEqual(named.editor.collect().OUROBOROS_CONNECTIONS, [{ kind: 'openai-compatible', name: 'lab', label: 'Lab', base_url: 'http://lab.lan/v1', continue_unconfirmed: 'http://lab.lan/v1' }]);
    await named.test(fresh);
    assert.deepEqual(named.sent.at(-1), { provider_id: 'openai-compatible', connection: 'lab', overrides: { base_url: 'http://lab.lan/v1', continue_unconfirmed: 'http://lab.lan/v1' } });
    assert.equal(named.editor.consent('openai-compatible'), null, 'the built-in key is another row');

    // A hold or an old Ollama is found by the Test: no box, the server's words, whatever is ticked before.
    for (const code of ['ollama_guard_held', 'ollama_guard_unavailable']) {
        const held = mount({ OPENAI_COMPATIBLE_BASE_URL: 'http://lab.lan/v1' }, () => REFUSAL(code, 'Ollama 0.12.5 can silently truncate history. Upgrade to 0.12.6 or later.'));
        await held.test(held.row());
        assert.deepEqual([held.view(held.row()).offered, held.view(held.row()).line.dataset.testCode], [false, code]);
    }

    // A draft that already holds a consent (an install's, or the wizard's own after Back): shown ticked, as loaded; withdrawing is an edit.
    const loaded = mount({ OPENAI_COMPATIBLE_BASE_URL: 'http://lab.lan/v1', OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED: 'http://lab.lan/v1' });
    const stood = loaded.row();
    assert.deepEqual([loaded.view(stood).offered, loaded.view(stood).why, loaded.view(stood).ticked, loaded.editor.consent('openai-compatible')], [true, 'ticked', true, null]);
    await loaded.tick(stood, false);
    assert.equal(loaded.editor.consent('openai-compatible'), '');
    await loaded.type(stood, 'base_url', 'http://other.lan/v1');
    assert.deepEqual([loaded.view(stood).ticked, loaded.editor.consent('openai-compatible')], [false, ''], 'a new address withdraws the consent that stood');
    await loaded.type(stood, 'base_url', 'http://lab.lan/v1');
    assert.deepEqual([loaded.view(stood).ticked, loaded.editor.consent('openai-compatible')], [true, null], 'typed back, it covers the address again');
    // A key that serves no private server has no such box.
    assert.equal(loaded.root.querySelector('[data-connection-row][data-kind="openrouter"] [data-connection-consent]'), null);
    for (const each of [wizard, named, loaded]) each.editor.destroy();
});

test('an added key\'s card row draws its API: a new one is saved and tested on Responses, a loaded one keeps what it stored', async (t) => {
    const { createConnectionsEditor } = await import('../modules/settings_connections.js');
    const world = server(t, { values: { OUROBOROS_CONNECTIONS: [
        { kind: 'openai', name: 'lab', label: 'Lab', api_key: 'sk-lab', api_surface: 'responses' },
        { kind: 'openai', name: 'old', label: 'Old', api_key: 'sk-old' }] } });
    const element = globalThis.Element;
    globalThis.Element = NodeStub;
    t.after(() => { globalThis.Element = element; });
    const root = new Stub('div');
    root.innerHTML = `<div data-settings-panel="providers">${PROVIDER_CARDS.map(providerSettingsCard).join('')}</div>`;
    const sent = [];
    const editor = createConnectionsEditor({ root, test: async (body) => { sent.push(body); return { ok: true }; } });
    const document = world.state.document();
    editor.load(document, { modelSlots: document._meta.setup_contract.modelSlots });
    const panel = root.querySelector('[data-settings-panel="providers"]');
    const row = (name) => root.querySelectorAll('[data-connection-row]').find((each) => each.dataset.name === name);
    const shown = (each) => each.querySelector('[data-connection-choice-value][aria-pressed="true"]')?.dataset.connectionChoiceValue;
    assert.equal(root.querySelectorAll('[data-connection-row][data-builtin="1"] [data-connection-choice]').length, 0,
        'the built-in key\'s choice stands on its own page and under the wizard\'s flat key');
    assert.deepEqual([shown(row('lab')), shown(row('old'))], ['responses', 'chat_completions'], 'a stored word, or none: Chat');
    assert.deepEqual(editor.collect(), {}, 'loading changes nothing');
    await panel.fire('click', { target: row('old').querySelector('[data-connection-test]') });
    assert.deepEqual(sent.pop(), { provider_id: 'openai', connection: 'old', overrides: {} }, 'a legacy row is tested as saved');
    await panel.fire('click', { target: root.querySelector('[data-connection-add="openai"]') });
    const fresh = root.querySelectorAll('[data-connection-row][data-draft="1"]').at(-1);
    assert.equal(shown(fresh), 'responses', 'a new key starts where its save would put it');
    fresh.querySelector('[data-connection-label]').value = 'Work';
    fresh.querySelector('[data-connection-field="api_key"]').value = 'sk-work';
    await panel.fire('click', { target: fresh.querySelector('[data-connection-test]') });
    assert.equal(sent.pop().overrides.api_surface, 'responses', 'Test asks on the API Save stores');
    assert.deepEqual(editor.collect().OUROBOROS_CONNECTIONS.at(-1), { kind: 'openai', name: 'work', label: 'Work', api_key: 'sk-work', api_surface: 'responses' });
    await panel.fire('click', { target: fresh.querySelector('[data-connection-choice-value="chat_completions"]') });
    assert.equal(editor.collect().OUROBOROS_CONNECTIONS.at(-1).api_surface, 'chat_completions', 'the owner\'s press is what travels');
    await panel.fire('click', { target: row('old').querySelector('[data-connection-choice-value="chat_completions"]') });
    assert.equal(editor.collect().OUROBOROS_CONNECTIONS.find((each) => each.name === 'old').api_surface, undefined,
        'pressing the option a legacy row already shows stores nothing');
    editor.destroy();
});

/* ---- An OpenAI key's API (settings_provider_cards.js: a choice among a kind's fields) -------------------------------
   The stored words are the selected contract's: the flat `OPENAI_API_SURFACE`, a row's `api_surface`, `chat_completions`
   or `responses`. A key saved before the choice existed stores neither and keeps Chat Completions. */
const OPENAI = PROVIDER_CARDS.find((card) => card.kind === 'openai');
const API = OPENAI.fields.find((field) => field.choices);
const NOTES = Object.fromEntries(API.choices.map((choice) => [choice.value, choice.note]));
/** The API strip of the key page mounted in `box`: what it shows, what travels, and a press. */
function apiStrip(box) {
    const input = () => box.querySelector('[data-connection-field="api_surface"]');
    const options = () => box.querySelectorAll('[data-connection-choice-value]');
    return {
        input,
        shown: () => options().filter((option) => option.getAttribute('aria-pressed') === 'true').map((option) => option.textContent),
        marked: () => options().filter((option) => option.classList.contains('active')).map((option) => option.textContent),
        held: () => options().map((option) => option.disabled),
        note: () => box.querySelector('[data-connection-choice-note]').textContent,
        tone: () => box.querySelector('[data-connection-choice-note]').dataset.tone || '',
        press: async (value) => { await box.fire('click', { target: options().find((option) => option.dataset.connectionChoiceValue === value) }); await settle(); },
    };
}
const pressOn = async (box, selector) => { await box.fire('click', { target: box.querySelector(selector) }); await settle(); };

test('an OpenAI key saved before the choice keeps Chat Completions; only a pressed change is tested and saved, alone', async (t) => {
    const world = server(t, { values: { OPENAI_API_KEY: 'sk-openai-real', OUROBOROS_MODEL: 'openai::gpt-kept' } });
    const page = await world.mount('accounts/key/openai');
    const api = apiStrip(world.container);
    assert.deepEqual([api.shown(), api.marked(), api.input().value], [['Chat Completions'], ['Chat Completions'], ''],
        'no stored word reads as Chat Completions, and nothing stands to be sent');
    assert.equal(api.note(), NOTES.chat_completions);
    assert.match(world.container.textContent, /Test checks a short tool call through the API chosen above\./);
    // An ordinary save of another field is no migration: the key travels alone.
    const key = world.find('[data-connection-field="api_key"]');
    key.value = 'sk-openai-new';
    await world.container.fire('input', { target: key });
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts(), [{ OPENAI_API_KEY: 'sk-openai-new' }]);
    assert.equal('OPENAI_API_SURFACE' in world.state.values, false, 'the choice was not written by the way');
    // A press is a draft: it ends what the last Test said of the key as it was, Test asks with it, nothing is saved
    // until Save, and the line says what the option means.
    const verdict = () => [world.find('[data-editor-result]').textContent, keyHeads(world.state.document()).keys.find((head) => head.id === 'openai').chip];
    await pressOn(world.container, '[data-connection-test]');
    assert.deepEqual(verdict(), ['Not ready — No credits', { text: 'Not ready — No credits', tone: 'warn' }]);
    await api.press('responses');
    assert.deepEqual(verdict(), ['', null], 'the verdict was about the draft that was tested');
    assert.deepEqual([api.shown(), api.input().value, api.note()], [['Responses'], 'responses', NOTES.responses]);
    assert.equal(world.posts().length, 1, 'a press writes nothing');
    await pressOn(world.container, '[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai', overrides: { OPENAI_API_SURFACE: 'responses' } });
    // Pressing the option the key already holds undoes the draft: nothing is left to send, and Test asks the saved key.
    await api.press('chat_completions');
    assert.deepEqual([api.shown(), api.input().value], [['Chat Completions'], '']);
    await pressOn(world.container, '[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai', overrides: {} });
    await pressOn(world.container, '[data-editor-save]');
    assert.equal(world.posts().length, 1);
    assert.equal(world.find('[data-editor-result]').textContent, 'Nothing changed.');
    // The deliberate change: one key in the body, and the page shows what is stored after it.
    await api.press('responses');
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts().at(-1), { OPENAI_API_SURFACE: 'responses' });
    assert.match(world.find('[data-editor-result]').textContent, /^Saved/);
    assert.deepEqual([apiStrip(world.container).shown(), apiStrip(world.container).input().value], [['Responses'], 'responses']);
    assert.equal(world.state.values.OUROBOROS_MODEL, 'openai::gpt-kept', 'the model a role names is left as it was');
    assert.match(keyHeads(world.state.document()).keys.find((head) => head.id === 'openai').meta, /^Responses API · Main$/);
    // Back to the earlier API is stored as its own word: the key now has a choice on record.
    await apiStrip(world.container).press('chat_completions');
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts().at(-1), { OPENAI_API_SURFACE: 'chat_completions' });
    page.unmount();
});

test('a new OpenAI key starts on Responses and stores it with the key; the proposal is never sent alone', async (t) => {
    const world = server(t);
    // The built-in key that holds nothing yet (More providers): the strip proposes Responses.
    let page = await world.mount('accounts/key/openai');
    let api = apiStrip(world.container);
    assert.deepEqual([api.shown(), api.input().value, api.note()], [['Responses'], 'responses', NOTES.responses]);
    await pressOn(world.container, '[data-editor-save]');
    assert.equal(world.posts().length, 0, 'a proposal nobody pressed is stored with the key, never alone');
    assert.equal(world.find('[data-editor-result]').textContent, 'Nothing changed.');
    await pressOn(world.container, '[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai', overrides: { OPENAI_API_SURFACE: 'responses' } }, 'Test asks through what a save would store');
    const key = world.find('[data-connection-field="api_key"]');
    key.value = 'sk-openai-first';
    await world.container.fire('input', { target: key });
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts(), [{ OPENAI_API_KEY: 'sk-openai-first', OPENAI_API_SURFACE: 'responses' }]);
    assert.deepEqual(apiStrip(world.container).shown(), ['Responses']);
    page.unmount();
    // A key added beside it, the provider chosen first: its row carries the word.
    page = await world.mount('accounts/key/new');
    assert.equal(world.find('[data-connection-choice]'), null, 'a kind without the choice draws no strip');
    Object.assign(world.find('[data-key-kind]'), { value: 'openai' });
    await world.container.fire('change', { target: world.find('[data-key-kind]') });
    api = apiStrip(world.container);
    assert.deepEqual([api.shown(), api.input().value], [['Responses'], 'responses']);
    world.find('[data-connection-label]').value = 'Work account';
    world.find('[data-connection-field="api_key"]').value = 'sk-work';
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts().at(-1).OUROBOROS_CONNECTIONS.at(-1), { kind: 'openai', name: 'work-account', label: 'Work account', api_key: 'sk-work', api_surface: 'responses' });
    page.unmount();
    // The owner may start a new key on the earlier API: that press is stored as its own word.
    page = await world.mount('accounts/key/new');
    Object.assign(world.find('[data-key-kind]'), { value: 'openai' });
    await world.container.fire('change', { target: world.find('[data-key-kind]') });
    await apiStrip(world.container).press('chat_completions');
    world.find('[data-connection-label]').value = 'Old tools';
    world.find('[data-connection-field="api_key"]').value = 'sk-old';
    await pressOn(world.container, '[data-editor-save]');
    assert.equal(world.posts().at(-1).OUROBOROS_CONNECTIONS.at(-1).api_surface, 'chat_completions');
    page.unmount();
    // Beside an older setup (the legacy base address this key reaches) the built-in key starts on the earlier API, as
    // the server seeds it, and the word shown is stored with the key.
    const older = server(t, { values: { OPENAI_BASE_URL: 'http://127.0.0.1:8080/v1' } });
    page = await older.mount('accounts/key/openai');
    api = apiStrip(older.container);
    assert.deepEqual([api.shown(), api.input().value, api.input().dataset.seeded], [['Chat Completions'], 'chat_completions', '1']);
    const reached = older.find('[data-connection-field="api_key"]');
    reached.value = 'sk-openai-legacy';
    await older.container.fire('input', { target: reached });
    await pressOn(older.container, '[data-editor-save]');
    assert.deepEqual(older.posts(), [{ OPENAI_API_KEY: 'sk-openai-legacy', OPENAI_API_SURFACE: 'chat_completions' }]);
    page.unmount();
});

test('named OpenAI keys choose on their own; an unknown stored word is said; the environment\'s and the organization\'s keys keep the rule', async (t) => {
    const world = server(t, { values: { OPENAI_API_KEY: 'sk-openai-real', OPENAI_API_SURFACE: 'responses', OUROBOROS_CONNECTIONS: [
        { kind: 'openai', name: 'work', label: 'Work account', api_key: 'sk-work' },
        { kind: 'openai', name: 'lab', label: 'Lab', api_key: 'sk-lab', api_surface: 'responses' },
        { kind: 'openai', name: 'odd', label: 'Odd', api_key: 'sk-odd', api_surface: 'Response' }] } });
    const metaOf = (id) => keyHeads(world.state.document()).keys.find((head) => head.id === id).meta;
    assert.deepEqual(['openai', 'openai@work', 'openai@lab', 'openai@odd'].map(metaOf),
        ['Responses API · No role uses it', 'Chat Completions API · No role uses it', 'Responses API · No role uses it', 'Unknown API · No role uses it']);
    assert.equal(keyHeads(world.state.document()).keys.find((head) => head.id === 'openrouter').meta, 'Main', 'a kind without the choice says nothing of it');
    // The key saved before the choice: its own row changes, the others and the built-in key stay as they were.
    let page = await world.mount('accounts/key/openai/work');
    let api = apiStrip(world.container);
    assert.deepEqual([api.shown(), api.input().value], [['Chat Completions'], '']);
    world.find('[data-connection-label]').value = 'Work';
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts().at(-1).OUROBOROS_CONNECTIONS[0], { kind: 'openai', name: 'work', label: 'Work', api_key: MASK }, 'renaming it writes no choice');
    await apiStrip(world.container).press('responses');
    await pressOn(world.container, '[data-connection-test]');
    assert.deepEqual(world.state.askedTests.at(-1), { provider_id: 'openai', connection: 'work', overrides: { api_surface: 'responses' } });
    await pressOn(world.container, '[data-editor-save]');
    assert.deepEqual(world.posts().at(-1).OUROBOROS_CONNECTIONS.map((row) => row.api_surface), ['responses', 'responses', 'Response']);
    assert.equal(world.state.values.OPENAI_API_SURFACE, 'responses');
    page.unmount();
    // A stored word that is neither option: neither is shown pressed, the line says so, and nothing travels by itself.
    page = await world.mount('accounts/key/openai/odd');
    api = apiStrip(world.container);
    assert.deepEqual([api.shown(), api.marked(), api.tone()], [[], [], 'error']);
    assert.equal(api.note(), 'Saved value “Response” is none of these. Choose one.');
    await api.press('responses');
    assert.deepEqual([api.shown(), api.tone(), api.note()], [['Responses'], '', NOTES.responses]);
    await pressOn(world.container, '[data-editor-save]');
    assert.equal(world.posts().at(-1).OUROBOROS_CONNECTIONS[2].api_surface, 'responses');
    page.unmount();
    // The server's refusal of a word is said on the page and the pressed option stays.
    page = await world.mount('accounts/key/openai');
    await apiStrip(world.container).press('chat_completions');
    world.state.refuse = [{ saved: false, error: 'OPENAI_API_SURFACE must be chat_completions or responses.' }, 400];
    await pressOn(world.container, '[data-editor-save]');
    assert.match(world.find('[data-editor-result]').textContent, /^Not saved\. OPENAI_API_SURFACE must be chat_completions or responses\./);
    assert.deepEqual(apiStrip(world.container).shown(), ['Chat Completions']);
    page.unmount();

    // A key the environment provides, with no word in the document, is an existing key: Chat Completions, not the seed.
    const fromEnvironment = server(t, { values: { OPENAI_API_KEY: '' } });
    const document = fromEnvironment.state.document;
    fromEnvironment.state.document = () => {
        const answer = document();
        Object.assign(answer._meta.connections.find((entry) => entry.id === 'openai'), { configured: true, source: 'environment' });
        return answer;
    };
    page = await fromEnvironment.mount('accounts/key/openai');
    api = apiStrip(fromEnvironment.container);
    assert.deepEqual([api.shown(), api.input().value, api.input().dataset.fresh || ''], [['Chat Completions'], '', '']);
    assert.equal(keyHeads(fromEnvironment.state.document()).keys.find((head) => head.id === 'openai').meta, 'From the environment · Chat Completions API · No role uses it');
    page.unmount();

    // The organization's Supervisor sends through the key: the choice is shown, never changed.
    const managed = server(t, { values: { OPENAI_API_KEY: 'sk-openai-real' } });
    const plain = managed.state.document;
    managed.state.document = () => ({ ...plain(), _meta: { ...plain()._meta, policy_state: { deployment: { checker: { connections: ['openai'], fields: [] } } } } });
    page = await managed.mount('accounts/key/openai');
    api = apiStrip(managed.container);
    assert.deepEqual([api.shown(), api.held()], [['Chat Completions'], [true, true]]);
    assert.equal(managed.find('[data-editor-save]'), null);
    page.unmount();
});

test('the wizard\'s flat OpenAI key: a key typed where none was starts on Responses, a configured one keeps what it stored', async () => {
    const fields = [{ id: 'openrouter-key', stateKey: 'openrouterKey', settingKey: 'OPENROUTER_API_KEY' },
        { id: 'openai-key', stateKey: 'openaiKey', settingKey: 'OPENAI_API_KEY' }];
    const mounted = ({ initial, held }) => {
        const state = { ...initial };
        let edits = 0;
        const choices = createFlatKeyChoices({ state, initial, providerFields: fields, held, onChange: () => { edits += 1; } });
        const root = new Stub('div');
        root.innerHTML = `<input id="openai-key">${choices.html('OPENAI_API_KEY')}`;
        choices.bind(root);
        const holder = root.querySelector('[data-flat-key-choice]');
        const type = async (text) => { state.openaiKey = text; await root.querySelector('#openai-key').fire('input', { target: { value: text } }); };
        return { state, choices, root, holder, api: apiStrip(holder), type, edits: () => edits };
    };
    assert.equal(createFlatKeyChoices({ state: {}, providerFields: fields }).html('OPENROUTER_API_KEY'), '', 'a kind without the choice draws nothing');
    // A fresh install: the strip waits for a key, comes with it on Responses, and the draft stores the word with the key.
    let wizard = mounted({ initial: { openaiKey: '', openaiApiSurface: '' } });
    assert.equal(wizard.holder.hidden, true);
    assert.deepEqual([wizard.choices.draft(), wizard.choices.words('openaiKey')], [{}, []], 'no key, no choice');
    await wizard.type('sk-typed');
    assert.equal(wizard.holder.hidden, false);
    assert.deepEqual([wizard.api.shown(), wizard.choices.draft(), wizard.choices.words('openaiKey')], [['Responses'], { OPENAI_API_SURFACE: 'responses' }, ['Responses API']]);
    await wizard.api.press('chat_completions');
    assert.deepEqual([wizard.api.shown(), wizard.choices.draft(), wizard.edits()], [['Chat Completions'], { OPENAI_API_SURFACE: 'chat_completions' }, 1]);
    assert.match(wizard.choices.html('OPENAI_API_KEY'), /class="ui-segment active" data-connection-choice-value="chat_completions" aria-pressed="true"/, 'a later render draws the pressed option');
    await wizard.type('');
    assert.deepEqual([wizard.holder.hidden, wizard.choices.draft()], [true, {}], 'the key gone, its choice is no part of the draft');
    // An install that already has a key (the bootstrap's configured marker, from the document or the environment) and
    // no stored word: Chat Completions, and an untouched choice sends nothing.
    wizard = mounted({ initial: { openaiKey: '***configured***' } });
    assert.deepEqual([wizard.holder.hidden, wizard.api.shown(), wizard.choices.draft(), wizard.choices.words('openaiKey')],
        [false, ['Chat Completions'], {}, ['Chat Completions API']]);
    await wizard.type('sk-replaced');
    assert.deepEqual([wizard.api.shown(), wizard.choices.draft()], [['Chat Completions'], {}], 'replacing its key does not switch it');
    await wizard.api.press('responses');
    assert.deepEqual(wizard.choices.draft(), { OPENAI_API_SURFACE: 'responses' });
    await wizard.api.press('chat_completions');
    assert.deepEqual([wizard.choices.draft(), wizard.state.openaiApiSurface], [{}, ''], 'pressed back, nothing is left to send');
    // A stored word is kept as it is; the organization's key shows it and takes no press.
    wizard = mounted({ initial: { openaiKey: '***configured***', openaiApiSurface: 'responses' }, held: (kind, key) => kind === 'openai' && key === 'OPENAI_API_SURFACE' });
    assert.deepEqual([wizard.api.shown(), wizard.api.held(), wizard.choices.draft()], [['Responses'], [true, true], {}]);
    // The setup contract lists the stored word as a field of its own: it is this strip, never a text field.
    const listed = createFlatKeyChoices({ state: {}, providerFields: [...fields, { id: 'openai-api-surface', stateKey: 'openaiApiSurface', settingKey: 'OPENAI_API_SURFACE' }] });
    assert.deepEqual([listed.owns({ settingKey: 'OPENAI_API_SURFACE' }), listed.owns({ settingKey: 'OPENAI_API_KEY' })], [true, false]);
    // Beside an older setup a typed key starts on the earlier API, and the draft stores that word with it.
    wizard = mounted({ initial: { openaiKey: '', legacyCompatibleBaseUrl: 'http://127.0.0.1:8080/v1' } });
    await wizard.type('sk-typed');
    assert.deepEqual([wizard.api.shown(), wizard.choices.draft()], [['Chat Completions'], { OPENAI_API_SURFACE: 'chat_completions' }]);
    // The profile a proposal on one key takes is that key's own choice: the built-in key's as the draft holds it, an
    // added key's as its row stores it (connectionDefaults asks per key, never one table for the whole kind).
    wizard = mounted({ initial: { openaiKey: '', connections: [] } });
    wizard.state.connections = [{ kind: 'openai', name: 'lab', api_surface: 'responses' }, { kind: 'openai', name: 'old' }];
    assert.equal(wizard.choices.profileOf('openai'), '', 'no OpenAI key in the draft: the kind\'s own');
    await wizard.type('sk-typed');
    assert.equal(wizard.choices.profileOf('openai'), 'openai-responses');
    assert.deepEqual([wizard.choices.profileOf('openai', 'lab'), wizard.choices.profileOf('openai', 'old')], ['openai-responses', ''],
        'an added key answers by its own stored word; none is the earlier API');
    await wizard.api.press('chat_completions');
    assert.equal(wizard.choices.profileOf('openai'), '', 'the earlier API proposes the kind\'s own defaults');
    assert.equal(wizard.choices.profileOf('anthropic', 'corp'), '', 'a kind without the choice has one profile');
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

