// The home of Settings (docs/DESIGN.md "Settings"): what Ouroboros's card lists is decided by one pure function from
// facts that were read; a row's answer says its fact or that it was not read; and showing the home starts nothing: no
// daemon, no model catalogue, no review pool, no tool-server refresh, no status poll. A page reads when it is shown.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import { renderPageHeader } from '../modules/page_header.js';
import { initSettings } from '../modules/settings.js';
import { readSettings } from '../modules/settings_controls.js';
import { askOuroboros, beingState, homeAnswers, mountSettingsHome, needRows } from '../modules/settings_home.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { ANSWERS, STARTS, Stub, settle, world } from './helpers/settings_dom.js';

const account = (harness, id, verification, enabled = true) => ({
    harness, profile_id: id, display_name: `${id}@example.com`, enabled, identity: {}, status: { verification, verification_source: 'vendor' },
});
const accounts = (rows, read = 'ok', kind = 'refresh') => ({ read, rows, kind, snapshot: { harnesses: [{ id: 'codex', display_name: 'Codex' }] } });
const restartState = (state = {}) => ({ _meta: { restart_state: { restart_required: false, restart_keys: [], restart_source_unknown_keys: [],
    local_model: { pending_keys: [] }, ...state } } });
const kinds = (rows) => rows.map((row) => `${row.kind}:${row.id}`);

test('accounts that need sign-in again are one row per harness, only over an accounts facet that was read', () => {
    const rows = needRows({ accounts: accounts([
        account('codex', 'a', 'failed'), account('codex', 'b', 'failed'), account('codex', 'c', 'passed'),
        account('codex', 'off', 'failed', false), account('codex', 'unknown', 'not_run'), account('claude', 'solo', 'expired'),
    ]) });
    assert.deepEqual(kinds(rows), ['sign-in:codex', 'sign-in:claude']);
    assert.equal(rows[0].text, '2 Codex accounts need sign-in again');
    assert.equal(rows[1].text, 'Claude Code account solo@example.com needs sign-in again');
    assert.deepEqual(rows.map((row) => [row.owner, row.action, row.open]), [[true, 'Sign in', 'accounts/codex'], [true, 'Sign in', 'accounts/claude']]);
    // Read, and nobody needs sign-in: no row, and no sentence that nothing waits.
    assert.deepEqual(needRows({ accounts: accounts([account('codex', 'c', 'passed'), account('codex', 'off', 'failed', false)]) }), []);
    // Not read: the same failing rows are a memory. The card lists no row for them (what was not read is said by the
    // row it concerns, never as a line of the card with a Check), and the Accounts row prints no count from them.
    for (const read of ['unread', 'not_read', 'failed', 'transport', 'indeterminate']) {
        assert.deepEqual(needRows({ accounts: accounts([account('codex', 'a', 'failed')], read) }), [], read);
        assert.equal(homeAnswers({ accounts: accounts([account('codex', 'a', 'failed')], read) }).accounts, undefined, read);
    }
    // A read that found the daemon unasked is no row either: nothing on the card offers to start it.
    assert.deepEqual(needRows({ accounts: accounts([], 'not_read', 'wake') }), []);
    assert.deepEqual(needRows({}), [], 'nothing read yet says nothing');
});

test('a tool server that needs sign-in is a row; a status that could not be read is no row, and its MCP servers row says Not checked', () => {
    const servers = [
        { id: 'slack', name: 'Slack', enabled: true, sign_in: { state: 'needed', reason: 'expired' } },
        { id: 'docs', name: 'Docs', enabled: true, sign_in: { state: 'signed_in', reason: 'insufficient_scope', required_scope: 'write' } },
        { id: 'git', name: 'GitHub', enabled: true, sign_in: { state: 'signed_in' } },
        { id: 'off', name: 'Off', enabled: false, sign_in: { state: 'needed' } },
        { id: 'local', name: 'Local', enabled: true },
    ];
    const rows = needRows({ tools: { enabled: true, servers } });
    assert.deepEqual(kinds(rows), ['tool-sign-in:slack', 'tool-sign-in:docs']);
    assert.deepEqual([rows[0].text, rows[0].open, rows[0].owner], ['MCP server Slack needs sign-in', 'tools/slack', true]);
    assert.deepEqual(needRows({ tools: { enabled: true, servers: servers.slice(2) } }), []);
    // A status that could not be read is no line of the card: the MCP servers row says it was not checked.
    assert.deepEqual(needRows({ tools: null }), []);
    assert.deepEqual(homeAnswers({ tools: null }).tools, { text: 'Not checked', owner: false });
    assert.deepEqual(needRows({ tools: undefined }), []);
});

test('saved settings that wait for a restart are one row; a local model\'s pending change is its own, not a restart', () => {
    const named = needRows({ settings: restartState({ restart_required: true, restart_keys: ['OUROBOROS_SKILLS_REPO_PATH'] }) });
    assert.deepEqual(named.map((row) => [row.kind, row.text, row.action, row.run, row.owner]),
        [['restart', 'Restart to apply: Skills folder', 'Restart', 'restart', true]]);
    // A key no row names is counted, never guessed.
    const counted = needRows({ settings: restartState({ restart_required: true, restart_keys: ['OUROBOROS_SKILLS_REPO_PATH', 'NO_ROW_NAMES_THIS'] }) });
    assert.equal(counted[0].text, 'Restart to apply 2 saved settings');
    assert.equal(needRows({ settings: restartState({ restart_source_unknown_keys: ['OUROBOROS_SERVER_HOST'] }) })[0].text,
        'The saved server address may need a restart');
    assert.deepEqual(needRows({ settings: restartState() }), [], 'nothing pending: the row has left');
    const local = needRows({ settings: restartState({ local_model: { pending_keys: ['LOCAL_MODEL_PORT'] } }) });
    assert.deepEqual(local.map((row) => [row.kind, row.open, row.action, row.run]), [['local-model', 'models/local', 'Open', undefined]]);
    // A settings document that could not be read is no line of the card: the rows it answers say it was not checked.
    assert.deepEqual(needRows({ settings: null }), []);
    assert.deepEqual(homeAnswers({ settings: null }).models, { text: 'Not checked', owner: false });
});

test('a row answers its fact, says when it was not read, and marks the owner\'s move', () => {
    const settings = { OUROBOROS_MODEL: 'openrouter::z-ai/glm-5.3', OUROBOROS_SAFETY_OWNER_RULES: ' never email clients ',
        OUROBOROS_SUBAGENTS: { enabled: true, items: [{ subagent_id: 'a' }, { subagent_id: 'b', enabled: false }, { subagent_id: 'c' }] },
        _meta: { policy_state: { access: { configured: 'cyber_pro', current_process: 'pro', restart_required: true } },
            connections: [{ id: 'openrouter', configured: true }, { id: 'openai', configured: false }, { id: 'anthropic', configured: true }] } };
    const state = { spent_usd: 500, budget_limit: 500, accounting: { available: true }, main_attention: [{ attention_reason: 'budget' }],
        evolution_enabled: null, bg_consciousness_enabled: true, bg_consciousness_state: { status: 'allowance_exhausted', detail: 'Stopped by its spending limit' } };
    const answers = homeAnswers({ settings, state, accounts: accounts([account('codex', 'a', 'passed')]),
        tools: { enabled: true, servers: [{ enabled: true, tool_count: 3 }, { enabled: false, tool_count: 0 }, { name: 'broken' }] },
        skills: { skills: [{}, {}] }, places: { places: [{}] }, storage: { count: null }, autostart: { state: 'off' },
        update: { managed: true, check_ok: true, checked_at: '2026-11-11T10:00:00Z', current_version: '7.10.2', available: true, target_version: '7.11.0' },
        device: { theme: 'dark', notify: { enabled: true, sound: false }, dictation: '' } });
    const text = (to) => answers[to]?.text;
    assert.equal(text('models'), 'glm-5.3');
    assert.equal(text('agents'), '2 of 3 on');
    assert.equal(text('rules'), 'Written');
    assert.equal(text('permissions'), 'Pro now, Cyber Pro after restart');
    assert.equal(text('accounts'), '1 account, 2 API keys');
    assert.equal(text('tools'), '1 of 3 servers on');
    assert.equal(text('page:skills'), '2 installed');
    assert.equal(text('page:circles'), '1 place');
    assert.equal(text('dashboard:storage'), 'Not measured');
    assert.equal(text('dashboard:evolution'), 'State unknown');
    assert.equal(text('computer'), 'Starts by hand');
    assert.equal(text('appearance'), 'Dark');
    assert.equal(text('notifications'), 'On, silent');
    assert.equal(text('voice'), 'Dictation not chosen');
    // The owner's move: a stop by the spending limit, a consciousness held by its allowance, an update that waits.
    assert.deepEqual(['dashboard:costs', 'dashboard:consciousness', 'dashboard:updates', 'models', 'rules'].map((to) => answers[to].owner),
        [true, true, true, false, false]);
    assert.equal(text('dashboard:consciousness'), 'Stopped by its spending limit');
    // The same rows without the trouble carry no mark; an empty rules text says Empty.
    const calm = homeAnswers({ settings: { ...settings, OUROBOROS_SAFETY_OWNER_RULES: '' },
        state: { ...state, main_attention: [], bg_consciousness_state: { status: 'idle', detail: 'Awake' } }, update: { managed: true, check_ok: true, current_version: '7.10.2' } });
    assert.deepEqual(['dashboard:costs', 'dashboard:consciousness', 'dashboard:updates'].map((to) => calm[to].owner), [false, false, false]);
    assert.equal(calm.rules.text, 'Empty');
    // Main whose key cannot send, or that waits for the owner to choose one of two keys, says so in the owner's colour.
    const main = (patch) => homeAnswers({ settings: { ...settings, ...patch, _meta: { ...settings._meta, ...patch._meta } } }).models;
    assert.deepEqual(main({ _meta: { connections: [{ id: 'openrouter', configured: true, enabled: false }] } }), { text: 'Its key is off', owner: true });
    assert.deepEqual(main({ _meta: { connections: [] } }), { text: 'No key for it', owner: true });
    assert.deepEqual(main({ OUROBOROS_MODEL: 'openai::gpt-x', _meta: { main_connection_choice: ['openai@work', 'openai@home'] } }),
        { text: 'Choose its key', owner: true });
    assert.deepEqual(main({ USE_LOCAL_MAIN: true, _meta: { connections: [] } }).owner, false, 'a local Main needs no key');
    // Not read yet: no answer is claimed. A read that failed says so, and an unread accounts facet never prints a count.
    assert.deepEqual(homeAnswers({}), {});
    const failed = homeAnswers({ settings: null, state: null, tools: null, storage: null, update: null, skills: null, places: null });
    for (const to of ['models', 'agents', 'rules', 'permissions', 'tools', 'page:skills', 'dashboard:storage', 'dashboard:costs', 'dashboard:updates']) {
        assert.equal(failed[to].text, 'Not checked', to);
    }
    assert.equal(homeAnswers({ settings, accounts: accounts([account('codex', 'a', 'passed')], 'not_read') }).accounts.text, '2 API keys');
    // A sign-in is the owner's move on the row too, only over what was read.
    const signIn = (read) => homeAnswers({ settings, accounts: accounts([account('codex', 'a', 'failed')], read) }).accounts.owner;
    assert.deepEqual([signIn('ok'), signIn('not_read'), answers.accounts.owner], [true, false, false]);
    const needy = { enabled: true, servers: [{ enabled: true, tool_count: 1, sign_in: { state: 'needed' } }] };
    assert.deepEqual([homeAnswers({ tools: needy }).tools.owner, answers.tools.owner], [true, false]);
    assert.equal(homeAnswers({ settings: { _meta: {} } }).permissions.text, 'Not reported');
});

test('This computer and Updates answer on every host: who can open it where the host starts nothing, the version until the status is read', () => {
    const listens = (reachability) => ({ _meta: { reachability } });
    const computer = (facts) => homeAnswers(facts).computer?.text;
    // The desktop app knows how it starts: that is the answer, whatever listens.
    assert.equal(computer({ autostart: { state: 'on' }, settings: listens('lan_reachable') }), 'Starts at sign-in');
    // A browser or a server host starts nothing itself (`unavailable`): the row says who can open it, from the settings read.
    assert.equal(computer({ autostart: { state: 'unavailable' }, settings: listens('loopback_only') }), 'Open only here');
    assert.equal(computer({ autostart: { state: 'unavailable' }, settings: listens('lan_reachable') }), 'Open to your network');
    assert.equal(computer({ autostart: { state: 'unavailable' }, settings: listens('host_ip_unknown') }), 'Open to your network');
    assert.equal(computer({ autostart: null, settings: listens('loopback_only') }), 'Open only here', 'a failed host read still leaves the settings\' fact');
    // Nothing is claimed before the host answered (no word that the next read would replace), nor from a document
    // that does not say what listens; a read that failed with nothing else to say says so.
    assert.equal(computer({ settings: listens('loopback_only') }), undefined);
    assert.equal(computer({ autostart: { state: 'unavailable' }, settings: { _meta: {} } }), undefined);
    assert.equal(computer({ autostart: { state: 'unavailable' } }), undefined);
    assert.equal(computer({ autostart: null, settings: null }), 'Not checked');
    assert.equal(computer({ autostart: { state: 'unavailable' }, settings: null }), 'Not checked');
    // Updates: the running version from the state snapshot until the status read answers, then what that read says.
    const updates = (facts) => homeAnswers(facts)['dashboard:updates']?.text;
    assert.equal(updates({ state: { version: '7.10.2' } }), 'Version 7.10.2');
    assert.equal(updates({ state: {} }), undefined, 'no version, no claim');
    assert.equal(updates({ state: { version: '7.10.2' }, update: { managed: true, check_ok: true, current_version: '7.10.2' } }), 'Up to date · 7.10.2');
    assert.equal(updates({ state: { version: '7.10.2' }, update: null }), 'Not checked', 'a status read that failed says so');
    // An update that waits says what it brings, in the owner's colour: the row's name already says Updates, and the
    // list's one line keeps the number whole ("Update available: 163 ne…" lost it).
    const waiting = { managed: true, check_ok: true, available: true, safe_to_apply: true, current_version: '7.10.2' };
    assert.deepEqual(homeAnswers({ update: { ...waiting, latest_version: '7.10.2', behind: 163 } })['dashboard:updates'], { text: '163 new changes', owner: true });
    assert.equal(updates({ update: { ...waiting, latest_version: '7.11.0' } }), '7.10.2 → 7.11.0');
});

test('the card says who it is and its state, never that nothing waits', () => {
    assert.deepEqual(beingState({ connected: true, state: { supervisor_ready: true } }), { text: 'Online', tone: 'ok' });
    assert.deepEqual(beingState({ connected: false, state: { supervisor_ready: true } }), { text: 'Reconnecting…', tone: 'muted' });
    assert.equal(beingState({ connected: true, state: { supervisor_ready: false, boot: { phase: 'failed', error: 'x' } } }).tone, 'danger');
});

test('showing the home reads only what is passive: no catalogue, no review pool, no wake, no refresh, no status poll', async (t) => {
    const { env, content } = world(t);
    const statusReads = [];
    const store = createClaudexorStatusStore({ fetchImpl: async (url) => { statusReads.push(url); return { ok: false, status: 503, json: async () => ({}) }; } });
    t.after(() => store.dispose());
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({ supervisor_ready: true }) } } });
    t.after(() => home.dispose());
    await home.refresh();
    await settle();
    assert.deepEqual([...new Set(env.calls)].sort(), Object.keys(ANSWERS).sort(), 'the home\'s own reads, each a GET that starts nothing');
    for (const url of STARTS) assert.ok(!env.calls.some((call) => String(call).startsWith(url)), url);
    assert.deepEqual(statusReads, [], 'the accounts store is read as it stands, never asked');
    assert.equal(store.subscriberCount, 1);
    assert.equal(store.polling, false, 'the home is no visible surface of the store: it arms no poll');
    // The card: an accounts facet nobody read is no line of it (no Check that would read or wake), and nothing claims
    // that nothing waits.
    const card = content.querySelector('[data-settings-card]');
    assert.deepEqual(card.querySelectorAll('[data-needs]').map((row) => [row.dataset.needs, row.dataset.needId]), []);
    assert.equal(card.querySelector('[data-need-action]'), null);
    assert.equal(card.querySelector('[data-being-state]').textContent, 'Online');
    assert.equal(card.querySelector('[data-being-count]').hidden, true);
    assert.doesNotMatch(card.textContent, /nothing/i);
    const answer = (to) => content.querySelector(`[data-settings-open="${to}"]`).querySelector('[data-settings-answer]').textContent;
    // The Accounts row says nothing it did not read: no count of accounts, and no keys from a document that lists none.
    assert.deepEqual(['models', 'rules', 'page:skills', 'dashboard:storage', 'tools', 'accounts'].map(answer),
        ['glm-5.3', 'Empty', '1 installed', 'Not measured', 'No MCP servers', '']);
    await settle();
    assert.deepEqual(statusReads, [], 'still no read of the status and no wake');
});

test('a page reads when it is shown, not when Settings is', async (t) => {
    const { env, content, state, showPage } = world(t);
    const reads = [];
    env.routes.set('/api/settings', async () => { reads.push(1); return { ok: true, status: 200, json: async () => ANSWERS['/api/settings'] }; });
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    await controls.open('');
    await settle();
    const atHome = reads.length;
    assert.ok(atHome >= 1 && !env.calls.some((call) => STARTS.some((url) => String(call).startsWith(url))));
    assert.equal(content.querySelectorAll('.settings-page').length, 0);
    // The skill sources page (settings_extension_forms.js): its own read fires for its own path only.
    assert.ok(settingsPageFor('skill-sources'));
    await controls.open('skill-sources');
    await settle();
    assert.ok(reads.length > atHome, 'opening the page read its settings');
    const before = reads.length;
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'models' } }));
    await settle();
    assert.equal(reads.length, before, 'another page being shown is not this page\'s read');
    const level = content.querySelector('.settings-page');
    assert.equal(level.dataset.settingsPage, 'skill-sources');
    assert.ok(level.querySelector('[data-setting-row="OUROBOROS_SKILLS_REPO_PATH"]'));
});

test('the Context door shows Main and nothing else: with a project room open it posts no context mode', async (t) => {
    const { env, posts, state, shown, showPage } = world(t);
    state.activePage = 'settings';
    const dashboard = [];
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {} });
    // A project room is open over Main: showPage('chat') with no options is what closes it (app.js showPage).
    await controls.open('main:context');
    await settle();
    assert.deepEqual(shown, [['chat']], 'Main is shown, the open Project closed by the page change itself');
    assert.deepEqual(posts, [], 'no POST /api/owner/context-mode, no write of any kind');
    assert.ok(!env.calls.some((call) => String(call).includes('context-mode')));
    assert.deepEqual(dashboard, []);
    // A door to a Dashboard tab asked for from outside Settings (a capsule's limit door) carries its row and no way
    // back: nobody came from Settings.
    await controls.open('dashboard:costs', { row: 'TOTAL_BUDGET' });
    assert.deepEqual(dashboard, [['costs', { row: 'TOTAL_BUDGET' }]]);
});

// The fixture's element, also reading `:scope > X` (a child) as a browser does: a door's way back finds the page's own
// header so (dashboard.js settingsReturn). Children an innerHTML builds are made of the same element.
class ScopeStub extends Stub {
    querySelectorAll(selector) {
        const scoped = selector.match(/^:scope > (\S+)$/);
        if (scoped) return this.children.filter((child) => child.matches(scoped[1]));
        return super.querySelectorAll(selector);
    }
    set innerHTML(html) {
        super.innerHTML = html;
        const upgrade = (node) => { Object.setPrototypeOf(node, ScopeStub.prototype); node.children.forEach(upgrade); };
        this.children.forEach(upgrade);
    }
    get innerHTML() { return super.innerHTML; }
    toggleAttribute(name, force) {
        const on = force ?? !this.hasAttribute(name);
        if (on) this.setAttribute(name, ''); else this.removeAttribute(name);
        return on;
    }
}

test('a door pressed inside Settings leads back to it: a Dashboard tab by its path, a page of its own by its header Back', async (t) => {
    const { env, content, state, shown, showPage } = world(t);
    env.document.createElement = (tag) => new ScopeStub(tag);
    const dashboard = [];
    const layered = [];
    const layers = { push: (layer) => layered.push(`push ${layer.id}`), remove: (id) => layered.push(`remove ${id}`), batch: (fn) => fn() };
    // The two pages as their modules draw them: the shared header (renderPageHeader), Circles with its action and the
    // Dashboard's shell, Skills with its tabs and its own parts under the header.
    const pageOf = (id, html) => {
        const page = env.document.createElement('div');
        page.id = id;
        page.className = 'page app-page-glass';
        page.innerHTML = html;
        content.appendChild(page);
        return page;
    };
    const circles = pageOf('page-circles', `${renderPageHeader({ page: 'circles', actionsHtml: '<button type="button">New circle</button>' })}
        <div class="dashboard-shell"><div class="dashboard-body"></div></div>`);
    const skills = pageOf('page-skills', `${renderPageHeader({ page: 'skills', tabsHtml: '<div class="skills-tabs"></div>' })}
        <div class="skills-search-chrome"></div><div class="skills-scroll"></div>`);
    const partsOf = (page) => page.children.map((part) => part.className);
    const skillsParts = partsOf(skills);
    const controls = initSettings({ state, ws: env.ws, showPage, layers, openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {} });
    await controls.open('');
    await settle();
    const homeEl = content.querySelector('[data-settings-home]');
    const press = async (to) => { await homeEl.fire('click', { target: homeEl.querySelector(`[data-settings-open="${to}"]`) }); await settle(); };
    const headerOf = (page) => page.querySelector(':scope > .app-page-header');
    // What stands in the header's Back slot, and whether the header says it holds one.
    const backOf = (page) => {
        const slot = headerOf(page).querySelector('[data-page-back]');
        return { inSlot: slot.children.map((node) => node.getAttribute('aria-label')),
            marked: headerOf(page).hasAttribute('data-back'), rows: page.querySelectorAll('.dashboard-back').length };
    };
    const none = { inSlot: [], marked: false, rows: 0 };
    const back = { inSlot: ['Back to Settings'], marked: true, rows: 0 };

    // A Dashboard tab: the path the door was pressed on travels with it, and the tab draws the row.
    await press('dashboard:costs');
    assert.deepEqual(dashboard, [['costs', { row: '', back: '' }]]);

    // Circles: Back stands in its header's Back slot, never as a row of its own, with its one layer.
    await press('page:circles');
    assert.deepEqual(shown.at(-1), ['circles']);
    assert.deepEqual(backOf(circles), back);
    assert.equal(layered.at(-1), 'push settings-door-back', 'Escape and the system\'s Back return to Settings too');
    // Its press returns to Settings, and the slot is emptied and its layer gone.
    await headerOf(circles).querySelector('[data-dashboard-back]').fire('click', {});
    await settle();
    assert.deepEqual([shown.at(-1), backOf(circles)], [['settings'], none]);
    assert.ok(layered.includes('remove settings-door-back'));

    // Skills: in its header's slot as well; the page's own parts stay as they were.
    await press('page:skills');
    assert.deepEqual(backOf(skills), back);
    assert.deepEqual(partsOf(skills), skillsParts);
    // Leaving that page any other way empties the slot.
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'files' } }));
    assert.deepEqual(backOf(skills), none);

    // The same door asked for from outside Settings shows the page and no way back.
    await controls.open('page:circles');
    await settle();
    assert.deepEqual([shown.at(-1), backOf(circles)], [['circles'], none]);
});

test('Ask Ouroboros puts the words into Main as an unsent draft and replaces no draft that is there', async () => {
    const events = [];
    const input = { value: '', dispatchEvent: (event) => events.push(event.type), focus() {} };
    const doc = { querySelector: (selector) => (selector === '#page-chat #chat-input' ? input : null) };
    const pages = [];
    assert.equal(await askOuroboros('never email my clients', { showPage: async (name) => { pages.push(name); return true; }, doc }), true);
    assert.deepEqual([input.value, events, pages], ['never email my clients', ['input'], ['chat']]);
    input.value = 'half a thought';
    await askOuroboros('quiet', { showPage: async () => true, doc });
    assert.deepEqual([input.value, events], ['half a thought', ['input']], 'a draft already there stays, and nothing is written');
    assert.equal(await askOuroboros('x', { showPage: async () => false, doc }), false, 'a vetoed leave shows nothing');
});

test('a choice of this device made beside the list is its row\'s answer at once, read after its own module stored it', async (t) => {
    const { env, content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    const kept = new Map();
    const before = globalThis.localStorage;
    globalThis.localStorage = { getItem: (key) => kept.get(key) ?? null, setItem: (key, value) => { kept.set(key, String(value)); } };
    t.after(() => { globalThis.localStorage = before; });
    env.window.ouroTheme = { choice: 'dark' };
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({}) } } });
    t.after(() => home.dispose());
    await home.refresh();
    await settle();
    const answer = (to) => content.querySelector(`[data-settings-open="${to}"]`).querySelector('[data-settings-answer]').textContent;
    assert.deepEqual([answer('appearance'), answer('notifications')], ['Dark', 'Off on this device']);
    const inside = (attr, value = '') => { const control = new Stub('div'); control.setAttribute(attr, value); const hit = new Stub('button'); control.appendChild(hit); return hit; };
    // The theme's strip: theme.js stored the choice from the document's handler, and the window hears the press after it.
    env.window.ouroTheme.choice = 'light';
    env.window.dispatchEvent({ type: 'click', target: inside('data-theme-control') });
    assert.equal(answer('appearance'), 'Light');
    env.window.ouroTheme.choice = 'system';
    env.window.dispatchEvent({ type: 'keydown', target: inside('data-theme-control') });
    assert.equal(answer('appearance'), 'System', 'an arrow in the strip chooses too');
    // A notification switch: the notifier stored its prefs on the change.
    kept.set('ouroboros.notifications', JSON.stringify({ enabled: true, sound: false }));
    env.window.dispatchEvent({ type: 'change', target: inside('data-notify-pref', 'enabled') });
    assert.equal(answer('notifications'), 'On, silent');
    // A press anywhere else reads nothing again.
    env.window.ouroTheme.choice = 'dark';
    env.window.dispatchEvent({ type: 'click', target: inside('data-other') });
    assert.equal(answer('appearance'), 'System');
    home.dispose();
    env.window.dispatchEvent({ type: 'click', target: inside('data-theme-control') });
    assert.equal(answer('appearance'), 'System', 'a disposed home hears nothing');
});

test('the list marks the page that is open: the top level\'s own row, else the nearest level under it that has one', async (t) => {
    const { content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({}) } } });
    t.after(() => home.dispose());
    const chosen = () => content.querySelectorAll('[data-settings-open]').filter((row) => row.getAttribute('aria-current') === 'page')
        .map((row) => row.dataset.settingsOpen);
    home.mark(['agents', 'permissions']);
    assert.deepEqual(chosen(), ['agents'], 'Permissions\' Reviewers door opened Agents over it');
    home.mark(['agents/reviewer', 'agents', 'permissions']);
    assert.deepEqual(chosen(), ['agents'], 'an agent\'s page is Agents\'');
    home.mark(['skill/voice-form', 'voice']);
    assert.deepEqual(chosen(), ['voice'], 'a skill\'s form has no row: the page under it keeps its own');
    home.mark('models');
    assert.deepEqual(chosen(), ['models']);
    home.mark([]);
    assert.deepEqual(chosen(), []);
});
