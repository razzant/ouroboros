// The home of Settings (docs/DESIGN.md "Settings"): what Ouroboros's card lists is decided by one pure function from
// facts that were read; a row's answer says its fact or that it was not read; and showing the home starts nothing: no
// daemon, no model catalogue, no review pool, no tool-server refresh, no status poll. A page reads when it is shown.
import assert from 'node:assert/strict';
import { settingsDoorName } from '../modules/settings_pages.js';
import test from 'node:test';

test('a door names its destination, without adding words to an internal row', () => {
    assert.equal(settingsDoorName('Costs', 'dashboard:costs'), 'Costs, Dashboard');
    assert.equal(settingsDoorName('Context', 'main:context'), 'Context, Main');
    assert.equal(settingsDoorName('Skills', 'page:skills'), 'Skills');
    assert.equal(settingsDoorName('Models', 'models'), 'Models');
});

import { STATUS_READ_LIFETIME_MS, createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import { getContributions } from '../modules/contributions.js';
import { createHostStart } from '../modules/host_starting_status.js';
import { modelCatalog } from '../modules/model_catalog_store.js';
import { initSettings } from '../modules/settings.js';
import { bindSettingRows, readSettings, settingsRow, writeEntry } from '../modules/settings_controls.js';
import { askOuroboros, beingState, homeAnswers, mountSettingsHome, needRows } from '../modules/settings_home.js';
import { registerSettingsPage, settingsPageFor } from '../modules/settings_pages.js';
import { rowById } from '../modules/settings_rows.js';
import { createStateSnapshotSequencer, readStateSnapshot } from '../modules/state_snapshots.js';
import { ANSWERS, STARTS, Stub, settle, world } from './helpers/settings_dom.js';
// The MCP servers page registers its family when it is imported (in the app, extensions.js imports it):
// the home's Extensions door and its status writer are tested here without the section itself.
import '../modules/mcp_settings.js';

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

test('a tool server that needs sign-in is a row; unreadable MCP status is named apart from the skills count', () => {
    const servers = [
        { id: 'slack', name: 'Slack', enabled: true, tool_count: 0, sign_in: { state: 'needed', reason: 'expired' } },
        { id: 'docs', name: 'Docs', enabled: true, tool_count: 0, sign_in: { state: 'signed_in', reason: 'insufficient_scope', required_scope: 'write' } },
        { id: 'git', name: 'GitHub', enabled: true, tool_count: 3, sign_in: { state: 'signed_in' } },
        { id: 'off', name: 'Off', enabled: false, tool_count: 0, sign_in: { state: 'needed' } },
        { id: 'local', enabled: true, tool_count: 0, code: 'MCP_CONFIG_ERROR' },
    ];
    const rows = needRows({ tools: { enabled: true, servers } });
    assert.deepEqual(kinds(rows), ['tool-sign-in:slack', 'tool-sign-in:docs']);
    // Its old address: Settings opens it in Extensions › MCP servers, with Back to Settings (settings.js).
    assert.deepEqual([rows[0].text, rows[0].open, rows[0].owner], ['MCP server Slack needs sign-in', 'tools/slack', true]);
    assert.deepEqual(needRows({ tools: { enabled: true, servers: servers.slice(2) } }), []);
    // A status that could not be read is no line of the card: the Extensions row says it was not checked.
    assert.deepEqual(needRows({ tools: null }), []);
    assert.deepEqual(homeAnswers({ tools: null })['page:skills'], { text: 'MCP status unavailable', owner: false });
    const skills = { skills: [{}, {}] };
    assert.deepEqual(homeAnswers({ skills })['page:skills'], { text: '2 skills', owner: false });
    assert.deepEqual(homeAnswers({ skills, tools: null })['page:skills'], { text: '2 skills · MCP status unavailable', owner: false });
    assert.equal(homeAnswers({ tools: null }).tools, undefined, 'Settings has no Tools row now');
    // A server waiting for its sign-in is the owner's move on Extensions' one door.
    assert.deepEqual(homeAnswers({ tools: { enabled: true, servers } })['page:skills'], { text: '3 of 5 MCP servers on', owner: true });
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
    // Extensions' one door says both kinds it holds, each by its own name.
    assert.equal(text('page:skills'), '2 skills · 1 of 3 MCP servers on');
    assert.equal(text('tools'), undefined);
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
    for (const to of ['models', 'agents', 'rules', 'permissions', 'dashboard:storage', 'dashboard:costs', 'dashboard:updates']) {
        assert.equal(failed[to].text, 'Not checked', to);
    }
    assert.deepEqual(failed['page:skills'], { text: 'Not checked · MCP status unavailable', owner: false });
    assert.equal(homeAnswers({ settings, accounts: accounts([account('codex', 'a', 'passed')], 'not_read') }).accounts.text, '2 API keys');
    // A sign-in is the owner's move on the row too, only over what was read.
    const signIn = (read) => homeAnswers({ settings, accounts: accounts([account('codex', 'a', 'failed')], read) }).accounts.owner;
    assert.deepEqual([signIn('ok'), signIn('not_read'), answers.accounts.owner], [true, false, false]);
    const needy = { enabled: true, servers: [{ enabled: true, tool_count: 1, sign_in: { state: 'needed' } }] };
    assert.deepEqual([homeAnswers({ tools: needy })['page:skills'].owner, answers['page:skills'].owner], [true, false]);
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
    assert.deepEqual(beingState({ connected: true, ready: true }), { text: 'Online', tone: 'ok' });
    assert.deepEqual(beingState({ connected: false, ready: true }), { text: 'Reconnecting…', tone: 'muted' });
    assert.deepEqual(beingState({ connected: true, ready: false, boot: { phase: 'failed', error: 'x' } }), { text: 'Start failed', tone: 'danger' });
    // Only a body that said ready is Online: none read yet, or one forgotten at a disconnect, keeps Starting….
    assert.deepEqual(beingState({ connected: true }), { text: 'Starting…', tone: 'muted' });
    assert.deepEqual(beingState({ connected: true, ready: false, boot: null }), { text: 'Starting…', tone: 'muted' });
    assert.deepEqual(beingState({ connected: true, ready: false, boot: {
        phase: 'serving', dialogue_ready: true, steps: [{ name: 'startup:task_recovery', state: 'running' }],
    } }), { text: 'Chat ready · restoring saved work', tone: 'muted' }, 'the card has room for the step the bar may drop');
});

test('a failed start is the card\'s first line: its cause beside it and Restart, never only a word', async (t) => {
    const failed = { supervisor_ready: false, boot: { phase: 'failed', error: 'Supervisor init failed: queue RPC failed', steps: [] } };
    assert.deepEqual(needRows({ state: failed }), [{ kind: 'start', id: 'start', owner: true, action: 'Restart', run: 'restart',
        text: 'Ouroboros could not finish starting.', note: 'Supervisor init failed: queue RPC failed' }]);
    for (const state of [{ supervisor_ready: true, boot: failed.boot }, { supervisor_ready: false, boot: { phase: 'serving', steps: [] } }, null]) {
        assert.deepEqual(needRows({ state }), [], 'readiness outranks the phase, and a start that goes on is no need');
    }
    // The page's newest snapshot reaches the card while it shows (app.js applies each): the start's word follows it.
    const { content } = world(t);
    const starting = { supervisor_ready: false, boot: { phase: 'serving', dialogue_ready: true, steps: [{ name: 'extensions', state: 'running' }] } };
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => starting } } });
    t.after(() => home.dispose());
    await home.refresh();
    await settle();
    const stateEl = content.querySelector('[data-being-state]');
    assert.equal(stateEl.textContent, 'Chat ready · loading extensions');
    home.stateApplied(failed);
    assert.equal(stateEl.textContent, 'Start failed');
    const startRow = content.querySelector('[data-needs="start"]');
    assert.equal(startRow.querySelector('.ui-row-meta').textContent, 'Supervisor init failed: queue RPC failed');
    assert.equal(startRow.querySelector('[data-need-action]').textContent, 'Restart');
    home.stateApplied({ supervisor_ready: true });
    assert.equal(stateEl.textContent, 'Online');
    assert.equal(content.querySelector('[data-needs="start"]'), null);
});

// The card's word against an outage (#3376 M2): a live socket proves only the transport, as for the room's bar
// (docs/DESIGN.md "Foreground and state colour"), so Online needs a snapshot that states readiness.
test('the card says Online only from a snapshot that states readiness; a close and a reconnect forget older snapshots', async (t) => {
    // The card's word for one body, as its reducer (createHostStart, the header's) reads it.
    const cardWord = (state) => {
        const start = createHostStart();
        start.read(state);
        return beingState({ connected: true, ready: start.ready(), boot: start.status().bootPhase });
    };
    for (const state of [null, {}, { supervisor_error: 'Supervisor init failed: x' }]) {
        assert.deepEqual(cardWord(state), { text: 'Starting…', tone: 'muted' }, JSON.stringify(state));
    }
    // A head start that failed and was replaced leaves its failed step in a ready boot: readiness outranks it.
    const recovered = { supervisor_ready: true, boot: { phase: 'ready', dialogue_ready: true, tasks_ready: true,
        steps: [{ name: 'startup:event_queue', state: 'failed', error: 'OSError: the first event-bus Queue() RPC is refused' }] } };
    assert.deepEqual([cardWord(recovered), needRows({ state: recovered })], [{ text: 'Online', tone: 'ok' }, []]);
    const { content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    let connected = true;
    let latest = { supervisor_ready: true };
    const handlers = new Map();
    const ws = { isConnected: () => connected, on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws, stateSnapshots: { latest: () => latest } } });
    t.after(() => home.dispose());
    await home.refresh(['settings']);
    const word = () => content.querySelector('[data-being-state]').textContent;
    assert.equal(word(), 'Online');
    connected = false;
    latest = null;  // the page's sequencer retires its bodies at every socket change (app.js)
    handlers.get('close')({});
    assert.equal(word(), 'Reconnecting…');
    // A valid HTTP poll can finish while the socket is down, answered by the server from before a restart.
    latest = { supervisor_ready: true };
    home.stateApplied(latest);
    assert.equal(word(), 'Reconnecting…');
    connected = true;
    latest = null;  // open retires that intermediate reply in the page's sequencer
    handlers.get('open')({ previouslyConnected: true });
    assert.equal(word(), 'Starting…', 'the reconnect forgets that reply, as the header does: it may not be this server\'s');
    home.stateApplied({ supervisor_ready: true });
    assert.equal(word(), 'Online', 'a body read after the reconnect proves readiness again');
    home.stateApplied({ supervisor_ready: false, boot: { phase: 'serving', dialogue_ready: true, steps: [{ name: 'extensions', state: 'running' }] } });
    assert.equal(word(), 'Chat ready · loading extensions', 'a newer snapshot speaks at once, without a reload');
    home.hide();  // off screen (settings.js): the newest snapshot is kept, nothing is painted
    home.stateApplied({ supervisor_ready: false, boot: { phase: 'failed', error: 'Supervisor init failed: x', steps: [] } });
    assert.equal(word(), 'Chat ready · loading extensions');
    await home.refresh([]);
    assert.equal(word(), 'Start failed', 'shown again, the card says the newest snapshot it was given');
    home.stateApplied({ supervisor_ready: true });
    assert.equal(word(), 'Online');
});

test('a state read from before an outage never repaints the card: the page\'s one sequencer orders the replies', async (t) => {
    const { env, content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    let connected = true;
    let home = null;
    const handlers = new Map();
    const ws = { isConnected: () => connected, on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    // As in app.js: every body the sequencer applies reaches the card; a socket change retires the older reads.
    const snapshots = createStateSnapshotSequencer((data) => home?.stateApplied(data));
    const answers = [];
    const body = (data) => ({ ok: true, status: 200, json: async () => data });
    env.routes.set('/api/state', () => new Promise((resolve) => { answers.push(resolve); }));
    home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws, stateSnapshots: snapshots } });
    t.after(() => home.dispose());
    const shown = home.refresh(['settings', 'state']);  // the list comes on screen: its state read is in flight
    await settle();
    assert.equal(answers.length, 1);
    connected = false;
    snapshots.retire();
    handlers.get('close')({});
    connected = true;
    snapshots.retire();
    handlers.get('open')({ previouslyConnected: true });
    const reread = snapshots.gate(true).then((request) => request && readStateSnapshot(snapshots, request));  // app.js's read after an open
    await settle();
    assert.equal(answers.length, 2);
    const word = () => content.querySelector('[data-being-state]').textContent;
    assert.equal(word(), 'Starting…', 'reconnected, nothing read yet');
    answers[1](body({ supervisor_ready: false, boot: { phase: 'serving', dialogue_ready: true,
        steps: [{ name: 'startup:worker_spawn', state: 'running' }] } }));
    await reread;
    assert.equal(word(), 'Chat ready · starting executors');
    answers[0](body({ supervisor_ready: true }));  // the older generation answers last
    await shown;
    await settle();
    assert.equal(word(), 'Chat ready · starting executors', 'an older generation never paints over a newer one');
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
    // A start is any read under these addresses but the passive ones the home reads, which ANSWERS lists exactly.
    for (const url of STARTS) assert.ok(!env.calls.some((call) => String(call).startsWith(url) && !Object.hasOwn(ANSWERS, String(call))), url);
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
    assert.deepEqual(['models', 'rules', 'page:skills', 'dashboard:storage', 'accounts'].map(answer),
        ['glm-5.3', 'Empty', '1 skill · No MCP servers', 'Not measured', '']);
    await settle();
    assert.deepEqual(statusReads, [], 'still no read of the status and no wake');
});

// The Models door says Main's source by the catalogue a model sheet already holds: a descriptor that names a
// credential harness draws that mark, and one adopter later repaints that one glyph — never a read, never a whole list.
test('the Models door wears its descriptor\'s mark, and a late descriptor repaints only that glyph', async (t) => {
    // Already held when the home is first shown: the true mark at once, and nothing read for it.
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity', credentialHarness: 'agy' }] });
    const { env, content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    env.routes.set('/api/settings', async () => ({ ok: true, status: 200, json: async () => ({ ...ANSWERS['/api/settings'],
        OUROBOROS_MODEL: 'claudexor::agy=claude-opus-5-5' }) }));
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({}) } } });
    t.after(() => home.dispose());
    await home.refresh(['settings']);
    await settle();
    const glyph = (to) => content.querySelector(`[data-settings-open="${to}"]`).querySelector('[data-settings-glyph]');
    const drawn = () => glyph('models').dataset.glyph;
    assert.match(drawn(), /viewBox="0 0 180 180"/, 'a held descriptor draws its mark at first paint');
    assert.ok(!env.calls.some((call) => String(call).startsWith('/api/model-catalog')), 'showing the home still reads no catalogue');
    // A held list that names no harness (another window's read, a cleared one): the cube, never a brand guessed from
    // an opaque id that happens to spell one.
    const rules = glyph('rules').dataset.glyph, reads = env.calls.length;
    modelCatalog.adopt({ items: [], model_sources: [] });
    await settle();
    assert.doesNotMatch(drawn(), /viewBox="0 0 180 180"/, 'no descriptor: the neutral cube');
    // A descriptor adopted later brands the route it names: one glyph, nothing read, no other row touched.
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity', credentialHarness: 'agy' }] });
    await settle();
    assert.match(drawn(), /viewBox="0 0 180 180"/, 'the descriptor\'s Antigravity mark');
    assert.equal(glyph('rules').dataset.glyph, rules, 'another row\'s glyph is untouched');
    assert.equal(env.calls.length, reads, 'a mark reads nothing');
    // The same descriptor again writes nothing: the drawn node stays the one the owner sees.
    const node = glyph('models').firstChild;
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity', credentialHarness: 'agy' }] });
    await settle();
    assert.equal(glyph('models').firstChild, node, 'an unchanged result writes nothing');
    // A descriptor that names no credential harness is no evidence of one: the cube returns.
    modelCatalog.adopt({ items: [], model_sources: [{ id: 'agy', label: 'Antigravity', credentialHarness: '' }] });
    await settle();
    assert.doesNotMatch(drawn(), /viewBox="0 0 180 180"/, 'an unbound opaque id is never branded');
});

test('the card\'s first word follows the socket: an open or a close repaints it and reads nothing', async (t) => {
    // The home can finish its reads before the socket's first open; since it reads only when it comes on screen, that
    // open alone must move the card off Reconnecting.
    const { env, content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    let connected = false;
    const handlers = new Map();
    const ws = { isConnected: () => connected, on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws, stateSnapshots: { latest: () => ({ supervisor_ready: true }) } } });
    t.after(() => home.dispose());
    await home.refresh();
    await settle();
    const word = () => content.querySelector('[data-settings-card]').querySelector('[data-being-state]').textContent;
    assert.equal(word(), 'Reconnecting…', 'painted while the socket was down');
    const reads = env.calls.length;
    connected = true;
    handlers.get('open')?.({});
    assert.equal(word(), 'Online', 'the first open repaints the card');
    connected = false;
    handlers.get('close')?.({});
    assert.equal(word(), 'Reconnecting…', 'a close repaints it back');
    connected = true;
    handlers.get('open')?.({ previouslyConnected: true });  // before app.js's retire: the newest body is still the old one
    assert.equal(word(), 'Starting…', 'a reconnect repaints it without reading back the body from before it, ready as it is');
    await settle();
    assert.equal(env.calls.length, reads, 'none of them reads anything');
    home.dispose();
    assert.equal(handlers.has('open') || handlers.has('close'), false, 'dispose lets the socket go');
});

test('the card keeps the header\'s readiness rule over the page\'s sequencer: a reconnect never paints a retired Online', async (t) => {
    // The home is mounted at start and reads only when it comes on screen. app.js retires the page's sequencer on the
    // socket's close and open before the home's own handlers run (it registers first); the card and the list's own
    // Ouroboros row follow every body the sequencer applies (`watch`), keep what the last one said over a failed read,
    // and forget it at the close and again at a reconnect, never at the first open, as the header's createHostStart
    // does (chat.js): a poll applied while the socket was down may be the answer of the server from before a restart.
    const { env, content } = world(t);
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    const snapshots = createStateSnapshotSequencer(() => {});
    let connected = false;
    const handlers = new Map();
    const ws = { isConnected: () => connected, on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    let opened = false;  // ws.js: an open after an earlier one is a reconnect (`previouslyConnected`)
    const emit = (type) => {  // app.js's handler, then the home's
        snapshots.retire();
        handlers.get(type)?.(type === 'open' ? { previouslyConnected: opened } : {});
        if (type === 'open') opened = true;
    };
    const apply = async (body) => snapshots.apply(await snapshots.gate(false), body);
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws, stateSnapshots: snapshots } });
    t.after(() => home.dispose());
    const words = () => [content.querySelector('[data-settings-card]').querySelector('[data-being-state]').textContent,
        content.querySelector('[data-overview-answer]').textContent];
    const said = (word) => [word, word];
    assert.deepEqual(words(), said('Reconnecting…'), 'painted while the socket was down');
    connected = true;
    emit('open');
    assert.deepEqual(words(), said('Starting…'), 'open, no state read yet: not Online');
    await apply({ supervisor_ready: true });
    assert.deepEqual(words(), said('Online'), 'a body the page read repaints the card: the home read nothing for it');
    snapshots.fail(await snapshots.gate(false));
    assert.deepEqual(words(), said('Online'), 'a failed read states nothing: the card keeps the last word, as the header does');
    await apply({ version: '7.10.2' });
    assert.deepEqual(words(), said('Online'), 'neither does a body without supervisor_ready');
    await apply({ supervisor_ready: true });
    await home.refresh([]);  // the home comes on screen while ready: its own copy now holds that Online body
    connected = false;
    emit('close');
    assert.deepEqual(words(), said('Reconnecting…'));
    connected = true;
    emit('open');
    assert.deepEqual(words(), said('Starting…'), 'the reconnect forgot the old Online: a restart is no longer Online here');
    const held = await snapshots.gate(false);
    assert.deepEqual(words(), said('Starting…'), 'still Starting… while the fresh read is in flight');
    snapshots.apply(held, { supervisor_ready: false, boot: { phase: 'failed', error: 'init failed' } });
    assert.deepEqual(words(), ['Start failed', 'Start failed · One thing needs you'],
        'the restart failed: the card says so, and its own row (#3376) is the one thing that needs the owner');
    await apply({ supervisor_ready: true });
    assert.deepEqual(words(), said('Online'));
    connected = false;
    emit('close');
    await apply({ supervisor_ready: true });  // a state poll the server from before a restart answered while the socket was down
    assert.deepEqual(words(), said('Reconnecting…'));
    connected = true;
    emit('open');
    assert.deepEqual(words(), said('Starting…'), 'the reconnect forgets that poll, as the header does: the server that answers now may be new');
    snapshots.fail(await snapshots.gate(false));
    assert.deepEqual(words(), said('Starting…'), 'a failed read after the reconnect proves nothing');
    await apply({ version: '7.10.2' });
    assert.deepEqual(words(), said('Starting…'), 'nor does a body that states no readiness');
    await apply({ supervisor_ready: true });
    assert.deepEqual(words(), said('Online'), 'a body read after the reconnect proves readiness again');
    // A failed start a poll read while the socket was down is forgotten with it: its Restart leaves with the word, and
    // returns only with a body read after the reconnect that says the start failed.
    connected = false;
    emit('close');
    await apply({ supervisor_ready: false, boot: { phase: 'failed', error: 'the old start failed' } });
    connected = true;
    emit('open');
    assert.deepEqual(words(), said('Starting…'));
    assert.equal(content.querySelector('[data-needs="start"]'), null, 'the failure read in the outage is no need of the server now');
    await apply({ supervisor_ready: false, boot: { phase: 'failed', error: 'init failed again' } });
    assert.deepEqual(words(), ['Start failed', 'Start failed · One thing needs you']);
    await apply({ supervisor_ready: true });
    connected = false;
    handlers.get('close')?.({});  // the home's close handler first this time, then app.js's retire
    snapshots.retire();
    connected = true;
    emit('open');
    assert.deepEqual(words(), said('Starting…'), 'in either order, the close does not read back the body before it');
    await apply({ supervisor_ready: true });
    assert.deepEqual(words(), said('Online'));
    assert.equal(env.calls.length, 0, 'the home read nothing');
    home.dispose();
    await apply({ supervisor_ready: false, boot: { phase: 'failed', error: 'later' } });
    assert.deepEqual(words(), said('Online'), 'dispose lets the sequencer go: a later body no longer repaints the card');
});

test('a page reads when it is shown, not when Settings is', async (t) => {
    const { env, content, state, showPage } = world(t);
    const reads = [];
    env.routes.set('/api/settings', async () => { reads.push(1); return { ok: true, status: 200, json: async () => ANSWERS['/api/settings'] }; });
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    await controls.open('');
    await settle();
    const atHome = reads.length;
    assert.ok(atHome >= 1 && !env.calls.some((call) => STARTS.some((url) => String(call).startsWith(url) && !Object.hasOwn(ANSWERS, String(call)))));
    assert.equal(content.querySelectorAll('.settings-page').length, 0);
    // The rules page (settings_safety.js): its own read fires for its own path only. (The skill sources
    // page moved to Extensions; a page reading there is extensions_section.test.js's subject.)
    assert.ok(settingsPageFor('rules'));
    await controls.open('rules');
    await settle();
    assert.ok(reads.length > atHome, 'opening the page read its settings');
    const before = reads.length;
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { place: 'skills', path: 'tools' } }));
    await settle();
    assert.equal(reads.length, before, 'showing MCP servers in Extensions does not reread rules or the Settings list');
    const level = content.querySelector('.settings-page');
    assert.equal(level.dataset.settingsPage, 'rules');
    assert.ok(level.querySelector('[data-setting-row="OUROBOROS_SAFETY_OWNER_RULES"]'));
});

test('the Context door shows Main and nothing else: with a project room open it posts no context mode', async (t) => {
    const { env, posts, state, shown, showPage } = world(t);
    state.activePage = 'settings';
    const dashboard = [];
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {} });
    // A project room is open over Main: showPage('chat') with no options is what closes it (app.js showPage).
    await controls.open('main:context');
    await settle();
    assert.deepEqual(shown.map(([page]) => page), ['chat'], 'Main is shown, the open Project closed by the page change itself');
    assert.deepEqual(posts, [], 'no POST /api/owner/context-mode, no write of any kind');
    assert.ok(!env.calls.some((call) => String(call).includes('context-mode')));
    assert.deepEqual(dashboard, []);
    // A door carries its destination row; the shell, not Settings, owns its return.
    await controls.open('dashboard:costs', { row: 'TOTAL_BUDGET' });
    assert.deepEqual(dashboard, [['costs', { row: 'TOTAL_BUDGET' }]]);
});

test('all seven Settings doors use ordinary navigation; a page\'s door retains its level for the shell return', async (t) => {
    const { env, content, state, shown, showPage, layers } = world(t);
    const dashboard = [];
    const controls = initSettings({ state, ws: env.ws, showPage, layers,
        openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {},
        // app.js' openExtensions: the section opens through the shell's ordinary showPage, keyed to the
        // section (extensions.js openFromOutside), never a private callback.
        openExtensions: (path = '', options = {}) => showPage('skills', { ...options, key: `skills:${path}` }) });
    await controls.open('');
    await settle();
    const home = content.querySelector('[data-settings-home]');
    home.clientHeight = 500;
    home.getBoundingClientRect = () => ({ top: 0, bottom: 500 });
    home.querySelectorAll('[data-settings-open]').forEach((row) => { row.getBoundingClientRect = () => ({ top: 0, bottom: 40 }); });
    const doors = ['page:skills', 'page:circles', 'dashboard:costs', 'dashboard:consciousness',
        'dashboard:evolution', 'dashboard:updates', 'dashboard:storage'];
    for (const destination of doors) {
        const row = home.querySelector(`[data-settings-open="${destination}"]`);
        assert.ok(row, destination);
        await home.fire('click', { target: row });
        await settle();
        const [kind, name] = destination.split(':');
        if (destination === 'page:skills') assert.deepEqual(shown.at(-1), ['skills', { row: '', key: 'skills:' }]);
        else if (kind === 'page') assert.deepEqual(shown.at(-1), [name, {}]);
        else assert.deepEqual(dashboard.at(-1), [name, { row: '' }]);
    }
    // The page that moved to Extensions (MCP servers) opens there now. A Settings page with a door to
    // Main, drawn and bound by the real row machinery (settings_controls.js), keeps the level contract:
    // an ordinary jump, and the shell's return restores the same node.
    const doorRow = structuredClone(rowById('voice:dictation'));
    registerSettingsPage('test-door-main', { title: 'Door page', mount(container, { ctx }) {
        container.innerHTML = settingsRow(doorRow);
        const bound = bindSettingRows(container, { rows: [doorRow],
            openDoor: (door, opener) => ctx.openSettings?.(door.control.to, { opener }) });
        return { unmount: () => bound.dispose() };
    } });
    await controls.open('test-door-main');
    await settle();
    const origin = controls.capture();
    const level = content.querySelector('.settings-page');
    assert.equal(level.dataset.settingsPage, 'test-door-main');
    await level.querySelector('.settings-page-body').fire('click', { target: level.querySelector('[data-setting-row="voice:dictation"]') });
    assert.deepEqual(shown.at(-1), ['chat', {}], 'a level door needs no private Back callback');
    origin.restore();
    assert.equal(content.querySelector('.settings-page'), level, 'the retained level is restored');
    assert.equal(origin.title, 'Settings › Door page');
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

test('the Accounts row counts the accounts beside every page: the home\'s passive roster until a page read them', async (t) => {
    const { env, content } = world(t);
    const roster = (n) => ({ profiles: Array.from({ length: n }, (_, i) => ({ profile: { harness_id: 'codex', profile_id: `p${i}`,
        display_name: `p${i}`, enabled: true }, status: { verification: 'passed' }, identity: {} })) });
    env.routes.set('/api/settings', async () => ({ ok: true, status: 200,
        json: async () => ({ ...ANSWERS['/api/settings'], _meta: { connections: [{ configured: true }, { configured: true }] } }) }));
    env.routes.set('/api/claudexor/status?view=quota', async () => ({ ok: true, status: 200,
        json: async () => ({ view: 'quota', profiles: roster(3), quota: [], reads: { catalog: 'not_read', accounts: 'ok', quota: 'ok' } }) }));
    const statusReads = [];
    const store = createClaudexorStatusStore({ fetchImpl: async (url) => { statusReads.push(url); return { ok: true, status: 200,
        json: async () => ({ daemon: { state: 'running' }, harnesses: [], profiles: roster(4), quota: [], reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } }) }; } });
    t.after(() => store.dispose());
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({}) } } });
    t.after(() => home.dispose());
    await home.refresh();
    await settle();
    const answer = () => content.querySelector('[data-settings-open="accounts"]').querySelector('[data-settings-answer]').textContent;
    const passive = () => env.calls.filter((call) => call === '/api/claudexor/status?view=quota').length;
    // Beside Models, before any page read the shared store: the roster the home read itself, never the store asked.
    assert.deepEqual([answer(), statusReads], ['3 accounts, 2 API keys', []]);
    // A page that shows the accounts read the store (Agents, a provider's page): its roster answers, and the home stops
    // reading its own.
    await store.refresh();
    await settle();
    assert.equal(answer(), '4 accounts, 2 API keys');
    const before = passive();
    await home.refresh();
    await settle();
    assert.deepEqual([answer(), passive()], ['4 accounts, 2 API keys', before]);
});

test('a Settings door keeps one history entry from the list or a page, and a refused leave keeps its levels', async (t) => {
    const { createLayerStack } = await import('../modules/layers.js');
    const { createPageNavigation } = await import('../modules/page_navigation.js');
    const { registerSettingsPage } = await import('../modules/settings_pages.js');
    for (const origin of ['list', 'one-column', 'two-pane']) for (const door of ['page:skills', 'dashboard:costs']) for (const allowed of [true, false]) {
        await t.test(`${origin}/${door}/${allowed}`, async (sub) => {
            const { env, content, state } = world(sub);
            const calls = [];
            const layers = createLayerStack({ win: null, doc: null, history: {
                pushState: () => calls.push('push'), back: () => calls.push('back'),
            } });
            sub.after(() => layers.destroy());
            let leave, controls;
            registerSettingsPage('test-door-history', { title: 'Door', mount(_body, { ctx }) { leave = ctx.openSettings; } });
            const shown = name => {
                state.activePage = name;
                env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name } }));
            };
            const nav = createPageNavigation({ layers, doc: env.document,
                canLeave: async key => key.startsWith('settings:') || allowed,
                capture: () => {
                    const saved = controls.capture();
                    return { key: `settings:${state.settingsPath}`, label: 'Settings', title: saved.title,
                        restore() { state.activePage = 'settings'; saved.restore(); shown('settings'); } };
                } });
            sub.after(() => nav.destroy());
            const showPage = (name, options = {}) => nav.go(options.key || name,
                () => { shown(name); options.enter?.(); }, options);
            controls = initSettings({ state, ws: env.ws, layers, showPage,
                openDashboardTab: name => showPage('dashboard', { key: `dashboard:${name}` }),
                openExtensions: (path = '', options = {}) => showPage('skills', { ...options, key: `skills:${path}` }) });
            const home = content.querySelector('[data-settings-home]');
            home.getClientRects = () => origin === 'one-column' ? [] : [{}];
            await controls.open(origin === 'list' ? '' : 'test-door-history', { map: true });
            await settle();
            const hadLayer = origin === 'one-column';
            assert.deepEqual(calls, hadLayer ? ['push'] : []);
            const retained = content.querySelector('[data-settings-page="test-door-history"]');
            if (origin === 'list') await home.fire('click', { target: home.querySelector(`[data-settings-open="${door}"]`) });
            else await leave(door);
            await settle();
            assert.deepEqual(calls, hadLayer || allowed ? ['push'] : [], 'the handover never consumes and replaces an existing history entry');
            assert.equal(layers.size(), hadLayer || allowed ? 1 : 0);
            assert.equal(layers.top(), allowed ? 'page-return-1' : hadLayer ? 'settings-level-1' : null);
            assert.equal(state.activePage, allowed ? (door === 'page:skills' ? 'skills' : 'dashboard') : 'settings');
            if (allowed) {
                assert.equal(layers.returnPage().back.label, 'Settings');
                layers.back('escape'); await settle();
                assert.equal(state.activePage, 'settings');
                assert.equal(state.settingsPath, origin === 'list' ? '' : 'test-door-history');
                assert.equal(layers.top(), hadLayer ? 'settings-level-1' : null);
            }
            if (retained) assert.equal(content.querySelector('[data-settings-page="test-door-history"]'), retained, 'the same level survives leave or refusal');
        });
    }
});

// The list's reads (settings.js shows it, settings_home.js reads): its sources are read when it comes on screen, one
// read of a source at a time, dropped when it leaves the screen, and again for the row of a page that is left.
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const LIST = ['/api/mcp/status', '/api/extensions', '/api/places', '/api/update/status', '/api/storage', '/api/desktop/autostart',
    '/api/claudexor/status?view=quota'];
function listHome(t, content, ctx = {}) {
    const store = createClaudexorStatusStore({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({}) }) });
    t.after(() => store.dispose());
    const home = mountSettingsHome(content, { open() {}, store, ctx: { readSettings, ws: { isConnected: () => true }, stateSnapshots: { latest: () => ({}) }, ...ctx } });
    t.after(() => home.dispose());
    return { home, answer: (to) => content.querySelector(`[data-settings-open="${to}"]`).querySelector('[data-settings-answer]').textContent };
}

test('the list reads when it comes on screen: never at a press beside it, again once Settings was left, as one column returns to it', async (t) => {
    const { env, content, state, showPage } = world(t);
    let pageCtx = null;
    registerSettingsPage('test-list-a', { title: 'A', mount(body, { ctx }) { pageCtx = ctx; } });
    registerSettingsPage('test-list-b', { title: 'B', mount() {} });
    const observers = [];
    const kept = { observer: globalThis.ResizeObserver };
    globalThis.ResizeObserver = class { constructor(fn) { this.fn = fn; observers.push(this); } observe() {} disconnect() {} };
    const proto = Object.getPrototypeOf(Stub.prototype);
    Object.assign(kept, { box: proto.getBoundingClientRect, prepend: proto.prepend });
    proto.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: 0 });
    proto.prepend = function prepend(node) { this.insertBefore(node, this.children[0] || null); };
    t.after(() => { globalThis.ResizeObserver = kept.observer; Object.assign(proto, { getBoundingClientRect: kept.box, prepend: kept.prepend }); });
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    t.after(() => {
        const previousWindow = globalThis.window;
        const previousDocument = globalThis.document;
        globalThis.window = env.window;
        globalThis.document = env.document;
        env.window.dispatchEvent({ type: 'pagehide', persisted: false });
        globalThis.window = previousWindow;
        globalThis.document = previousDocument;
    });
    // The sheets: two panes show the list beside the page; one column shows the list alone, or a page alone.
    const wide = { on: true };
    const shell = content.querySelector('.settings-shell');
    controls.page.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: wide.on ? 1000 : 300 });
    content.querySelector('[data-settings-home]').getClientRects = () => (wide.on || shell.dataset.settingsView === 'home' ? [{ width: 1, height: 1 }] : []);
    const reads = () => env.calls.filter((call) => LIST.includes(call)).length;
    const resized = async () => { observers.forEach((observer) => observer.fn([{ target: shell }])); await settle(); };
    await controls.open('test-list-a');
    await settle();
    assert.equal(reads(), LIST.length, 'Settings shown: each of the list\'s sources read once');
    pageCtx.openSettings('test-list-b');
    await settle();
    assert.equal(reads(), LIST.length, 'a page pressed beside the list reads none of them');
    // The socket's first open changes nothing (it waits for the app's first state read); a reconnect may follow a restart.
    env.ws.emit('open', { previouslyConnected: false });
    await settle();
    assert.equal(reads(), LIST.length, 'the socket\'s first open reads nothing again');
    env.ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(reads(), 2 * LIST.length, 'a reconnect reads again');
    // Leaving Settings takes the list off screen; shown again, it reads again.
    state.activePage = 'chat';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    await controls.open('test-list-b');
    await settle();
    assert.equal(reads(), 3 * LIST.length, 'Settings shown again: read again');
    // One column: the page hides the list, which reads when the column returns to it.
    wide.on = false;
    await resized();
    assert.equal(reads(), 3 * LIST.length, 'a width that takes the list away reads nothing');
    controls.pressedAgain();
    await settle();
    assert.equal(reads(), 4 * LIST.length, 'one column back on the list: read');
    // A width that brings the list beside an open page reads it as it comes.
    await controls.open('test-list-a');
    await settle();
    assert.equal(reads(), 4 * LIST.length, 'one column, a page over the list: nothing');
    wide.on = true;
    await resized();
    assert.equal(reads(), 5 * LIST.length, 'the list beside the page again: read');
});

test('the Settings shell drops held list reads off screen, keeps a press beside it, and renews only its live sources', async (t) => {
    const { env, content, state, showPage } = world(t);
    const held = new Map();
    const hold = (url) => {
        const request = { signal: null, ...deferred() };
        const previous = held.get(url) || [];
        held.set(url, [...previous, request]);
        return request;
    };
    for (const url of ['/api/extensions', '/api/update/status', '/api/desktop/autostart']) {
        env.routes.set(url, async (_url, init = {}) => {
            const request = hold(url);
            request.signal = init.signal;
            return request.promise;
        });
    }
    const controls = initSettings({ state, ws: env.ws, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const proto = Object.getPrototypeOf(Stub.prototype);
    const beforeRects = Stub.prototype.getClientRects;
    Stub.prototype.getClientRects = () => [];
    t.after(() => { Stub.prototype.getClientRects = beforeRects; });
    const beforeRect = proto.getBoundingClientRect;
    proto.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: 300 });
    t.after(() => { proto.getBoundingClientRect = beforeRect; });
    controls.page.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: 1000 });
    const homeRoot = content.querySelector('[data-settings-home]');
    homeRoot.getClientRects = () => [{ width: 1, height: 1 }];
    homeRoot.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: 300 });
    await controls.open('models');
    await settle();
    for (const url of ['/api/extensions', '/api/update/status', '/api/desktop/autostart']) {
        assert.equal(held.get(url)?.length, 1, `${url} was held by the shell's list read`);
        assert.equal(held.get(url)[0].signal?.aborted, false, `${url} is live while the list is shown`);
    }

    // A Settings page beside the list is a press inside the same screen: it must leave the held sources alone.
    await controls.open('agents');
    await settle();
    for (const [url, requests] of held) assert.equal(requests[0].signal.aborted, false, `${url} stays live beside a page`);

    // The shell's own page path hides the list, and therefore aborts every source it started for that screen.
    state.activePage = 'chat';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    await settle();
    for (const [url, requests] of held) assert.equal(requests[0].signal.aborted, true, `${url} is aborted when Settings is left`);
    for (const requests of held.values()) requests[0].resolve(reply({}));
    await settle();

    // Back on the list, each live source has one subscription and each subscription is quiet after the list leaves.
    const counts = new Map();
    const live = (url, body) => {
        env.routes.set(url, async () => { counts.set(url, (counts.get(url) || 0) + 1); return reply(body); });
    };
    live('/api/extensions', ANSWERS['/api/extensions']);
    live('/api/places', ANSWERS['/api/places']);
    live('/api/update/status', ANSWERS['/api/update/status']);
    env.routes.set('/api/ui/contributions', async () => reply({}));
    state.activePage = 'settings';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
    await settle();
    const detach = getContributions().attach({ ws: env.ws, target: env.window });
    t.after(detach);
    await settle();
    const count = (url) => counts.get(url) || 0;
    const renews = [['/api/extensions', () => env.ws.emit('extension_lifecycle')],
        ['/api/places', () => env.window.dispatchEvent(new CustomEvent('ouro:places-changed'))],
        ['/api/update/status', () => env.ws.emit('update_status_ready')]];
    for (const [url, signal] of renews) {
        const before = count(url);
        signal();
        await settle();
        assert.ok(count(url) > before, `${url} renews while the list is on screen`);
    }
    state.activePage = 'chat';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    await settle();
    await settle(); // finish the source refreshes requested just before the list left the screen
    const quiet = new Map([...counts].map(([url, value]) => [url, value]));
    env.ws.emit('extension_lifecycle');
    env.window.dispatchEvent(new CustomEvent('ouro:places-changed'));
    env.ws.emit('update_status_ready');
    await settle();
    assert.deepEqual([...counts], [...quiet], 'live-source subscriptions are quiet when the list is hidden');
    env.window.dispatchEvent({ type: 'pagehide', persisted: false });
});

test('one read of a source at a time: asked again in flight it starts no second request and runs once more after; off screen it is dropped', async (t) => {
    const { env, content } = world(t);
    const signals = [];
    const gates = [deferred(), deferred(), deferred()];
    env.routes.set('/api/places', async (_url, init = {}) => { signals.push(init.signal); return gates[signals.length - 1].promise; });
    const { home, answer } = listHome(t, content);
    const first = home.refresh(['places']);
    void home.refresh(['places']);
    void home.refresh(['places']);
    await settle();
    assert.equal(signals.length, 1, 'asked twice more while it runs: no second request');
    gates[0].resolve(reply({ places: [{}] }));
    await settle();
    assert.deepEqual([signals.length, answer('page:circles')], [2, '1 place'], 'its answer is painted, then one more read for the asks it heard');
    gates[1].resolve(reply({ places: [{}, {}] }));
    await first;
    assert.deepEqual([signals.length, answer('page:circles')], [2, '2 places']);
    // Off screen: the read in flight is aborted, freeing its connection, and its late answer paints nothing.
    void home.refresh(['places']);
    await settle();
    home.hide();
    assert.equal(signals[2].aborted, true);
    gates[2].resolve(reply({ places: [] }));
    await settle();
    assert.equal(answer('page:circles'), '2 places');
});

test('the passive roster read ends at the status reads\' lifetime: one that never answers frees the read asked after it; off screen it is dropped', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });  // before the world, so the timers it keeps are these
    const { env, content } = world(t);
    const roster = { view: 'quota', quota: [], reads: { catalog: 'not_read', accounts: 'ok', quota: 'ok' },
        profiles: { profiles: ['p0', 'p1'].map((id) => ({ profile: { harness_id: 'codex', profile_id: id, display_name: id,
            enabled: true }, status: { verification: 'passed' }, identity: {} })) } };
    const signals = [];
    // The first request and the last never answer, abort or not: an engine that stalls, not a host that is offline.
    env.routes.set('/api/claudexor/status?view=quota', async (_url, init = {}) => {
        signals.push(init.signal);
        return signals.length === 2 ? reply(roster) : new Promise(() => {});
    });
    const { home, answer } = listHome(t, content);
    const first = home.refresh(['roster']);
    void home.refresh(['roster']);                    // asked again while it runs: it waits for the first
    await settle();
    t.mock.timers.tick(STATUS_READ_LIFETIME_MS - 1);
    await settle();
    assert.equal(signals.length, 1, 'within its lifetime the read is out and the next one waits');
    t.mock.timers.tick(1);
    await first;
    assert.deepEqual([signals.length, signals[0].aborted, answer('accounts')], [2, true, '2 accounts'],
        'at its lifetime its transport ends and the read asked meanwhile goes out');
    // Off screen the read in flight is dropped as before, freeing its connection, and paints nothing.
    void home.refresh(['roster']);
    await settle();
    home.hide();
    assert.equal(signals[2].aborted, true);
    await settle();
    assert.equal(answer('accounts'), '2 accounts');
});

test('the MCP servers status writer updates Extensions\' door beside another page without reading; off screen it paints and reads nothing', async (t) => {
    const { env, content } = world(t);
    const status = (enabled) => ({ sdk_available: true, enabled: true, servers: [{ id: 'docs', enabled, tool_count: 3,
        ...(enabled ? { sign_in: { state: 'sign_in_required' } } : {}) }] });
    env.routes.set('/api/mcp/status', () => reply(status(false)));
    const { home, answer } = listHome(t, content);
    await home.refresh();
    home.mark('rules');
    // Extensions' one door says both kinds it holds: the listing's one skill, then the servers.
    assert.equal(answer('page:skills'), '1 skill · 0 of 1 MCP server on');
    const pending = deferred();
    env.routes.set('/api/mcp/status', () => pending.promise);
    const tools = await settingsPageFor('tools').mount(new Stub(), { path: 'tools', ctx: { readSettings } });
    t.after(() => tools.unmount());
    await settle();
    const before = [...env.calls];
    pending.resolve(reply(status(true)));  // the page's completed status is stored by refreshStatus
    await settle();
    assert.equal(answer('page:skills'), '1 skill · 1 of 1 MCP server on');
    assert.ok(content.querySelector('[data-needs="tool-sign-in"]'));
    assert.deepEqual(env.calls, before, 'adopting the completed status starts no request');
    const late = deferred();
    env.routes.set('/api/mcp/status', () => late.promise);
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools' } }));
    await settle();
    home.hide();
    const hiddenReads = [...env.calls];
    const hiddenText = content.textContent;
    late.resolve({ ok: false, status: 503, json: async () => ({}) });
    await settle();
    assert.equal(content.textContent, hiddenText, 'a completed status paints nothing off screen');
    assert.deepEqual(env.calls, hiddenReads, 'a completed status reads nothing off screen');
    const recovery = deferred();
    env.routes.set('/api/mcp/status', () => recovery.promise);
    const reopened = home.refresh(['tools']);
    assert.equal(answer('page:skills'), '1 skill · MCP status unavailable', 'use the shared completed failure before the new read settles');
    assert.equal(content.querySelector('[data-needs="tool-sign-in"]'), null, 'no flash of the old sign-in need');
    recovery.resolve(reply(status(false)));
    await reopened;
    assert.equal(answer('page:skills'), '1 skill · 0 of 1 MCP server on', 'the status updates when the new read answers');
});

test('an answer is painted though another source was read meanwhile: a slow source, and the state snapshot by its sequencer', async (t) => {
    const { env, content } = world(t);
    const places = deferred();
    const state = deferred();
    env.routes.set('/api/places', async () => places.promise);
    env.routes.set('/api/state', async () => state.promise);
    const snapshots = createStateSnapshotSequencer(() => {});
    snapshots.apply(snapshots.begin(), { evolution_enabled: false });
    const { home, answer } = listHome(t, content, { stateSnapshots: snapshots });
    const held = home.refresh(['places', 'state']);
    await home.refresh(['settings']);
    assert.deepEqual([answer('page:circles'), answer('dashboard:evolution'), answer('models')], ['', 'Off', 'glm-5.3']);
    places.resolve(reply({ places: [{}, {}] }));
    state.resolve(reply({ evolution_enabled: true }));
    await held;
    await settle();
    assert.deepEqual([answer('page:circles'), answer('dashboard:evolution')], ['2 places', 'On']);
});

test('the row of a page that is left reads its sources again, a setting\'s row and one that is not; a save reads its page\'s row', async (t) => {
    const { env, content } = world(t);
    const now = { model: 'openrouter::z-ai/glm-5.3', places: [], servers: [] };
    env.routes.set('/api/settings', async () => reply({ ...ANSWERS['/api/settings'], OUROBOROS_MODEL: now.model }));
    env.routes.set('/api/places', async () => reply({ places: now.places }));
    env.routes.set('/api/mcp/status', async () => reply({ enabled: true, servers: now.servers }));
    const { home, answer } = listHome(t, content);
    await home.refresh();
    const go = async (type, detail) => { env.window.dispatchEvent(new CustomEvent(type, { detail })); await settle(); };
    const count = (url) => env.calls.filter((call) => call === url).length;
    await go('ouro:settings-page-shown', { path: 'models' });
    const before = Object.fromEntries([...LIST, '/api/settings'].map((url) => [url, count(url)]));
    const since = () => Object.entries(before).filter(([url, n]) => count(url) > n).map(([url, n]) => `${url} ${count(url) - n}`);
    now.model = 'openrouter::openai/gpt-5';
    await go('ouro:settings-page-shown', { path: 'models/role/main' });
    assert.deepEqual([answer('models'), since()], ['glm-5.3', []], 'a page of the same row: nothing is left, nothing read');
    await go('ouro:settings-page-shown', { path: 'agents' });
    assert.deepEqual([answer('models'), since()], ['gpt-5', ['/api/settings 1']], 'Models left: its one source, nothing else of the list');
    const beforeAgents = count('/api/settings');
    await go('ouro:settings-page-shown', { path: 'rules' });
    assert.equal(count('/api/settings'), beforeAgents + 1, 'Agents left: its settings source is read once');
    // A door leaves Settings before its destination is shown, so the home does not own that page's read.
    // A server switched on Extensions' MCP servers page is one entry of MCP_SERVERS written; its event names the row,
    // not the page. That page is no page of this home (Extensions' door stands for it), so the document alone is read
    // again; the servers' status is the door's answer when the home reads it on coming on screen.
    env.routes.set('/api/settings', async (_url, init = {}) => (init.method === 'POST'
        ? reply({ status: 'saved', changes: { MCP_SERVERS: { withheld: 'structure', token: 'servers@2' } } })
        : reply({ ...ANSWERS['/api/settings'], MCP_SERVERS: [{ id: 'docs', enabled: false }], _meta: { tokens: { MCP_SERVERS: 'servers@1' } } })));
    now.servers = [{ id: 'docs', enabled: true, tool_count: 2 }];
    const statusReads = count('/api/mcp/status');
    assert.equal((await writeEntry('MCP_SERVERS', 'docs', (before) => ({ ...before, enabled: true }))).state, 'saved');
    await settle();
    assert.equal(count('/api/mcp/status'), statusReads, 'a row of Extensions is not this home\'s read');
    await home.refresh(['tools']);
    assert.equal(answer('page:skills'), '1 skill · 1 of 1 MCP server on');
    // A width change covers the list and brings it back beside the same page, without announcing a new path.
    home.hide();
    await home.refresh();
    const afterWidth = count('/api/settings');
    await go('ouro:settings-page-shown', { path: 'models' });
    assert.equal(count('/api/settings'), afterWidth + 1, 'the first press after resizing still leaves Rules');
    await go('ouro:settings-page-shown', { path: 'models/role/main' });
    assert.equal(count('/api/settings'), afterWidth + 1, 'a level of the same page still reads nothing');
});
