// The owner's consciousness, places and safety surfaces in Settings, read against the contract's shapes
// (fixtures/consciousness_contract.json): the words, the one client module, the Safety Supervisor section and
// its owner-only saves, the Agents second-check mark and the Consciousness section.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    AUTONOMY_LEVELS, PLACE_STATES, SAFETY_LEVELS, SILENT_CARD, durationText, placeStatusText, usdText, wakesForText,
} from '../modules/consciousness_words.js';
import {
    SAFETY_MODEL_KEYS, connectPlace, markRecheckRow, normalizeConsciousness, normalizePlacesResponse, parseRoster,
    recheckFromSettings, saveCheckerSettings, saveSafetyLevel, saveSafetyRules, updatePlace, disconnectPlace,
} from '../modules/consciousness_api.js';
import {
    agentRowHandle, applySafetySettings, checkerSaveEffect, checkerTabsText, checkerText, safetyOwnerDraft, safetySectionHtml,
    saveSafetyChecker, saveSafetyOwnerChanges, secondCheckText, takeCheckerChanges,
} from '../modules/settings_safety.js';
import { chooseRecheckRow, recheckDraft, recheckMarkHtml, syncRecheckState } from '../modules/safety_recheck.js';
import {
    bindConsciousnessSettings, consciousnessModelText, consciousnessSectionHtml, migrationProblemLines, settingsPlacesHtml,
} from '../modules/settings_consciousness.js';
import { parseAvailableSubagentsSetting, buildAvailableSubagentsSetting } from '../modules/subagents_settings.js';
import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { apiClient } from '../modules/api_client.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));

/** Record every request the client sends and answer from `reply(url, body)`. */
function stubFetch(reply) {
    const calls = [];
    const previous = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        const body = init.body ? JSON.parse(init.body) : null;
        calls.push({ url: String(url), method: init.method || 'GET', body });
        const answer = reply(String(url), body, init);
        return { ok: answer.status ? answer.status < 400 : true, status: answer.status || 200, json: async () => answer.json ?? answer };
    };
    return { calls, restore: () => { globalThis.fetch = previous; } };
}

test('the owner names stand with their notes; stored values never change', () => {
    assert.deepEqual(AUTONOMY_LEVELS.map((level) => [level.value, level.label]), [['observe', 'Advises'], ['act', 'Acts'], ['full', 'Evolves']]);
    assert.equal(AUTONOMY_LEVELS[0].note, 'On its own it studies, keeps its memory and tells you; it changes nothing outside.');
    assert.equal(AUTONOMY_LEVELS[1].note, 'Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself.');
    assert.equal(AUTONOMY_LEVELS[2].note, 'Also changes itself: its own code, its settings, connecting new places.');
    assert.deepEqual(PLACE_STATES.map((state) => [state.value, state.label]), [['active', 'Active'], ['silent', 'Silent'], ['paused', 'Paused']]);
    assert.equal(SILENT_CARD, "Writes through this connector's declared tools and delivery queue are blocked. Scripts, browser actions and unintegrated code may still write.");
    assert.deepEqual(SAFETY_LEVELS.map((level) => level.label), ['Off', 'Light', 'Full']);
    assert.equal(SAFETY_LEVELS[1].note, 'A second model checks steps that reach other people and services: messages, posts, skills, connectors, starting work. Not shell commands.');
});

test('a place row says its state, what wakes it and what waits; absent money is the dash', () => {
    const { places } = normalizePlacesResponse(FIXTURE.places);
    assert.equal(placeStatusText(places[0]), 'active · wakes for direct messages, mentions and replies · 2 open turns');
    assert.equal(placeStatusText(places[1]), 'silent · wakes for mentions', 'a Silent place still wakes for its list');
    assert.equal(placeStatusText(places[2]), 'paused', 'a Paused place says only its state');
    assert.equal(placeStatusText({ state: 'gone' }), 'state unknown');
    assert.equal(wakesForText([]), '');
    assert.equal(usdText(null), '—');
    assert.equal(usdText(0), '$0.00');
    assert.equal(usdText(0.88), '$0.88');
    assert.equal(durationText('2026-10-20T11:51:00Z', '2026-10-20T11:51:40Z'), '40 s');
    assert.equal(durationText('', '2026-10-20T11:51:40Z'), '');
});

test('the client reads the contract shapes once: places, connectors, the snapshot and its wakes', () => {
    const { places, connectors } = normalizePlacesResponse(FIXTURE.places);
    assert.equal(places.length, 3);
    assert.deepEqual(places[0].wakes.map((item) => item.author), ['connector', 'connector', 'owner']);
    assert.equal(places[0].notes[0].id, 'n1');
    assert.equal(places[0].ownerAccounts[0].label, '@anton.r');
    assert.equal(places[1].icon.name, 'chat');
    assert.deepEqual(places[2].icon, {});
    assert.equal(places[0].revision, 4);
    assert.equal(places[0].icon.kind, 'svg');
    assert.equal(connectors[0].label, 'Slack');
    assert.equal(places[1].spent24hUsd, null, 'an unknown spend stays unknown');
    assert.equal(places[2].spent24hUsd, 0, 'a real zero is a fact');
    assert.deepEqual(connectors.map((entry) => [entry.connector, entry.connected]), [['slack-bridge', true], ['agentrq-bridge', false]]);

    const state = normalizeConsciousness(FIXTURE.state.bg_consciousness_state, { enabled: FIXTURE.state.bg_consciousness_enabled });
    assert.equal(state.enabled, true);
    assert.equal(state.spendByCause[2].unmeteredCalls, 3);
    assert.equal(state.countedRoots[0].causeLabel, 'Slack · Acme');
    assert.equal(state.placesBrief.length, 2);
    assert.equal(state.recentWakes, null, '/api/state carries no wakes');

    const full = normalizeConsciousness(FIXTURE.consciousness);
    const old = full.recentWakes.find((wake) => wake.taskId === '214a31b6');
    assert.equal(old.live, false);
    assert.equal(old.phase, 'done');
    assert.equal(old.cost.cost_presentation.tracked_amount, null);
    assert.deepEqual(old.checks, [{ chip: 'checked', tool: 'run_command' }]);
    assert.equal(full.recentWakes.find((wake) => wake.taskId === 'w-running').live, true);
    assert.equal(normalizeConsciousness({}).enabled, null, 'an absent control is unknown, never off');
});

test('every write sends exactly the contract keys and announces the confirmed place', async () => {
    const place = FIXTURE.places.places[0];
    const stub = stubFetch((url) => (url.startsWith('/api/places') ? { ok: true, place } : { ok: true }));
    try {
        assert.equal((await connectPlace({ connector: 'agentrq-bridge' })).id, 'slack-acme');
        await updatePlace('slack acme', { state: 'silent', expected_revision: 4 });
        await disconnectPlace('slack-acme', 4);
        await saveSafetyLevel('light');
        await saveSafetyLevel('full', 'consciousness');
        await saveSafetyRules('No posts from my folders.');
        await saveCheckerSettings({ OUROBOROS_MODEL_SAFETY: '' });
        await saveCheckerSettings({ OUROBOROS_SUBAGENTS: markRecheckRow(parseRoster(FIXTURE.settings.OUROBOROS_SUBAGENTS), '') });
    } finally { stub.restore(); }
    assert.deepEqual(stub.calls.map(({ url, method, body }) => [method, url, body]), [
        ['POST', '/api/places', { connector: 'agentrq-bridge' }],
        ['POST', '/api/places/slack%20acme', { state: 'silent', expected_revision: 4 }],
        ['DELETE', '/api/places/slack-acme', { expected_revision: 4 }],
        ['POST', '/api/owner/safety-mode', { mode: 'light' }],
        ['POST', '/api/owner/safety-mode', { mode: 'full', kind: 'consciousness' }],
        ['POST', '/api/owner/safety-rules', { rules: 'No posts from my folders.' }],
        ['POST', '/api/owner/safety-model', { settings: { OUROBOROS_MODEL_SAFETY: '' } }],
        ['POST', '/api/owner/safety-model', { settings: { OUROBOROS_SUBAGENTS: { ...FIXTURE.settings.OUROBOROS_SUBAGENTS,
            items: FIXTURE.settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, rechecks_flags: false })) } } }],
    ]);
});

test('a refused write says the server\'s words and saves nothing in the browser', async () => {
    const stub = stubFetch(() => ({ status: 409, json: { ok: false, error: 'Place changed since you opened it.' } }));
    try {
        await assert.rejects(updatePlace('slack-acme', { state: 'paused' }), /Place changed since you opened it/);
    } finally { stub.restore(); }
});

test('Safety Supervisor: two rows of Off · Light · Full, the short copy, the rules and the honesty lines', () => {
    const html = safetySectionHtml();
    assert.match(html, /A second model judges a step before it runs\. Your work: what you asked for\. Consciousness: what it does on its own, places included\./);
    assert.match(html, /<label>Your work<\/label>\s*<input id="s-safety-mode" type="hidden"/);
    assert.match(html, /<label>Consciousness<\/label>\s*<input id="s-consciousness-safety-mode" type="hidden"/);
    const order = [...html.matchAll(/data-effort-value="(\w+)"/g)].map((match) => match[1]);
    assert.deepEqual(order, ['off', 'light', 'full', 'off', 'light', 'full']);
    assert.match(html, /data-policy-state="supervisor_consciousness"/);
    assert.match(html, />Check consciousness fully</);
    assert.match(html, /Not measured yet: no replay of real checks exists\./);
    assert.match(html, /Ouroboros and the Safety Supervisor read these in every task and check\. Rules for one place live in that place's notes\./);
    assert.match(html, /outside Cyber Pro, Ouroboros cannot lower its own checks/);
    assert.match(html, /id="s-safety-skip-counter"/);
});

test('the checker and second-check lines name what really checks', () => {
    const settings = FIXTURE.settings;
    assert.equal(checkerText(settings), 'Checked by: the Light model');
    assert.equal(checkerText({ ...settings, OUROBOROS_MODEL_SAFETY: 'anthropic::claude-opus-5.5' }), 'Checked by: claude-opus-5.5');
    const recheck = recheckFromSettings(settings);
    assert.equal(agentRowHandle('packet_checker', settings), 'gpt-5.6');
    assert.equal(secondCheckText(recheck, settings), 'Second check: gpt-5.6 re-checks flagged actions');
    const unmarked = { enabled: true, items: settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, rechecks_flags: false })) };
    assert.equal(secondCheckText(recheckFromSettings({ OUROBOROS_SUBAGENTS: unmarked }), settings), 'Second check: Off');
    assert.match(secondCheckText(recheckFromSettings({ OUROBOROS_SUBAGENTS: { enabled: true, items: [] } }), settings), /^Second check: Off\. No Agents row can re-check yet/);
    assert.equal(secondCheckText(recheckFromSettings({}), settings), 'Second check: Off', 'an unreadable roster says nothing more');
    const disabled = { enabled: true, items: settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, enabled: false })) };
    assert.equal(secondCheckText(recheckFromSettings({ OUROBOROS_SUBAGENTS: disabled }), settings), 'Second check: Off (the marked row is disabled)');
    const delegationOff = { ...settings.OUROBOROS_SUBAGENTS, enabled: false };
    assert.equal(secondCheckText(recheckFromSettings({ OUROBOROS_SUBAGENTS: delegationOff }), settings), 'Second check: Off (Delegation is off in Agents)');
});

/** A Safety Supervisor section with only the nodes `applySafetySettings` paints. */
function sectionRoot() {
    const nodes = new Map();
    const painted = ['#s-safety-mode', '#s-consciousness-safety-mode', '#s-safety-rules', '[data-safety-checker]', '[data-safety-second-check]',
        '[data-safety-unmeasured]', '[data-safety-check-fully-line]', '[data-policy-state="supervisor_consciousness"]'];
    const node = (selector) => {
        if (!nodes.has(selector)) nodes.set(selector, { value: '', hidden: true, disabled: false, innerHTML: '', textContent: '' });
        return nodes.get(selector);
    };
    return { querySelector: (selector) => (painted.includes(selector) ? node(selector) : null), querySelectorAll: () => [] };
}

test('«Not measured yet» is said wherever the second check is offered, the default Off included', () => {
    const unmarked = { enabled: true, items: FIXTURE.settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, rechecks_flags: false })) };
    for (const roster of [unmarked, FIXTURE.settings.OUROBOROS_SUBAGENTS]) {
        const root = sectionRoot();
        applySafetySettings(root, { ...FIXTURE.settings, OUROBOROS_SUBAGENTS: roster });
        assert.equal(root.querySelector('[data-safety-unmeasured]').hidden, false);
    }
    const root = sectionRoot();
    applySafetySettings(root, { ...FIXTURE.settings, OUROBOROS_SUBAGENTS: 'not a roster' });
    assert.equal(root.querySelector('[data-safety-unmeasured]').hidden, true, 'no readable roster, no second check offered');
});

function draftRoot(values) {
    const node = (value) => ({ value });
    return { querySelector: (selector) => ({ '#s-safety-mode': node(values.yourWork), '#s-consciousness-safety-mode': node(values.consciousness),
        '#s-safety-rules': node(values.rules) }[selector] || null) };
}

test('Save writes only what changed: the checker first, the levels and rules after; lowering asks first', async () => {
    const settings = { ...FIXTURE.settings, OUROBOROS_SAFETY_MODE: 'full', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'full' };
    syncRecheckState(recheckFromSettings(settings), null);
    const stub = stubFetch(() => ({ ok: true }));
    const asked = [];
    let result;
    try {
        const body = { OUROBOROS_MODEL_SAFETY: 'anthropic::claude-opus-5.5' };
        const draft = safetyOwnerDraft(draftRoot({ yourWork: 'light', consciousness: 'full', rules: settings.OUROBOROS_SAFETY_OWNER_RULES }), body, settings);
        assert.deepEqual(body, {}, 'the changed checker key left the generic body');
        assert.equal(await saveSafetyChecker(draft), true);
        result = await saveSafetyOwnerChanges(draft, settings, { askOwner: async (dialog) => { asked.push(dialog.title); return true; } });
    } finally { stub.restore(); }
    assert.deepEqual(asked, ['Lower a safety level']);
    assert.deepEqual(stub.calls.map(({ url, body }) => [url, body]), [
        ['/api/owner/safety-model', { settings: { OUROBOROS_MODEL_SAFETY: 'anthropic::claude-opus-5.5' } }],
        ['/api/owner/safety-mode', { mode: 'light' }],
    ]);
    assert.equal(result.error, '');
    assert.equal(result.saved, 'Safety Supervisor model saved as claude-opus-5.5. Safety for your work saved as Light.');
});

test('a refused lowering keeps the level and still saves the rest; an unknown server key is never written', async () => {
    const settings = { OUROBOROS_SAFETY_MODE: 'full', OUROBOROS_SAFETY_OWNER_RULES: '' };
    const stub = stubFetch(() => ({ ok: true }));
    let result;
    try {
        const draft = { yourWork: 'off', consciousness: 'off', rules: 'Ask before paying.', checker: null };
        result = await saveSafetyOwnerChanges(draft, settings, { askOwner: async () => false });
    } finally { stub.restore(); }
    // No OUROBOROS_CONSCIOUSNESS_SAFETY_MODE in the document: that row is not written.
    assert.deepEqual(stub.calls.map(({ url }) => url), ['/api/owner/safety-rules']);
    assert.match(result.error, /Safety for your work was not changed: The change was not confirmed\./);
    assert.equal(result.unknown, false);
});

test('the second-check mark shows only on eligible rows, one row at a time, and saves as the owner chose', async () => {
    const packet = (id, marked = false) => ({ subagent_id: id, route: { kind: 'api_model', target_id: 'openai::gpt-5.6' }, delivery: 'packet', rechecks_flags: marked });
    const roster = { enabled: true, items: [packet('a', true), packet('b')] };
    syncRecheckState(recheckFromSettings({ OUROBOROS_SUBAGENTS: roster }), null);
    assert.match(recheckMarkHtml({ subagent_id: 'a', _uiKey: 'k1' }, 1), /checked> Re-checks flagged actions/);
    assert.match(recheckMarkHtml({ subagent_id: 'b' }, 2), /data-recheck-row="b"><input/);
    assert.doesNotMatch(recheckMarkHtml({ subagent_id: 'b' }, 2), /checked>/);
    assert.match(recheckMarkHtml({ subagent_id: 'c' }, 3), /data-recheck-row="c" hidden>/);
    assert.match(recheckMarkHtml({ subagent_id: 'a', _uiKey: 'k1' }, 1), /id="s-recheck-k1"/, 'its id joins the Settings draft');
    assert.equal(recheckDraft(), null, 'untouched');
    chooseRecheckRow('b', true, null);
    assert.equal(recheckDraft(), 'b');
    assert.doesNotMatch(recheckMarkHtml({ subagent_id: 'a' }, 1), /checked>/, 'one row at a time');
    chooseRecheckRow('b', false, null);
    assert.equal(recheckDraft(), '', 'unticked: no row re-checks');
    const stub = stubFetch(() => ({ ok: true }));
    try {
        // The Agents editor did not load: the mark goes into the saved roster.
        const draft = safetyOwnerDraft(draftRoot({ yourWork: 'full', consciousness: 'full', rules: null }), {}, { OUROBOROS_SUBAGENTS: roster });
        await saveSafetyChecker(draft);
        const result = await saveSafetyOwnerChanges(draft, { OUROBOROS_SUBAGENTS: roster });
        assert.equal(result.saved, 'Second check turned off.');
    } finally { stub.restore(); }
    assert.deepEqual(stub.calls.map(({ url, body }) => [url, body]), [['/api/owner/safety-model', { settings: { OUROBOROS_SUBAGENTS: {
        enabled: true, items: [packet('a'), packet('b')] } } }]]);
    // A marked row that cannot run now keeps its mark in view, so the owner can untick it.
    syncRecheckState(recheckFromSettings({ OUROBOROS_SUBAGENTS: { enabled: true, items: [{ ...packet('a', true), enabled: false }, packet('b')] } }), null);
    assert.match(recheckMarkHtml({ subagent_id: 'a' }, 1), /data-recheck-row="a"><input[^>]*checked> Re-checks flagged actions/);
    assert.match(recheckMarkHtml({ subagent_id: 'b' }, 2), /data-recheck-row="b"><input/);
});

// ---- Save: the checker's keys travel through the owner transaction, before the generic save -----------------

const ROSTER = () => buildAvailableSubagentsSetting(parseAvailableSubagentsSetting(FIXTURE.settings.OUROBOROS_SUBAGENTS).setting);
// The settings document as GET /api/settings answers it on a default install (empty Safety Supervisor slot)...
const LOADED = {
    ...FIXTURE.settings,
    OUROBOROS_MODEL: 'anthropic/claude-opus-5.5', USE_LOCAL_LIGHT: 'false', USE_LOCAL_MAIN: false,
    OUROBOROS_MODEL_ACCOUNTS: '{"main": "acct-1"}', OUROBOROS_MODEL_CONTEXT_WINDOWS: { main: 200000 },
    OUROBOROS_SUBAGENTS: JSON.stringify(FIXTURE.settings.OUROBOROS_SUBAGENTS), OUROBOROS_TOOL_TIMEOUT_SEC: 120,
};
// ...and the body Save collects from it, untouched: the same values in the editors' own spellings.
const collected = () => ({
    OUROBOROS_MODEL_SAFETY: '', OUROBOROS_MODEL_LIGHT: 'anthropic/claude-sonnet-5.5', OUROBOROS_MODEL: 'anthropic/claude-opus-5.5',
    USE_LOCAL_LIGHT: false, USE_LOCAL_MAIN: false, OUROBOROS_MODEL_ACCOUNTS: { main: 'acct-1' },
    OUROBOROS_MODEL_CONTEXT_WINDOWS: { main: 200000 }, OUROBOROS_SUBAGENTS: ROSTER(), OUROBOROS_TOOL_TIMEOUT_SEC: 120,
});
const ownerRoot = () => draftRoot({ yourWork: LOADED.OUROBOROS_SAFETY_MODE, consciousness: LOADED.OUROBOROS_CONSCIOUSNESS_SAFETY_MODE,
    rules: LOADED.OUROBOROS_SAFETY_OWNER_RULES });

/** The checker a settings document resolves, as `safety_config.safety_configuration` reads it: the Safety
 *  Supervisor slot, else Light, else Main, with that slot's Local flag; the maps' `safety` entries; the marked row. */
function resolvedChecker(doc) {
    const json = (value) => (typeof value === 'string' ? JSON.parse(value || '{}') : value || {});
    const slot = String(doc.OUROBOROS_MODEL_SAFETY || '').trim() ? 'SAFETY' : String(doc.OUROBOROS_MODEL_LIGHT || '').trim() ? 'LIGHT' : 'MAIN';
    const model = String(doc[{ SAFETY: 'OUROBOROS_MODEL_SAFETY', LIGHT: 'OUROBOROS_MODEL_LIGHT', MAIN: 'OUROBOROS_MODEL' }[slot]] || '').trim();
    const local = model.endsWith(' (local)') || /^(true|1|yes|on)$/i.test(String(doc[`USE_LOCAL_${slot}`] ?? ''));
    const options = ['OUROBOROS_MODEL_ACCOUNTS', 'OUROBOROS_MODEL_CONTEXT_WINDOWS', 'OUROBOROS_MODEL_PROCESSING_PREFERENCES'].map((key) => json(doc[key]).safety ?? '');
    const roster = json(doc.OUROBOROS_SUBAGENTS);
    const marked = (roster.items || []).filter((item) => item.rechecks_flags).map((item) => [item.subagent_id, item.route, item.enabled !== false, roster.enabled]);
    return JSON.stringify([model.replace(/ \(local\)$/, ''), local, options, marked]);
}

/** A gateway outside Cyber Pro: the generic save refuses a change to the resolved checker; the owner transaction
 *  writes its keys. */
function settingsServer(stored) {
    const state = { ...stored };
    const stub = stubFetch((url, body) => {
        if (url === '/api/owner/safety-model') {
            // The endpoint's own judgement: a roster with no Reviewer row saves only with the owner's confirmation.
            const roster = body.settings.OUROBOROS_SUBAGENTS;
            if (roster && !roster.items.some((item) => item.review_eligible) && body.allow_empty_review_pool !== true) {
                return { status: 400, json: { saved: false, code: 'empty_review_pool', error: 'No row is marked Reviewer.' } };
            }
            Object.assign(state, body.settings);
            return { ok: true };
        }
        if (url === '/api/settings' && resolvedChecker({ ...state, ...body }) !== resolvedChecker(state)) {
            return { status: 400, json: { saved: false, error: 'The resolved Safety Supervisor is owner-controlled; use /api/owner/safety-model.' } };
        }
        if (url === '/api/settings') { Object.assign(state, body); return { json: { status: 'saved' } }; }
        return { ok: true };
    });
    return { state, ...stub };
}

test('the checker keys mirror the gateway\'s owner transaction', () => {
    const server = fs.readFileSync(new URL('../../ouroboros/gateway/safety_settings.py', import.meta.url), 'utf8');
    const block = server.match(/SAFETY_MODEL_KEYS = frozenset\(\{([\s\S]*?)\}\)/)?.[1] || '';
    assert.deepEqual([...SAFETY_MODEL_KEYS].sort(), [...block.matchAll(/"([A-Z_]+)"/g)].map((match) => match[1]).sort());
});

test('an unchanged checker key is not sent: it stays in the generic body as collected', () => {
    syncRecheckState(recheckFromSettings(LOADED), null);
    const body = collected();
    assert.equal(takeCheckerChanges(body, LOADED), null);
    assert.deepEqual(body, collected());
    const local = { ...collected(), USE_LOCAL_LIGHT: true };
    assert.deepEqual(takeCheckerChanges(local, LOADED).settings, { USE_LOCAL_LIGHT: true }, "Light's Local box is a checker key too");
    assert.equal('USE_LOCAL_LIGHT' in local, false);
});

test('Models: a Light change outside Cyber Pro saves; only the changed key goes first, through the owner transaction', async () => {
    syncRecheckState(recheckFromSettings(LOADED), null);
    const server = settingsServer(LOADED);
    let result;
    try {
        // What B1 was: the same change in the generic body is refused.
        await assert.rejects(apiClient.saveSettings({ ...collected(), OUROBOROS_MODEL_LIGHT: 'openai::gpt-5.6-mini' }), /owner-controlled/);
        server.calls.length = 0;
        const body = { ...collected(), OUROBOROS_MODEL_LIGHT: 'openai::gpt-5.6-mini', OUROBOROS_TOOL_TIMEOUT_SEC: 300 };
        const draft = safetyOwnerDraft(ownerRoot(), body, LOADED);
        assert.equal(await saveSafetyChecker(draft), true);
        assert.equal((await apiClient.saveSettings(body)).status, 'saved');
        result = await saveSafetyOwnerChanges(draft, LOADED);
        assert.equal(checkerTabsText(draft), 'Models');
        // The checker's model is read live: it applies at the next check, not from the next task.
        assert.deepEqual(checkerSaveEffect({ status: 'saved', no_changes: true }, draft), { status: 'saved', no_changes: false, immediate_changed: true });
        assert.equal(checkerSaveEffect({ status: 'saved' }, { checker: { settings: { OUROBOROS_SUBAGENTS: '{}' } } }).next_task_changed, true);
    } finally { server.restore(); }
    const { OUROBOROS_MODEL_LIGHT: _light, ...rest } = collected();
    assert.deepEqual(server.calls.map(({ url, body }) => [url, body]), [
        ['/api/owner/safety-model', { settings: { OUROBOROS_MODEL_LIGHT: 'openai::gpt-5.6-mini' } }],
        ['/api/settings', { ...rest, OUROBOROS_TOOL_TIMEOUT_SEC: 300 }],
    ]);
    assert.equal(server.state.OUROBOROS_MODEL_LIGHT, 'openai::gpt-5.6-mini');
    assert.equal(server.state.OUROBOROS_TOOL_TIMEOUT_SEC, 300);
    assert.deepEqual(result, { saved: '', error: '', unknown: false });
    assert.equal(checkerSaveEffect({ status: 'saved' }, { checker: null }).next_task_changed, undefined, 'no checker change, the answer as it is');
});

test('the Agents roster still saves, with the second-check mark in the same one write', async () => {
    syncRecheckState(recheckFromSettings(LOADED), null);
    const server = settingsServer(LOADED);
    const reader = { subagent_id: 'reader', recommended_use: 'Reads.', route: { kind: 'api_model', target_id: 'openai::gpt-5.6' }, delivery: 'packet' };
    const edited = { ...ROSTER(), items: [...ROSTER().items, reader] };
    let result;
    try {
        chooseRecheckRow('reader', true, null);
        const body = { ...collected(), OUROBOROS_SUBAGENTS: edited, allow_empty_review_pool: true };
        const draft = safetyOwnerDraft(ownerRoot(), body, LOADED);
        assert.equal('OUROBOROS_SUBAGENTS' in body, false);
        assert.equal('allow_empty_review_pool' in body, false, "the roster's own save flag leaves with it");
        assert.equal(checkerTabsText(draft), 'Agents');
        await saveSafetyChecker(draft);
        await apiClient.saveSettings(body);
        result = await saveSafetyOwnerChanges(draft, LOADED);
    } finally { server.restore(); }
    assert.deepEqual(server.calls.map(({ url }) => url), ['/api/owner/safety-model', '/api/settings']);
    assert.deepEqual(server.calls[0].body, { settings: { OUROBOROS_SUBAGENTS: markRecheckRow(edited, 'reader') },
        allow_empty_review_pool: true }, "the roster's own confirmation travels with it");
    assert.deepEqual(server.state.OUROBOROS_SUBAGENTS.items.map((item) => [item.subagent_id, item.rechecks_flags]),
        [['packet_checker', false], ['reader', true]], 'the new row is kept and carries the mark');
    assert.equal(result.saved, 'Second check saved.');
});

test('«Save without reviewers»: the owner\'s tick reaches the owner transaction, and without it the server refuses', async () => {
    syncRecheckState(recheckFromSettings(LOADED), null);
    const unmarked = { ...ROSTER(), items: ROSTER().items.map((item) => ({ ...item, review_eligible: false })) };
    for (const ticked of [true, false]) {
        const server = settingsServer(LOADED);
        try {
            const body = { ...collected(), OUROBOROS_SUBAGENTS: unmarked, ...(ticked ? { allow_empty_review_pool: true } : {}) };
            const draft = safetyOwnerDraft(ownerRoot(), body, LOADED);
            assert.equal('allow_empty_review_pool' in body, false);
            if (ticked) {
                assert.equal(await saveSafetyChecker(draft), true);
                assert.equal(server.calls[0].body.allow_empty_review_pool, true);
                assert.deepEqual(server.state.OUROBOROS_SUBAGENTS.items.map((item) => item.review_eligible), [false]);
            } else {
                await assert.rejects(saveSafetyChecker(draft), /No row is marked Reviewer/);
                assert.equal('allow_empty_review_pool' in server.calls[0].body, false);
            }
        } finally { server.restore(); }
    }
});

test('a refused checker save shows the server\'s sentence; the generic save is not reached', async () => {
    syncRecheckState(recheckFromSettings(LOADED), null);
    const stub = stubFetch(() => ({ status: 400, json: { saved: false, error: 'OUROBOROS_MODEL_LIGHT must be a model string' } }));
    try {
        const draft = safetyOwnerDraft(ownerRoot(), { ...collected(), OUROBOROS_MODEL_LIGHT: 'x', OUROBOROS_MODEL: 'y' }, LOADED);
        assert.equal(checkerTabsText(draft), 'Models');
        await assert.rejects(saveSafetyChecker(draft), (error) => error.message === 'OUROBOROS_MODEL_LIGHT must be a model string' && error.body.saved === false);
    } finally { stub.restore(); }
    assert.deepEqual(stub.calls.map(({ url, body }) => [url, body]), [['/api/owner/safety-model', { settings: { OUROBOROS_MODEL_LIGHT: 'x', OUROBOROS_MODEL: 'y' } }]]);
});

test('a local Safety Supervisor model is its ` (local)` label; no Local flag of its own is read or written', () => {
    for (const name of ['settings_safety.js', 'consciousness_api.js']) {
        assert.doesNotMatch(fs.readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8'), /USE_LOCAL_SAFETY/);
    }
    assert.equal(checkerText({ OUROBOROS_MODEL_SAFETY: 'qwen3-8b (local)' }), 'Checked by: qwen3-8b (local)');
});

test('the roster keeps the owner\'s second-check mark exactly as loaded and refuses a malformed one', () => {
    const parsed = parseAvailableSubagentsSetting(FIXTURE.settings.OUROBOROS_SUBAGENTS);
    assert.equal(parsed.error, '');
    assert.equal(buildAvailableSubagentsSetting(parsed.setting).items[0].rechecks_flags, true);
    const unmarked = parseAvailableSubagentsSetting({ enabled: true, items: [{ ...FIXTURE.settings.OUROBOROS_SUBAGENTS.items[0], rechecks_flags: undefined }] });
    assert.equal('rechecks_flags' in buildAvailableSubagentsSetting(unmarked.setting).items[0], false, 'an unmarked row writes no new key');
    const bad = parseAvailableSubagentsSetting({ enabled: true, items: [{ ...FIXTURE.settings.OUROBOROS_SUBAGENTS.items[0], rechecks_flags: 'yes' }] });
    assert.match(bad.error, /second-check mark must be true or false/);
});

test('Consciousness: Advises · Acts · Evolves with notes, the allowance and task copy, the moved fields keep their ids', () => {
    const html = consciousnessSectionHtml();
    assert.match(html, /What Ouroboros does when nobody asked: on its own clock, and in the places you connected\./);
    assert.deepEqual([...html.matchAll(/data-effort-value="(\w+)">(\w+)</g)].map((m) => [m[1], m[2]]), [['observe', 'Advises'], ['act', 'Acts'], ['full', 'Evolves']]);
    assert.match(html, /<strong>Acts<\/strong> &mdash; Also acts on its own/);
    assert.match(html, />Daily allowance \(USD\)</);
    assert.match(html, /Spending cap over a rolling 24 hours: its own wakes, wakes in places, and the work they start\. Settled dollars only; a subscription shows as unmetered\./);
    assert.match(html, />Max concurrent tasks</);
    assert.match(html, /How many of its own tasks may run at once\. Work you asked for never counts\./);
    for (const id of ['s-consciousness-autonomy', 's-consciousness-daily-usd', 's-consciousness-max-tasks', 's-bg-wakeup-min', 's-bg-wakeup-max', 's-presence-max-active']) {
        assert.equal(html.split(`id="${id}"`).length, 2, `${id} is rendered once`);
    }
    assert.match(html, /<details class="consciousness-more"[^>]*>\s*<summary>.*More<\/summary>/s);
    assert.match(html, />Concurrent conversations</);
    assert.match(html, />Connect a place</);
    assert.equal(consciousnessModelText({}), 'Uses Main');
    assert.equal(consciousnessModelText({ OUROBOROS_MODEL_CONSCIOUSNESS: 'openai::gpt-5.6' }), 'gpt-5.6');
});

test('the Settings places list: icon, authored label, state word, open turns, readiness; each row opens its sheet', () => {
    const { places } = normalizePlacesResponse(FIXTURE.places);
    const html = settingsPlacesHtml({ places });
    assert.equal((html.match(/data-place-open=/g) || []).length, 3);
    assert.match(html, /<span class="consciousness-place-name" data-i18n-authored>Slack · Acme<\/span>\s*<span class="consciousness-place-facts">active · 2 open turns · Connector on<\/span>/);
    assert.doesNotMatch(html, /aria-label="[^"]*Mail · inbox/, 'authored names stay in the protected visible label');
    assert.match(settingsPlacesHtml({ places: [] }), /No place yet\. Connect one from an installed connector\./);
    assert.match(settingsPlacesHtml({ error: 'Places could not be read: HTTP 404' }), /data-tone="warn"/);
});

test('an old connection the places migration could not move is said once, in Settings › Consciousness › Places', () => {
    assert.deepEqual(normalizePlacesResponse(FIXTURE.places).migrationProblems, [], 'a server without the migration says nothing');
    const { places, migrationProblems } = normalizePlacesResponse({ ...FIXTURE.places, migration_problems: [
        { binding_id: 'b-1', code: 'connector_missing', detail: 'the connector telegram-bot is not installed.' },
        { binding_id: 'b-2', code: 'connector_missing', detail: 'the connector telegram-bot is not installed' },
        { binding_id: 'b-3', code: 'binding_unreadable', detail: '' },
        { binding_id: '', code: '', detail: '' },
    ] });
    assert.deepEqual(migrationProblemLines(migrationProblems), [
        'Could not move an old connection: the connector telegram-bot is not installed. Connect it again.',
        'Could not move an old connection: binding unreadable. Connect it again.',
    ]);
    const html = settingsPlacesHtml({ places, problems: migrationProblems });
    assert.equal((html.match(/Could not move an old connection/g) || []).length, 2);
    assert.match(html, /<p class="settings-inline-note" data-tone="warn">Could not move/);
    assert.match(settingsPlacesHtml({ places: [], problems: migrationProblems }), /Connect it again\.<\/p><p class="settings-inline-note">No place yet/);
});

test('Settings reads Places when visible and after a change, including recovery from a failed read', async (t) => {
    const { window } = setup(t);
    const root = new NodeStub();
    root.innerHTML = '<section data-consciousness-places><div data-place-list></div></section>';
    root.dataset.activeSettingsTab = 'behavior';
    const state = { activePage: 'chat' };
    let reads = 0;
    const dispose = bindConsciousnessSettings(root, { state, read: async () => {
        if (++reads === 1) throw new Error('offline');
        return normalizePlacesResponse(FIXTURE.places);
    } });
    const settled = () => new Promise((resolve) => setImmediate(resolve));
    await settled();
    assert.equal(reads, 0, 'no boot read while Settings is hidden');
    state.activePage = 'settings';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
    await settled();
    assert.equal(reads, 1);
    assert.match(root.textContent, /Places could not be read: offline/);
    window.dispatchEvent(new CustomEvent('ouro:places-changed'));
    await settled();
    assert.equal(reads, 2);
    assert.match(root.textContent, /Slack · Acme/);
    assert.match(root.textContent, /Connector on/);
    assert.doesNotMatch(root.textContent, /offline/);
    dispose();
    window.dispatchEvent(new CustomEvent('ouro:places-changed'));
    await settled();
    assert.equal(reads, 2);
});

test('Settings reads Places on mount only in the visible Behavior tab', async (t) => {
    setup(t);
    for (const tab of ['behavior', 'models']) {
        const root = new NodeStub();
        root.innerHTML = '<section data-consciousness-places><div data-place-list></div></section>';
        root.dataset.activeSettingsTab = tab;
        let reads = 0;
        const dispose = bindConsciousnessSettings(root, { state: { activePage: 'settings' }, read: async () => {
            reads++; return normalizePlacesResponse(FIXTURE.places);
        } });
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(reads, tab === 'behavior' ? 1 : 0);
        dispose();
    }
});
