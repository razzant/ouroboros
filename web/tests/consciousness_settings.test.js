// The owner's consciousness, places and safety words, read against the contract's shapes
// (fixtures/consciousness_contract.json): the words, the one client module, the facts Settings says beside the two
// Safety levels (settings_safety.js) and the Agents second-check mark. The two levels and Your rules as rows and a
// page are in settings_plain_pages.test.js.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    AUTONOMY_LEVELS, PLACE_STATES, SAFETY_LEVELS, SILENT_CARD, durationText, usdText,
} from '../modules/consciousness_words.js';
import {
    SAFETY_MODEL_KEYS, connectPlace, markRecheckRow, normalizeConsciousness, normalizePlacesResponse, parseRoster,
    recheckFromSettings, saveCheckerSettings, saveSafetyLevel, saveSafetyRules, updatePlace, disconnectPlace,
} from '../modules/consciousness_api.js';
import { agentRowHandle, refreshSafetySkipCounter, safetyFacts, secondCheckWords } from '../modules/settings_safety.js';
import { chooseRecheckRow, recheckDraft, recheckMarkHtml, syncRecheckState } from '../modules/safety_recheck.js';
import { parseAvailableSubagentsSetting, buildAvailableSubagentsSetting } from '../modules/subagents_settings.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));

/** Record every request the client sends and answer from `reply(url, body)`. */
function stubFetch(reply) {
    const calls = [];
    const previous = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        const body = init.body ? JSON.parse(init.body) : null;
        calls.push({ url: String(url), method: init.method || 'GET', body, headers: init.headers || {} });
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
    // The Safety scale is named by what is read before it runs (owner, 11 Nov 2026); stored off / light / full.
    assert.deepEqual(SAFETY_LEVELS.map((level) => [level.value, level.label]), [['off', 'Off'], ['light', 'Outgoing'], ['full', '+ Commands']]);
    assert.equal(SAFETY_LEVELS[1].note, 'Messages, posts, skills, connectors and starting work are read first.');
    assert.equal(SAFETY_LEVELS[2].note, 'Shell commands and scripts are read first too. File edits are not.');
});

test('absent money is the dash; a known zero and a known amount are said; durations need both ends', () => {
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

test('the facts beside the two levels name what really checks and who re-checks', () => {
    const settings = FIXTURE.settings;
    assert.equal(safetyFacts(settings).supervisor, 'The Light model');
    assert.equal(safetyFacts({ ...settings, OUROBOROS_MODEL_SAFETY: 'anthropic::claude-opus-5.5' }).supervisor, 'claude-opus-5.5');
    // A local Safety Supervisor model is its ` (local)` label; no Local flag of its own is read or written.
    assert.equal(safetyFacts({ OUROBOROS_MODEL_SAFETY: 'qwen3-8b (local)' }).supervisor, 'qwen3-8b (local)');
    for (const name of ['settings_safety.js', 'consciousness_api.js']) {
        assert.doesNotMatch(fs.readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8'), /USE_LOCAL_SAFETY/);
    }
    const recheck = recheckFromSettings(settings);
    assert.equal(agentRowHandle('packet_checker', settings), 'gpt-5.6');
    assert.equal(secondCheckWords(recheck, settings), 'gpt-5.6 re-checks flagged actions');
    assert.equal(safetyFacts(settings).secondCheck, 'gpt-5.6 re-checks flagged actions');
    const unmarked = { enabled: true, items: settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, rechecks_flags: false })) };
    assert.equal(secondCheckWords(recheckFromSettings({ OUROBOROS_SUBAGENTS: unmarked }), settings), 'Off');
    assert.match(secondCheckWords(recheckFromSettings({ OUROBOROS_SUBAGENTS: { enabled: true, items: [] } }), settings), /^Off\. No agent can re-check yet/);
    assert.equal(secondCheckWords(recheckFromSettings({}), settings), 'Off', 'an unreadable roster says nothing more');
    const disabled = { enabled: true, items: settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, enabled: false })) };
    assert.equal(secondCheckWords(recheckFromSettings({ OUROBOROS_SUBAGENTS: disabled }), settings), 'Off: the marked agent is disabled');
    const delegationOff = { ...settings.OUROBOROS_SUBAGENTS, enabled: false };
    assert.equal(secondCheckWords(recheckFromSettings({ OUROBOROS_SUBAGENTS: delegationOff }), settings), 'Off: Delegation is off in Agents');
});

test('«not measured yet» is a fact wherever the second check is offered, the default Off included', () => {
    const unmarked = { enabled: true, items: FIXTURE.settings.OUROBOROS_SUBAGENTS.items.map((item) => ({ ...item, rechecks_flags: false })) };
    for (const roster of [unmarked, FIXTURE.settings.OUROBOROS_SUBAGENTS]) {
        assert.equal(safetyFacts({ ...FIXTURE.settings, OUROBOROS_SUBAGENTS: roster }).unmeasured, true);
    }
    assert.equal(safetyFacts({ ...FIXTURE.settings, OUROBOROS_SUBAGENTS: 'not a roster' }).unmeasured, false, 'no readable roster, no second check offered');
});

test('an organization\'s Supervisor is said, with what it closes, and offers nothing to change', () => {
    const managed = { ...FIXTURE.settings, _meta: { policy_state: { deployment: {
        checker: { model: 'anthropic::claude-opus-5.5' }, organization_rules: true, allowed_exits: ['repeat_after_dangerous'] } } } };
    const facts = safetyFacts(managed);
    assert.equal(facts.fixed, true);
    assert.equal(facts.supervisor, 'claude-opus-5.5 · Set by your organization');
    assert.equal(facts.secondCheck, 'None. Your organization’s Supervisor decides alone.');
    assert.equal(facts.organizationRules, true);
    assert.match(facts.exits, /^Set by your organization: your approval cannot override a DANGEROUS refusal; /);
    assert.deepEqual([safetyFacts(FIXTURE.settings).fixed, safetyFacts(FIXTURE.settings).exits, safetyFacts(FIXTURE.settings).organizationRules],
        [false, '', false], 'outside a deployment none of it is said');
});

test('the let-through counter and the spare-model line say 24 h only when the events they read cover a day', async () => {
    const now = Date.parse('2026-11-06T12:00:00Z');
    const at = (hoursAgo) => new Date(now - hoursAgo * 3600 * 1000).toISOString();
    const walk = (hoursAgo, model) => ({ type: 'safety_route_fallback', ts: at(hoursAgo), answered: model ? { model, role: 'fallback:0' } : null });
    const reads = [];
    const counter = async (entries) => {
        // One node per line, as the page draws them (settings_ui.js): each selector answers with its own.
        const nodes = { '[data-safety-skips]': { textContent: '' }, '[data-safety-spares]': { textContent: '' } };
        await refreshSafetySkipCounter({ querySelector: (selector) => nodes[selector] ?? null },
            { now, logsTail: async (...read) => { reads.push(read); return { entries }; } });
        return Object.values(nodes).map((node) => node.textContent);
    };
    // A short log reaches past the day: it is the whole day, and what is older than the day is not counted.
    assert.deepEqual(await counter([{ type: 'safety_mode_skip', ts: at(2) }, walk(3, 'openai::gpt-6-luna'), walk(5, null),
        { type: 'safety_verdict', ts: at(4) }, walk(25, 'openai::gpt-6-luna'), { type: 'safety_verdict', ts: at(26) },
        { type: 'task_done', ts: at(30) }]), ['1 safety check let through in the last 24 h.',
        '1 safety check was answered by a spare model in the last 24 h, because the Supervisor\'s own model gave no answer:'
            + ' gpt-6-luna ×1. 1 check went to the spare models and got no answer. 1 check with a recorded verdict in the same span.']);
    // A full tail of 2,000 events from the last three hours never claims the day it did not read.
    const busy = Array.from({ length: 2000 }, (_, i) => (i === 1 ? walk(3 - i / 1000, 'openai::gpt-6-luna')
        : { type: i ? 'llm_round' : 'safety_mode_skip', ts: at(3 - i / 1000) }));
    const [said, spares] = await counter(busy);
    assert.match(said, /^1 safety check let through since \d\d:\d\d \(the latest 2,000 events\)\.$/);
    assert.match(spares, /^1 safety check was answered by a spare model since (\d\d:\d\d) \(the latest 2,000 events\), because the Supervisor's own model gave no answer: gpt-6-luna ×1\. 0 checks with a recorded verdict in the same span\.$/);
    assert.equal(spares.match(/since (\d\d:\d\d)/)[1], said.match(/since (\d\d:\d\d)/)[1], 'one span for both lines');
    assert.doesNotMatch(said + spares, /24 h/);
    // It reads each event's type and time, and a walk's answering model: the compact view, never 2,000 whole rows.
    assert.deepEqual(reads, [['events', 2000, 'compact'], ['events', 2000, 'compact']]);
});

test('the shared safety helpers declare no origin unless their caller does', async () => {
    const stub = stubFetch(() => ({ ok: true }));
    try {
        await saveSafetyLevel('full');
        await saveSafetyRules('Ask first.');
        await saveCheckerSettings({ OUROBOROS_MODEL: 'openai::main' });
    } finally { stub.restore(); }
    assert.deepEqual(stub.calls.map(({ headers }) => headers), Array(3).fill({ 'Content-Type': 'application/json' }));
});

test('the second-check mark shows only on eligible rows, one row at a time, and marks the roster as the owner chose', () => {
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
    // The roster carries the owner's choice as one mark at most.
    assert.deepEqual(markRecheckRow(roster, recheckDraft()), { enabled: true, items: [packet('a'), packet('b')] });
    assert.deepEqual(markRecheckRow(roster, 'b').items.map((item) => item.rechecks_flags), [false, true]);
    // A marked row that cannot run now keeps its mark in view, so the owner can untick it.
    syncRecheckState(recheckFromSettings({ OUROBOROS_SUBAGENTS: { enabled: true, items: [{ ...packet('a', true), enabled: false }, packet('b')] } }), null);
    assert.match(recheckMarkHtml({ subagent_id: 'a' }, 1), /data-recheck-row="a"><input[^>]*checked> Re-checks flagged actions/);
    assert.match(recheckMarkHtml({ subagent_id: 'b' }, 2), /data-recheck-row="b"><input/);
});

test('the checker keys mirror the gateway\'s owner transaction', () => {
    const server = fs.readFileSync(new URL('../../ouroboros/gateway/safety_settings.py', import.meta.url), 'utf8');
    const block = server.match(/SAFETY_MODEL_KEYS = frozenset\(\{([\s\S]*?)\}\)/)?.[1] || '';
    assert.deepEqual([...SAFETY_MODEL_KEYS].sort(), [...block.matchAll(/"([A-Z_]+)"/g)].map((match) => match[1]).sort());
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
