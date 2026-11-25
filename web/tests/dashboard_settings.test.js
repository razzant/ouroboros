// The Dashboard tabs' own settings (dashboard.js mountTabSettings): rows of settings_rows.js that save themselves, drawn
// once outside every region a tab repaints, in the cards the approved picture gives them; the way back to Settings
// (the page layer); Costs' subscriptions from the passive quota read. The fake server below answers as the server does:
// the receipt (`changes`), `_meta.effects`, a token per key, a refused `if_unchanged` and `unset`.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { STATUS_READ_LIFETIME_MS } from '../modules/claudexor_status_store.js';
import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { AUTONOMY_LEVELS } from '../modules/consciousness_autonomy.js';
import { cardWords, initConsciousnessDashboard, waitWords } from '../modules/consciousness_dashboard.js';
import { allowancePresentation, costDashboardPresentation, initCosts, subscriptionDoors } from '../modules/costs.js';
import { focusSettingRow, initDashboard, mountTabSettings } from '../modules/dashboard.js';
import { SAFETY_LEVELS } from '../modules/consciousness_words.js';
import { initEvolution, whenMode, whenName, whenWords } from '../modules/evolution.js';
import { CODECS, GUARDED_WRITERS, HIDDEN_WHEN, bindSettingRows, commitSetting, forgetSettingRows, outcomeLine, readSettings, settingsReadMessage, settingsRow,
    settingsSnapshot } from '../modules/settings_controls.js';
import { spendGapsWords } from '../modules/route_editor_primitives.js';
import { ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { initStorage } from '../modules/storage.js';
import { initUpdates } from '../modules/updates.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const source = (name) => readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8');
const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));
const shown = (window, tab) => window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab } }));
const rowNode = (root, id) => root.querySelectorAll('[data-setting-row]').find((node) => node.dataset.settingRow === id);
const result = (root, id) => rowNode(root, id).querySelector('[data-setting-result]').textContent;

// When a key's change applies, as GET /api/settings `_meta.effects` says it (settings_scales.setting_effect).
const EFFECTS = { OUROBOROS_BG_WAKEUP_MIN: 'next_wake', OUROBOROS_BG_WAKEUP_MAX: 'next_wake', OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 'next_wake' };

/** /api/settings over `stored`, answering as the server does: a read gives the values with `_meta.effects` and a token
 *  per key; a save stores its body and answers the receipt (`changes`: old, new and the installed value's token). A
 *  save that carries `if_unchanged` is refused whole with 409 when a token is no longer the key's, and writes nothing;
 *  `unset` removes its keys. */
function settingsServer(t, values = {}) {
    forgetSettingRows();
    const env = setup(t);
    const stored = { ...values };
    const posts = [];
    const tokens = Object.fromEntries(Object.keys(stored).map((key) => [key, `${key}@0`]));
    let version = 0;
    env.routes.set('/api/settings', (_url, init = {}) => {
        if (init.method !== 'POST') return response({ ...stored, _meta: { effects: EFFECTS, tokens: { ...tokens } } });
        const { if_unchanged: expected = {}, unset = [], ...body } = JSON.parse(init.body);
        posts.push(JSON.parse(init.body));
        const stale = Object.keys(expected).find((key) => tokens[key] !== expected[key]);
        if (stale) return response({ error: `${stale} changed since it was read.`, saved: false, code: 'settings_changed_elsewhere' }, 409);
        const changes = {};
        for (const key of [...Object.keys(body), ...unset]) {
            tokens[key] = `${key}@${version += 1}`;
            changes[key] = { old: stored[key], new: body[key], token: tokens[key], ...(key in stored ? {} : { old_absent: true }) };
            if (unset.includes(key)) delete stored[key]; else stored[key] = body[key];
        }
        return response({ status: 'saved', changes });
    });
    return { ...env, stored, posts, tokens };
}

const MOVED = {
    'dashboard:consciousness': ['OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX',
        'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'OUROBOROS_PRESENCE_MAX_ACTIVE'],
    'dashboard:evolution': ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE',
        'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE'],
    'dashboard:updates': ['OUROBOROS_UPDATE_CHANNEL'],
    'dashboard:storage': ['OUROBOROS_GC_RETENTION_DAYS'],
};

test('each moved key is written by exactly one row, on its own tab; the doors write none and are found elsewhere', () => {
    for (const [page, keys] of Object.entries(MOVED)) {
        for (const key of keys) assert.deepEqual(ROWS.filter((row) => row.keys.includes(key)).map((row) => row.page), [page], key);
        assert.deepEqual(rowsOf(page).flatMap((row) => row.keys).sort(), [...keys].sort(), `${page} writes nothing else`);
    }
    assert.deepEqual(rowById('OUROBOROS_BG_WAKEUP').keys, ['OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX'], 'the wake range is one row');
    assert.deepEqual(rowsOf('dashboard:consciousness').map((row) => row.id), ['OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'OUROBOROS_BG_WAKEUP',
        'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'consciousness-allowance', 'OUROBOROS_PRESENCE_MAX_ACTIVE', 'consciousness-model', 'consciousness-safety',
        'consciousness-places', 'consciousness-evolution']);
    assert.deepEqual(rowsOf('dashboard:consciousness', { fold: true }).map((row) => row.id), ['OUROBOROS_PRESENCE_MAX_ACTIVE']);
    for (const id of ['consciousness-allowance', 'consciousness-model', 'consciousness-safety', 'consciousness-places', 'consciousness-evolution']) {
        const door = rowById(id);
        assert.deepEqual([door.control.kind, door.keys, door.aliases], ['door', [], []], `${id}: search finds its setting in its own home`);
    }
    // The strip is the levels' one list, and the row says the chosen level in that list's own sentence.
    const autonomy = rowById('OUROBOROS_CONSCIOUSNESS_AUTONOMY');
    assert.deepEqual(autonomy.control.options, AUTONOMY_LEVELS.map(({ value, label }) => ({ value, label })));
    for (const level of AUTONOMY_LEVELS) {
        assert.equal(cardWords({ OUROBOROS_CONSCIOUSNESS_AUTONOMY: level.value }).OUROBOROS_CONSCIOUSNESS_AUTONOMY.caption, level.note);
    }
    assert.equal(cardWords({ OUROBOROS_CONSCIOUSNESS_AUTONOMY: 'nonsense' }).OUROBOROS_CONSCIOUSNESS_AUTONOMY.caption, '', 'an unknown level says nothing');
    // No row of these tabs hides a fact behind a per-row More: the picture has one line under a name.
    for (const page of Object.keys(MOVED)) assert.deepEqual(rowsOf(page).filter((row) => row.more).map((row) => row.id), [], page);
    const channel = rowById('OUROBOROS_UPDATE_CHANNEL');
    assert.equal(channel.hiddenWhen, 'managedDeployment', 'a managed deployment leaves the channel out');
    assert.deepEqual(channel.control.options.map((option) => [option.value, option.label, Boolean(option.note)]),
        [['stable', 'Stable', true], ['qa', 'QA', true], ['development', 'Development', true]], 'each choice carries its sentence');
});

test('the evolution choice is one row over two keys: one body carries both, and they read back as the choice', async (t) => {
    const server = settingsServer(t, { OUROBOROS_POST_TASK_EVOLUTION: 'false', OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    const row = rowById('OUROBOROS_POST_TASK_EVOLUTION');
    assert.deepEqual(row.keys, ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE']);
    assert.deepEqual(CODECS.evolutionWhen.read(await readSettings(), row), [false, 'llm']);
    const saved = await commitSetting(row, [true, 'every_n:5']);
    assert.deepEqual(server.posts, [{ OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:5' }]);
    assert.equal(outcomeLine(row, saved).text, 'Saved · applies to new tasks · was Off', 'it names the choice it replaced');
    assert.deepEqual(CODECS.evolutionWhen.read(await readSettings(), row), [true, 'every_n:5']);
    assert.equal(outcomeLine(row, await commitSetting(row, [false, 'llm'])).text, 'Saved · applies to new tasks · was Every 5 tasks');
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_POST_TASK_EVOLUTION: false, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    // The server's reading of what it stores: words for on, a cadence in any case, a malformed one as "after each task".
    const read = (on, cadence) => CODECS.evolutionWhen.read({ OUROBOROS_POST_TASK_EVOLUTION: on, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: cadence }, row);
    assert.deepEqual(read('True', ' EVERY_N:3 '), [true, 'every_n:3']);
    assert.deepEqual(read(true, 'every_n:0'), [true, 'llm']);
    assert.deepEqual(read('0', 'off'), [false, 'off']);
    assert.equal(CODECS.evolutionWhen.read(null, row), undefined, 'nothing read is not Off');
    assert.deepEqual([[true, 'llm'], [true, 'every_n:3'], [true, 'off'], [false, 'every_n:3']].map(whenMode), ['llm', 'every_n', 'off', 'off']);
    // A count that is no whole number above 0 is refused here and never sent.
    for (const bad of ['every_n:0', 'every_n:', 'every_n:2.5', 'every_n:abc', 'every_n:-1']) {
        const refused = await commitSetting(row, [true, bad]);
        assert.equal(outcomeLine(row, refused).text, 'Not saved. Enter a whole number of tasks, 1 or more.', bad);
    }
    assert.equal(server.posts.length, 2);
    // The row says its choice in its own sentence: "every N" is when it asks itself, never a promise of a cycle.
    const said = (on, cadence) => whenWords({ OUROBOROS_POST_TASK_EVOLUTION: on, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: cadence }).OUROBOROS_POST_TASK_EVOLUTION.caption;
    assert.deepEqual([said(false, 'every_n:3'), said(true, 'llm'), said(true, 'every_n:3'), said(true, 'every_n:1')], [
        'It starts no cycle by itself.',
        'After each task it decides whether one improvement is worth a cycle.',
        'After every 3 tasks, chats included, it decides whether one improvement is worth a cycle.',
        'After every task, chats included, it decides whether one improvement is worth a cycle.']);
    assert.equal(whenWords(null).OUROBOROS_POST_TASK_EVOLUTION.caption, '', 'nothing read says nothing');
});

test('the wake range is one row of two numbers; a first number above the second is refused before anything is sent', async (t) => {
    const server = settingsServer(t, { OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200 });
    const row = rowById('OUROBOROS_BG_WAKEUP');
    await readSettings();
    const refused = await commitSetting(row, [7200, 300]);
    assert.equal(outcomeLine(row, refused).text, 'Not saved. The shortest wait is above the longest.');
    assert.equal(server.posts.length, 0);
    const saved = await commitSetting(row, [600, 3600]);
    assert.deepEqual(server.posts, [{ OUROBOROS_BG_WAKEUP_MIN: 600, OUROBOROS_BG_WAKEUP_MAX: 3600 }], 'both numbers in one body');
    assert.equal(outcomeLine(row, saved).text, 'Saved · applies from the next wake · was 300 to 7200');
    await commitSetting(row, [900, 900]);
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_BG_WAKEUP_MIN: 900, OUROBOROS_BG_WAKEUP_MAX: 900 }, 'one fixed wait is a range too');
    // The caption says the seconds in minutes or hours; the doors say their values.
    assert.deepEqual([60, 300, 90, 3600, 5400, 86400, 0, '', 'x'].map(waitWords),
        ['1 minute', '5 minutes', '1.5 minutes', '1 hour', '1.5 hours', '24 hours', '', '', '']);
    const words = (settings) => Object.fromEntries(Object.entries(cardWords(settings)).map(([id, said]) => [id, said.caption ?? said.value]));
    assert.deepEqual(words({ OUROBOROS_CONSCIOUSNESS_AUTONOMY: 'act', OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200,
        OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20, OUROBOROS_MODEL_CONSCIOUSNESS: '', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'light',
        OUROBOROS_POST_TASK_EVOLUTION: 'true', OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:3' }), {
        OUROBOROS_CONSCIOUSNESS_AUTONOMY: AUTONOMY_LEVELS[1].note, OUROBOROS_BG_WAKEUP: 'It picks a moment between 5 minutes and 2 hours.',
        'consciousness-allowance': '$20', 'consciousness-model': '',
        'consciousness-safety': SAFETY_LEVELS.find((level) => level.value === 'light').label, 'consciousness-evolution': 'Every 3 tasks' });
    assert.deepEqual(words({ OUROBOROS_BG_WAKEUP_MIN: 900, OUROBOROS_BG_WAKEUP_MAX: 900, OUROBOROS_CONSCIOUSNESS_DAILY_USD: '12.5',
        OUROBOROS_MODEL_CONSCIOUSNESS: 'anthropic::claude-test', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'off',
        OUROBOROS_POST_TASK_EVOLUTION: false, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:3' }), {
        OUROBOROS_CONSCIOUSNESS_AUTONOMY: '', OUROBOROS_BG_WAKEUP: 'It wakes every 15 minutes.',
        'consciousness-allowance': '$12.50', 'consciousness-model': '', 'consciousness-safety': 'Off', 'consciousness-evolution': 'Off' });
    assert.deepEqual(words({}), { OUROBOROS_CONSCIOUSNESS_AUTONOMY: '', OUROBOROS_BG_WAKEUP: '', 'consciousness-allowance': '',
        'consciousness-model': '', 'consciousness-safety': '', 'consciousness-evolution': '' },
    'a value that was not read is not said: no "$0", no level and no "Off" the document did not carry');
    // The model door says the route the next wake takes, as the consciousness read publishes it and Costs words it:
    // the slot's text alone does not say its lane (its box checked under a cloud Main is the local runtime).
    const model = (route, settings = {}) => cardWords(settings, route)['consciousness-model'].value;
    assert.equal(model({ value: 'claudexor::codex=gpt-6-astra', layer: 'main', local: false }), 'gpt-6-astra (Main model)');
    assert.equal(model({ value: 'anthropic::claude-test', layer: 'consciousness', local: false }), 'claude-test');
    assert.equal(model({ value: 'openai/gpt-6', layer: 'consciousness', local: true }, { OUROBOROS_MODEL_CONSCIOUSNESS: '' }), 'Local model');
    assert.equal(model(null, { OUROBOROS_MODEL_CONSCIOUSNESS: 'anthropic::claude-test' }), '', 'a route that was not read is not guessed from the slot');
    // The Safety door speaks the scale's one list, whatever its words are, and the evolution door the choice's own names.
    for (const level of SAFETY_LEVELS) assert.equal(words({ OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: level.value })['consciousness-safety'], level.label);
    assert.deepEqual([[true, 'llm'], ['1', 'every_n:1'], [true, 'every_n:12'], ['false', 'llm']].map(whenName),
        ['After each task', 'Every task', 'Every 12 tasks', 'Off']);
});

test('Consciousness: its cards are drawn once beside the summary; a refresh leaves them and a typed field alone', async (t) => {
    const server = settingsServer(t, { OUROBOROS_CONSCIOUSNESS_AUTONOMY: 'act', OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200,
        OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 2, OUROBOROS_PRESENCE_MAX_ACTIVE: 3, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20, OUROBOROS_MODEL_CONSCIOUSNESS: '',
        OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'full', OUROBOROS_POST_TASK_EVOLUTION: 'true', OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    const opened = [];
    const controller = initConsciousnessDashboard({
        mount: server.mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' },
        read: async () => ({ ...normalizeConsciousness({ ...FIXTURE.consciousness,
            model_route: { value: 'openai/gpt-6', layer: 'consciousness', local: true } }), recentWakes: [] }),
        readPlacesFn: async () => ({ places: [] }),
        openSettings: (path, options) => { opened.push(['settings', path, options]); },
        openDashboardTab: (tab, options) => { opened.push(['tab', tab, options]); },
    });
    await controller.refresh();
    await settle();
    const card = server.mount.querySelector('[data-cx-settings]');
    assert.equal(server.mount.querySelector('[data-cx-summary]').querySelector('[data-setting-row]'), null);
    // Two cards as the picture draws them: what it does on its own (with its fold), then the doors around it.
    const sections = card.querySelectorAll('section');
    const ids = (node) => node.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow);
    assert.deepEqual(sections.map((section) => [section.getAttribute('aria-label'), section.querySelector('.dashboard-card-title').textContent]),
        [['On its own', 'On its own'], ['Around it', 'Around it']]);
    assert.deepEqual(sections.map(ids), [['OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'OUROBOROS_BG_WAKEUP', 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS',
        'consciousness-allowance', 'OUROBOROS_PRESENCE_MAX_ACTIVE'],
    ['consciousness-model', 'consciousness-safety', 'consciousness-places', 'consciousness-evolution']], 'the doors in the picture\'s order');
    assert.deepEqual(ids(card).sort(), rowsOf('dashboard:consciousness').map((row) => row.id).sort(), 'every row of the tab is drawn once');
    const fold = card.querySelector('[data-settings-fold]');
    assert.equal(fold.querySelector('summary').textContent, 'More: conversations at once');
    assert.deepEqual(ids(fold), ['OUROBOROS_PRESENCE_MAX_ACTIVE']);
    const autonomy = rowNode(card, 'OUROBOROS_CONSCIOUSNESS_AUTONOMY');
    assert.deepEqual(autonomy.querySelectorAll('[data-setting-value]').map((button) => button.textContent), ['Advises', 'Acts', 'Evolves'],
        'what a row says of itself never overwrites a choice of its strip');
    assert.equal(autonomy.querySelector('.active').textContent, 'Acts');
    assert.equal(autonomy.querySelector('[data-row-caption]').textContent, AUTONOMY_LEVELS[1].note, 'the chosen level in its own sentence');
    assert.equal(autonomy.hasAttribute('data-no-meta'), false, 'drawn with no meta, its caption is its line (settings.css: two lines)');
    const range = rowNode(card, 'OUROBOROS_BG_WAKEUP');
    assert.deepEqual(range.querySelectorAll('[data-setting-control]').map((input) => input.value), ['300', '7200']);
    assert.equal(range.querySelectorAll('.settings-row-unit').map((unit) => unit.textContent).join(' '), 'to s', '[300] to [7200] s');
    assert.equal(range.querySelector('[data-row-caption]').textContent, 'It picks a moment between 5 minutes and 2 hours.');
    // One name for one limit: the door carries the name its row has on Costs (a rolling 24 hours), and the sum alone.
    assert.equal(rowNode(card, 'consciousness-allowance').querySelector('.ui-row-title').textContent, rowById('OUROBOROS_CONSCIOUSNESS_DAILY_USD').name);
    assert.equal(rowNode(card, 'consciousness-allowance').querySelector('[data-row-value]').textContent, '$20');
    assert.equal(rowNode(card, 'consciousness-model').querySelector('[data-row-value]').textContent, 'Local model',
        'an empty slot whose box is checked under a cloud Main: the published route, not the slot\'s text');
    assert.equal(rowNode(card, 'consciousness-places').querySelector('[data-row-value]'), null, 'a door with nothing to say says nothing');
    assert.deepEqual(['consciousness-safety', 'consciousness-evolution'].map((id) => [rowNode(card, id).querySelector('.ui-row-title').textContent,
        rowNode(card, id).querySelector('[data-row-value]').textContent]),
    [['Its actions are checked', SAFETY_LEVELS.find((level) => level.value === 'full').label], ['Improving itself', 'After each task']]);
    // The summary's refresh (every 20 s) and a save elsewhere repaint around the card: its nodes and a typed draft stay.
    const tasks = rowNode(card, 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS').querySelector('[data-setting-control]');
    tasks.focus();
    tasks.value = '4';
    await card.fire('input', { target: tasks });
    server.stored.OUROBOROS_CONSCIOUSNESS_MAX_TASKS = 5;
    await controller.refresh();
    server.window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }));
    await settle();
    assert.equal(rowNode(card, 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS').querySelector('[data-setting-control]'), tasks, 'the same field');
    assert.equal(tasks.value, '4', 'the draft stays through the refresh');
    // Leaving the field saves it and the row says what happened.
    await card.fire('change', { target: tasks });
    await settle();
    assert.deepEqual(server.posts, [{ OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 4 }]);
    assert.equal(result(card, 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS'), 'Saved · applies from the next wake · was 5 · Undo');
    // A range upside down is refused in its row; nothing is sent.
    const [shortest, longest] = range.querySelectorAll('[data-setting-control]');
    shortest.value = '7200';
    longest.value = '300';
    await card.fire('change', { target: longest });
    await settle();
    assert.equal(result(card, 'OUROBOROS_BG_WAKEUP'), 'Not saved. The shortest wait is above the longest.');
    assert.equal(server.posts.length, 1);
    // A saved range says itself anew in minutes or hours.
    shortest.value = '600';
    longest.value = '600';
    await card.fire('change', { target: longest });
    await settle();
    assert.equal(range.querySelector('[data-row-caption]').textContent, 'It wakes every 10 minutes.');
    // The allowance door opens its row in Costs; the model door its role in Models, which leads back here; the places
    // door opens Circles.
    let circles = 0;
    const onCircles = () => { circles += 1; };
    server.window.addEventListener('ouro:open-circles', onCircles);
    await card.fire('click', { target: rowNode(card, 'consciousness-allowance') });
    await card.fire('click', { target: rowNode(card, 'consciousness-model') });
    await card.fire('click', { target: rowNode(card, 'consciousness-places') });
    server.window.removeEventListener('ouro:open-circles', onCircles);
    assert.deepEqual(opened.slice(0, 1), [['tab', 'costs', { row: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD' }]]);
    assert.deepEqual(opened[1], ['settings', 'models/role/consciousness', undefined]);
    assert.deepEqual([opened.length, circles], [2, 1], 'the places door leaves through Circles alone');
    await card.fire('click', { target: rowNode(card, 'consciousness-safety') });
    await card.fire('click', { target: rowNode(card, 'consciousness-evolution') });
    assert.deepEqual(opened[2], ['settings', 'permissions', { row: 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE' }]);
    assert.deepEqual(opened[3], ['tab', 'evolution', { row: 'OUROBOROS_POST_TASK_EVOLUTION' }]);
    controller.destroy();
});

test('Undo on the wake range and the evolution choice sends the old values back with their tokens; a stale token writes nothing', async (t) => {
    const [MIN, MAX, ON, CADENCE] = ['OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX', 'OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE'];
    const server = settingsServer(t, { [MIN]: 300, [MAX]: 7200, [ON]: 'false', [CADENCE]: 'every_n:4' });
    // Both writers hand the precondition and the unset keys to the generic write, so they are in the guarded set.
    assert.deepEqual(['wakeRange', 'evolutionWhen'].map((name) => GUARDED_WRITERS.has(name)), [true, true]);
    const host = new NodeStub();
    server.mount.appendChild(host);
    const [range, when] = [rowById('OUROBOROS_BG_WAKEUP'), rowById('OUROBOROS_POST_TASK_EVOLUTION')];
    host.innerHTML = settingsRow(range) + settingsRow(when);
    const rows = bindSettingRows(host);
    await readSettings();
    rows.paint();
    const undo = (id) => rowNode(host, id).querySelector('[data-setting-undo]');
    const press = async (id) => { await host.fire('click', { target: undo(id) }); await settle(); };
    // The range: both old numbers go back in one body, each key under the token its own save installed.
    const [shortest, longest] = rowNode(host, range.id).querySelectorAll('[data-setting-control]');
    const type = async (a, b) => { shortest.value = a; longest.value = b; await host.fire('change', { target: longest }); await settle(); };
    await type('600', '3600');
    assert.equal(result(host, range.id), 'Saved · applies from the next wake · was 300 to 7200 · Undo');
    const installed = { [MIN]: server.tokens[MIN], [MAX]: server.tokens[MAX] };
    await press(range.id);
    assert.deepEqual(server.posts.at(-1), { [MIN]: 300, [MAX]: 7200, if_unchanged: installed });
    assert.deepEqual([server.stored[MIN], server.stored[MAX]], [300, 7200]);
    assert.deepEqual([shortest.value, longest.value], ['300', '7200'], 'the fields show what Undo put back');
    // A stale token: another device changed the shortest wait meanwhile. Undo is refused whole and writes nothing.
    await type('900', '1800');
    Object.assign(server.stored, { [MIN]: 1200 });
    Object.assign(server.tokens, { [MIN]: 'elsewhere' });
    const before = server.posts.length;
    await press(range.id);
    assert.deepEqual([server.posts.length, Object.keys(server.posts.at(-1).if_unchanged)], [before + 1, [MIN, MAX]]);
    assert.deepEqual([server.stored[MIN], server.stored[MAX]], [1200, 1800], 'neither number was written');
    assert.equal(result(host, range.id), 'Changed on another device. This is the current value.');
    assert.equal(undo(range.id), null, 'no Undo is offered over a value this page did not install');
    // The evolution choice: "was" names the old choice, and Undo sends the stored pair itself back, the word 'false' as off.
    await commitSetting(when, [true, 'llm']);
    await settle();
    assert.equal(result(host, when.id), 'Saved · applies to new tasks · was Off · Undo');
    const pair = { [ON]: server.tokens[ON], [CADENCE]: server.tokens[CADENCE] };
    await press(when.id);
    assert.deepEqual(server.posts.at(-1), { [ON]: false, [CADENCE]: 'every_n:4', if_unchanged: pair }, 'the count it had comes back with it');
    assert.equal(result(host, when.id), 'Saved · applies to new tasks · was After each task · Undo', 'and that Undo can be undone');
    await commitSetting(when, [true, 'every_n:2']);
    await settle();
    Object.assign(server.tokens, { [CADENCE]: 'elsewhere' });
    const stored = { ...server.stored };
    await press(when.id);
    assert.deepEqual(server.stored, stored, 'a stale token on one key of the pair writes neither');
    assert.equal(result(host, when.id), 'Changed on another device. This is the current value.');
    rows.dispose();
});

test('Undo of the evolution choice restores a key that was absent by unsetting it', async (t) => {
    const [ON, CADENCE] = ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE'];
    const server = settingsServer(t, { [ON]: false });
    const host = new NodeStub();
    server.mount.appendChild(host);
    const when = rowById(ON);
    host.innerHTML = settingsRow(when);
    const rows = bindSettingRows(host);
    await readSettings();
    await commitSetting(when, [true, 'every_n:2']);
    await settle();
    const installed = { [ON]: server.tokens[ON], [CADENCE]: server.tokens[CADENCE] };
    await host.fire('click', { target: rowNode(host, ON).querySelector('[data-setting-undo]') });
    await settle();
    assert.deepEqual(server.posts.at(-1), { [ON]: false, unset: [CADENCE], if_unchanged: installed });
    assert.deepEqual(server.stored, { [ON]: false }, 'the cadence is absent again, not stored as a word');
    rows.dispose();
});

test('Evolution: the cycles card presses one choice; "Every N tasks" shows its count, and both keys go in one body', async (t) => {
    const server = settingsServer(t, { OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:7',
        OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD: 0, OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE: '' });
    await readSettings();
    // The fixture's nodes lack two calls the history's painter makes; the card does not use them.
    for (const [name, body] of [['replaceChildren', function (...nodes) { this.textContent = ''; this.append(...nodes); }],
        ['prepend', function (...nodes) { nodes.reverse().forEach((node) => this.insertBefore(node, this.firstChild)); }]]) {
        if (!NodeStub.prototype[name]) {
            NodeStub.prototype[name] = body;
            t.after(() => { delete NodeStub.prototype[name]; });
        }
    }
    server.routes.set('/api/state', response({ runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } }));
    server.routes.set('/api/evolution-data', () => new Promise(() => {}));
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' };
    const { dispose } = initEvolution({ ws: { on: () => () => {} }, mount: server.mount, state });
    const card = server.mount.querySelector('[data-evo-settings]');
    // The choice and the floor are one card's rows; the standing direction is a card of its own, its row drawn solo.
    const sections = card.querySelectorAll('section');
    assert.deepEqual(sections.map((section) => [section.getAttribute('aria-label'), section.hasAttribute('data-solo'),
        section.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow)]), [
        ['Cycles after your tasks', false, ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD']],
        ['What to work on', true, ['OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE']]]);
    assert.match(card.querySelector('#setting-OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE').getAttribute('placeholder'), /^For example: /);
    const caption = () => rowNode(card, 'OUROBOROS_POST_TASK_EVOLUTION').querySelector('[data-row-caption]').textContent;
    const choices = card.querySelectorAll('[data-evo-when]');
    const pressed = () => choices.filter((button) => button.getAttribute('aria-pressed') === 'true').map((button) => button.textContent);
    const count = card.querySelector('[data-evo-n]');
    const painted = (n) => {
        assert.equal(count.value, String(n));
        assert.equal(caption(), `After every ${n} tasks, chats included, it decides whether one improvement is worth a cycle.`);
    };
    painted(7);  // bind uses the snapshot, before this tab's own read answers
    assert.deepEqual(pressed(), ['Every N tasks']);
    assert.equal(count.parentElement.hidden, false);
    assert.equal(count.parentElement.querySelector('.settings-row-unit').textContent, 'N');
    await settle();
    const click = (target) => card.fire('click', { type: 'click', target });
    const leave = (target) => card.fire('change', { type: 'change', target });
    const type = async (n) => { count.value = String(n); await card.fire('input', { target: count }); document.activeElement = document.body; };
    // A read from elsewhere paints both halves, even without this tab's own read or a showing event.
    server.stored.OUROBOROS_POST_TASK_EVOLUTION_CADENCE = 'every_n:11';
    await readSettings();
    painted(11);
    server.stored.OUROBOROS_POST_TASK_EVOLUTION_CADENCE = 'every_n:7';
    await readSettings();
    painted(7);
    await type(9);
    await readSettings();
    assert.equal(count.value, '9', 'an unsubmitted draft also survives a landed read');
    const originalSettings = server.routes.get('/api/settings');
    const held = deferred();
    const followup = deferred();
    let writeMode = 'hold';
    let holdRead = false;
    server.routes.set('/api/settings', async (url, init = {}) => {
        if (init.method !== 'POST' && holdRead) return followup.promise;
        if (init.method === 'POST' && writeMode === 'hold') return held.promise;
        if (init.method === 'POST' && writeMode === 'unknown') return response({ error: 'answer lost' }, 500);
        return originalSettings(url, init);
    });
    await leave(count);
    await settle();
    assert.deepEqual([count.value, count.disabled], ['9', true], 'the write holds the typed count');
    await readSettings();
    assert.equal(count.value, '9', 'a landed read during the write keeps it too');
    held.resolve(response({ saved: false, error: 'the document refused this value' }, 500));
    await settle();
    await readSettings();
    assert.deepEqual([count.value, count.disabled], ['9', false], 'a refusal and a later read keep the edit');
    assert.match(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), /Not saved\. The document refused this value/);
    await type(9);
    writeMode = 'unknown';
    const beforeRecovery = server.calls.length;
    await leave(count);
    await settle();
    assert.equal(server.calls.length, beforeRecovery + 2, 'an unconfirmed write starts its recovery read');
    await readSettings();
    assert.equal(count.value, '9', 'both the recovery read and a later read keep an unconfirmed draft');
    assert.equal(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), 'Not confirmed yet');
    writeMode = 'normal';
    // Clear the unknown outcome's prerequisite read, then hold just the read after the accepted write.
    const write = server.routes.get('/api/settings');
    server.routes.set('/api/settings', async (url, init = {}) => {
        const answer = await write(url, init);
        if (init.method === 'POST') holdRead = true;
        return answer;
    });
    await leave(count);
    await settle();
    painted(9);
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:9' });
    assert.match(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), /^Saved/);
    followup.resolve(response({ error: 'read failed' }, 500));
    await settle();
    painted(9);
    holdRead = false;
    server.routes.set('/api/settings', originalSettings);
    server.stored.OUROBOROS_POST_TASK_EVOLUTION_CADENCE = 'every_n:11';
    await readSettings();
    painted(11);  // nothing pending: a later external value wins over the last save
    await type(0);
    await leave(count);
    await settle();
    assert.equal(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), 'Not saved. Enter a whole number of tasks, 1 or more.');
    assert.equal(server.posts.length, 1, 'a count of 0 is never sent');
    await readSettings();
    assert.equal(count.value, '0');
    state.dashboardActiveSubtab = 'logs';
    shown(server.window, 'logs');
    painted(11);  // leaving ends the rejected draft
    state.dashboardActiveSubtab = 'evolution';
    shown(server.window, 'evolution');
    await settle();
    await click(choices[2]);
    assert.equal(server.posts.length, 1, 'pressing the pressed choice sends nothing');
    server.routes.set('/api/settings', (url, init = {}) => init.method === 'POST' ? originalSettings(url, init) : response({}, 500));
    await click(choices[1]);
    await settle();
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    assert.deepEqual(pressed(), ['After each task']);
    assert.equal(count.parentElement.hidden, true);
    assert.equal(caption(), 'After each task it decides whether one improvement is worth a cycle.');
    dispose();
});

test('Updates marks its accepted channel while the follow-up read is held, and keeps it when that read fails', async (t) => {
    const server = settingsServer(t, { OUROBOROS_UPDATE_CHANNEL: 'stable' });
    initUpdates({ mount: server.mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'updates' }, ws: server.ws });
    await settle();
    const card = server.mount.querySelector('[data-updates-settings]');
    const choices = card.querySelectorAll('[data-setting-value]');
    const pressed = () => choices.filter((button) => button.getAttribute('aria-pressed') === 'true').map((button) => button.dataset.settingValue);
    assert.deepEqual(pressed(), ['stable']);
    const original = server.routes.get('/api/settings');
    const followup = deferred();
    server.routes.set('/api/settings', (url, init = {}) => init.method === 'POST' ? original(url, init) : followup.promise);
    await card.fire('click', { target: choices[1] });
    await settle();
    assert.deepEqual(pressed(), ['qa'], 'the accepted write paints before the GET answers');
    assert.match(result(card, 'OUROBOROS_UPDATE_CHANNEL'), /^Saved/);
    followup.resolve(response({ error: 'read failed' }, 500));
    await settle();
    assert.deepEqual(pressed(), ['qa'], 'a failed follow-up cannot put Stable back');
    // Another device moves the channel; Undo is refused as changed elsewhere, and the card shows the value that stands.
    server.routes.set('/api/settings', original);
    Object.assign(server.stored, { OUROBOROS_UPDATE_CHANNEL: 'development' });
    Object.assign(server.tokens, { OUROBOROS_UPDATE_CHANNEL: 'elsewhere' });
    await card.fire('click', { target: rowNode(card, 'OUROBOROS_UPDATE_CHANNEL').querySelector('[data-setting-undo]') });
    await settle();
    assert.match(result(card, 'OUROBOROS_UPDATE_CHANNEL'), /^Changed on another device/);
    assert.deepEqual(pressed(), ['development'], 'a write refused as changed elsewhere paints the current value at once');
});

test('Storage: how long finished work is kept is a row of its Cleanup card, read only while the tab is shown', async (t) => {
    const server = settingsServer(t, { OUROBOROS_GC_RETENTION_DAYS: 7 });
    server.routes.set('/api/storage', response({ count: null, counting: null, error: null, volume: null, retention_days: 7 }));
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'storage' };
    const storage = initStorage({ state, mount: server.mount });
    await settle();
    const cleanup = server.mount.querySelector('[aria-label="Cleanup"]');
    const row = rowNode(cleanup, 'OUROBOROS_GC_RETENTION_DAYS');
    const field = row.querySelector('[data-setting-control]');
    // The row leads the card, above the tray the tab repaints, as the picture has it.
    const parts = cleanup.children.filter((child) => child.nodeType !== 3);
    assert.deepEqual(parts.slice(0, 3).map((child) => child.className.split(' ')[0]), ['storage-card-title', 'storage-retention', 'storage-tray']);
    assert.equal(cleanup.querySelector('[data-storage-tray]').querySelector('[data-setting-row]'), null);
    assert.equal(field.value, '7');
    assert.equal(row.querySelector('.settings-row-unit').textContent, 'days');
    const reads = () => server.calls.filter((url) => url === '/api/settings').length;
    assert.equal(reads(), 1);
    // Saving reads the tab again: the server's retention shapes the cleanup it plans.
    const storageReads = () => server.calls.filter((url) => url === '/api/storage').length;
    const before = storageReads();
    field.value = '30';
    await cleanup.querySelector('[data-storage-settings]').fire('change', { target: field });
    await settle();
    assert.deepEqual(server.posts, [{ OUROBOROS_GC_RETENTION_DAYS: 30 }]);
    assert.equal(storageReads(), before + 1);
    assert.equal(rowNode(cleanup, 'OUROBOROS_GC_RETENTION_DAYS'), row, 'the read repainted the card around the row');
    const hidden = reads();
    state.dashboardActiveSubtab = 'logs';
    shown(server.window, 'logs');
    server.window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }));
    await settle();
    assert.equal(reads(), hidden, 'a hidden tab reads nothing on a save elsewhere');
    storage.destroy();
    state.dashboardActiveSubtab = 'storage';
    shown(server.window, 'storage');
    await settle();
    assert.equal(reads(), hidden, 'a destroyed tab reads nothing');
});

test('Updates: the channel is a card of three choices with their sentences; a press saves it; a managed deployment has none', async (t) => {
    const server = settingsServer(t, { OUROBOROS_UPDATE_CHANNEL: 'stable' });
    const host = new NodeStub();
    server.mount.appendChild(host);
    const saved = [];
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'updates' };
    // The card as updates.js mounts it (its custom control is that module's own markup, checked in the browser test).
    const row = rowById('OUROBOROS_UPDATE_CHANNEL');
    const mounted = mountTabSettings(host, 'updates', { state, cards: [{ label: 'Which updates', solo: true }], onSaved: (id) => saved.push(id),
        onPaint: (settings) => host.querySelectorAll('[data-setting-value]').forEach((choice) =>
            choice.setAttribute('aria-pressed', String(choice.dataset.settingValue === String(settings?.OUROBOROS_UPDATE_CHANNEL ?? '')))),
        custom: () => row.control.options.map((option) => `<button data-setting-value="${option.value}" aria-pressed="false">${option.label}</button>`).join('') });
    await settle();
    const card = host.querySelector('section');
    assert.deepEqual([card.getAttribute('aria-label'), card.hasAttribute('data-solo'), card.hidden], ['Which updates', true, false]);
    assert.equal(rowNode(card, 'OUROBOROS_UPDATE_CHANNEL').querySelector('.ui-row-title').textContent, 'Which updates', 'the row names the card');
    const choices = card.querySelectorAll('[data-setting-value]');
    const chosen = () => choices.filter((choice) => choice.getAttribute('aria-pressed') === 'true').map((choice) => choice.dataset.settingValue);
    assert.deepEqual(chosen(), ['stable']);
    await host.fire('click', { target: choices[1] });
    await settle();
    assert.deepEqual(server.posts, [{ OUROBOROS_UPDATE_CHANNEL: 'qa' }]);
    assert.deepEqual([chosen(), saved], [['qa'], ['OUROBOROS_UPDATE_CHANNEL']], 'the saved choice is the marked one, and the tab hears its own save');
    assert.equal(result(card, 'OUROBOROS_UPDATE_CHANNEL'), 'Saved · applies to new tasks · was Stable · Undo', 'a delayed choice names what it replaced');
    await host.fire('click', { target: choices[1] });
    await settle();
    assert.equal(server.posts.length, 1, 'pressing the chosen one sends nothing');
    // A deployment that updates the installation itself: the row is left out, and its card takes no room.
    assert.equal(HIDDEN_WHEN.managedDeployment({ _meta: { policy_state: { deployment: { managed: true } } } }), true);
    server.routes.set('/api/settings', () => response({ OUROBOROS_UPDATE_CHANNEL: 'qa', _meta: { policy_state: { deployment: { managed: true } } } }));
    shown(server.window, 'updates');
    await settle();
    assert.deepEqual([rowNode(card, 'OUROBOROS_UPDATE_CHANNEL').hidden, card.hidden], [true, true]);
    mounted.dispose();
});

test('a card says what its rows add from the last read, and only then: captions and door values are spans made once', async (t) => {
    const server = settingsServer(t, { OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 2 });
    const host = new NodeStub();
    server.mount.appendChild(host);
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' };
    let said = { OUROBOROS_CONSCIOUSNESS_MAX_TASKS: { caption: 'first' }, 'consciousness-allowance': { value: '$1 a day' } };
    const mounted = mountTabSettings(host, 'consciousness', { state, words: () => said,
        cards: [{ title: 'A', rows: ['OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'consciousness-allowance'] }, { title: 'B <b>', lead: 'L & l' }] });
    assert.equal(host.querySelector('[data-row-caption]'), null, 'nothing is said before a read');
    await settle();
    const tasks = rowNode(host, 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS');
    const caption = tasks.querySelector('[data-row-caption]');
    assert.equal(caption.textContent, 'first');
    assert.equal(caption.parentElement.children.at(-1), caption, 'the caption ends the meta line');
    assert.equal(tasks.querySelector('.ui-row-actions').nextElementSibling, tasks.querySelector('[data-setting-result]'),
        'the result line stands after the control, never inside the words');
    const door = rowNode(host, 'consciousness-allowance');
    assert.equal(door.querySelector('.ui-row-actions').firstChild, door.querySelector('[data-row-value]'), 'a value stands before the chevron');
    said = { OUROBOROS_CONSCIOUSNESS_MAX_TASKS: { caption: 'second' } };
    shown(server.window, 'consciousness');
    await settle();
    assert.equal(tasks.querySelector('[data-row-caption]'), caption, 'the same span');
    assert.deepEqual([caption.textContent, door.querySelector('[data-row-value]').textContent], ['second', ''], 'what is no longer said goes');
    // A card without `rows` takes every row no other card names; its title and lead are text, never markup.
    const [first, rest] = host.querySelectorAll('section');
    assert.deepEqual(first.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow), ['OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'consciousness-allowance']);
    assert.equal(rest.querySelectorAll('[data-setting-row]').length, rowsOf('dashboard:consciousness').length - 2);
    assert.deepEqual([rest.querySelector('.dashboard-card-title').textContent, rest.querySelector('.dashboard-settings-lead').textContent, rest.querySelector('b')],
        ['B &lt;b&gt;', 'L &amp; l', null]);
    mounted.dispose();
});

test('one writer: these tabs post nothing to /api/settings themselves and read no old save booleans', () => {
    const code = (name) => source(name).split('\n').filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line)).join('\n');
    for (const name of ['consciousness_dashboard.js', 'evolution.js', 'updates.js', 'storage.js', 'dashboard.js', 'costs.js']) {
        const text = code(name);
        assert.doesNotMatch(text, /\/api\/settings/, `${name} reaches settings only through settings_controls.js`);
        assert.doesNotMatch(text, /\b(immediate_changed|next_task_changed|saveSettings|jsonPost)\b/, name);
        // Updates reads restart_required of the update's own answers, never of a settings save.
        const restart = [...text.matchAll(/^.*\brestart_required\b.*$/gm)].map(([line]) => line);
        assert.deepEqual(restart.filter((line) => !/(progress|data|err\?\.body)\??\.restart_required|'restart_required'/.test(line)), [], name);
        if (name !== 'updates.js') assert.deepEqual(restart, [], name);
    }
    // The links into the old Settings tabs are gone.
    assert.doesNotMatch(source('consciousness_dashboard.js'), /openSettingsTab|data-cx-act="settings"|settings-consciousness/);
    assert.doesNotMatch(source('updates.js'), /openSettingsTab|data-open-settings-advanced/);
    for (const [name, tab] of [['consciousness_dashboard.js', 'consciousness'], ['evolution.js', 'evolution'], ['updates.js', 'updates'], ['storage.js', 'storage']]) {
        assert.match(source(name), new RegExp(`mountTabSettings\\([^]*?'${tab}'`), `${name} mounts its card`);
    }
});

test('returning to a Settings level preserves its history entry, including a refused return', async (t) => {
    const { createLayerStack } = await import('../modules/layers.js');
    const { createPageNavigation } = await import('../modules/page_navigation.js');
    const { bindPageBack, renderPageHeader } = await import('../modules/page_header.js');
    for (const trigger of ['button', 'escape']) for (const allowed of [true, false]) {
        await t.test(`${trigger}/${allowed}`, async (sub) => {
            const { mount, document, window } = setup(sub);
            const calls = [], order = [];
            const layers = createLayerStack({ win: window, doc: document, history: {
                pushState: () => calls.push('push'), back: () => calls.push('back'),
            } });
            sub.after(() => layers.destroy());
            let decide, returning = false;
            const decision = new Promise(resolve => { decide = resolve; });
            const nav = createPageNavigation({ layers, doc: document, canLeave: () => returning ? decision : true,
                capture: () => ({ key: 'settings:tools', label: 'Settings', title: 'Settings › Tools', restore() {
                    order.push('restore');
                    layers.push({ id: 'settings-level-1', page: true, modal: false, focus: false });
                } }) });
            sub.after(() => nav.destroy());
            const panel = new NodeStub('section'); mount.append(panel);
            panel.innerHTML = renderPageHeader({ page: 'skills' }) + '<input data-setting-control>';
            const off = bindPageBack({ layers, currentHeader: () => panel.querySelector('.app-page-header'), doc: document, win: window });
            sub.after(off);
            await nav.go('skills', () => {});
            returning = true;
            const field = panel.querySelector('input');
            field.blur = () => { order.push('blur'); document.activeElement = document.body; };
            field.focus();
            await settle();
            const button = panel.querySelector('[data-shell-back]');
            assert.equal(button.getAttribute('aria-label'), 'Back to Settings › Tools');
            if (trigger === 'button') button.dispatchEvent({ type: 'click' });
            else layers.back('escape');
            await settle();
            assert.deepEqual(calls, ['push'], 'a pending leave retains the return entry');
            assert.equal(panel.querySelector('[data-shell-back]'), button);
            decide(allowed);
            await settle();
            assert.deepEqual(calls, ['push'], 'a return never consumes and replaces the entry');
            assert.deepEqual(layers.ids(), [allowed ? 'settings-level-1' : 'page-return-1']);
            assert.equal(panel.querySelector('[data-shell-back]'), allowed ? null : button);
            assert.deepEqual(order, allowed ? ['blur', 'restore'] : [], 'a refused return leaves the field alone; an accepted return blurs before restoring');
            if (allowed) {
                layers.back('escape'); await settle();
                assert.deepEqual(calls, ['push', 'back'], 'the Settings level consumes only its own entry');
            }
        });
    }
});

test('the automatic Dashboard view keeps its focused rail through both width crossings before any callback', (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.append(content);
    window.CSS = { supports: () => true };
    window.ouroLayout = { layout: 'docked' };
    let width = 1000;
    const before = { box: NodeStub.prototype.getBoundingClientRect, observer: globalThis.ResizeObserver };
    NodeStub.prototype.getBoundingClientRect = () => ({ width });
    const observers = [];
    globalThis.ResizeObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
    t.after(() => { NodeStub.prototype.getBoundingClientRect = before.box; globalThis.ResizeObserver = before.observer; });
    const layers = new Map();
    const dashboard = initDashboard({ state: { activePage: 'dashboard', dashboardActiveSubtab: 'costs' },
        layers: { push: layer => layers.set(layer.id, layer), remove: id => layers.delete(id) } });
    const shell = dashboard.page.querySelector('.dashboard-shell');
    const header = dashboard.page.querySelector('.app-page-header');
    const list = dashboard.page.querySelector('.dashboard-tabs');
    const band = () => ['data-empty', 'data-controls'].map((name) => header.hasAttribute(name));
    // The stub moves focus only when code calls focus(), so no claim about the focused row is made here: a browser keeps
    // it while the rail stays drawn, and that is the 'auto' view and the same mounted list asserted below (the sheets
    // draw the rail in 'auto' at every width: tests/test_shell_layout_static.py
    // ::test_an_automatic_section_page_keeps_its_list_when_the_container_narrows_before_javascript). Back's return of
    // the focus to the chosen row is the test 'a page explicitly chosen beside the Dashboard list survives narrowing'.
    try {
        for (const [layout, pixels] of [['stack', 390], ['docked', 1000], ['stack', 390]]) {
            width = pixels;
            assert.deepEqual([dashboard.page.dataset.sectionView, shell.dataset.sectionView], ['auto', 'auto'],
                'only an explicit page may hide the rail: the container can narrow before either callback');
            window.ouroLayout.layout = layout;
            window.dispatchEvent(new CustomEvent('ouro:layout-changed', { detail: { layout } }));
            assert.equal(dashboard.page.querySelector('.dashboard-tabs'), list, 'the same list stays mounted');
            observers.forEach(fn => fn([{ target: dashboard.page }]));
            assert.equal(shell.dataset.sectionView, 'auto', 'the later observer keeps the automatic view');
            assert.deepEqual(band(), [false, false], 'the band carries no measured fact: the two-pane query decides it is empty');
        }
        const controls = new NodeStub(); controls.setAttribute('data-dashboard-controls', 'costs');  // the stub reads attributes
        dashboard.setControls('costs', controls);
        assert.deepEqual(band(), [false, true], 'the open page\'s controls are the one fact the band reads');
        dashboard.activateTab('costs');
        for (const pixels of [1000, 390]) {
            width = pixels;
            observers.forEach(fn => fn([{ target: dashboard.page }]));
            assert.equal(shell.dataset.sectionView, 'page', 'an explicit page stays open in either layout');
        }
        layers.get('dashboard-level').onClose();
        assert.equal(shell.dataset.sectionView, 'auto', 'Back restores the list and its automatic page on a desk');
    } finally { dashboard.destroy(); }
});

test('the Dashboard owns only its list level; a jump keeps its return while its pages change', (t) => {
    const { mount } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.append(content);
    const stack = new Map();
    let jump = null;
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    const dashboard = initDashboard({ state, layers: {
        push: layer => stack.set(layer.id, layer), remove: id => stack.delete(id), returnPage: () => jump,
    } });
    dashboard.activateTab('costs');
    assert.equal(stack.get('dashboard-level').back.label, 'Dashboard');
    stack.get('dashboard-level').onClose();
    assert.equal(dashboard.page.dataset.sectionView, 'auto');
    jump = { id: 'return', back: { label: 'Settings' } };
    dashboard.activateTab('costs');
    dashboard.activateTab('logs');
    assert.deepEqual([...stack.keys()], [], 'a jump has no second list Back');
    assert.equal(jump.back.label, 'Settings', 'its section choices leave the return alone');
    const saved = dashboard.capture();
    // A return names the section, then its page, as Settings does ("Back to Dashboard › Logs").
    assert.deepEqual([saved.label, saved.title], ['Logs', 'Dashboard › Logs']);
    dashboard.activateTab('activity');
    saved.restore();
    assert.equal(state.dashboardActiveSubtab, 'logs');
    const panel = dashboard.page.querySelector('#dashboard-panel-costs');
    panel.innerHTML = '<details><summary>More</summary><div data-setting-row="rounds"><input data-setting-control></div></details>';
    let view;
    panel.querySelector('[data-setting-row]').scrollIntoView = () => { view = dashboard.page.dataset.sectionView; };
    dashboard.activateTab('costs', { row: 'rounds' });
    assert.equal(view, 'page', 'reveal happens after the level is visible');
    assert.equal(panel.querySelector('details').open, true);
    assert.equal(document.activeElement, panel.querySelector('input'));
    assert.equal(focusSettingRow(panel, 'missing'), false);
    dashboard.destroy();
});

// ---- Costs › Subscriptions: the passive quota read --------------------------------------------------------------

const NOW = Date.parse('2026-11-11T12:00:00Z');
const inHours = (hours) => new Date(NOW + hours * 3_600_000).toISOString();
const profile = (harness, id, name, enabled = true) => ({ profile: { harness_id: harness, profile_id: id, display_name: name, enabled }, status: {}, identity: {} });
const windowOf = (harness, id, used, resets) => ({ subject: { harness, subject_id: id }, freshness: 'fresh', constraints: [{ id: 'w', used_ratio: used, resets_at: resets }] });
const passive = (overrides = {}) => ({
    view: 'quota', reads: { catalog: 'not_read', accounts: 'ok', quota: 'ok' }, read_errors: [], quota_absences: [],
    profiles: { profiles: [profile('codex', 'work', 'Work'), profile('codex', 'home', 'Home'), profile('codex', 'old', 'Old', false),
        profile('claude', 'max', 'Max plan')] },
    quota: [windowOf('codex', 'work', 0.2, inHours(2)), windowOf('codex', 'home', 1, inHours(5)), windowOf('claude', 'max', 0.8, inHours(30))],
    ...overrides,
});

test('Subscriptions: one door per kind of account in counted facts (never an average), then Accounts and keys', () => {
    const general = (meta) => ({ path: 'accounts', name: 'Accounts and keys', meta, value: '', tone: '', harness: '' });
    const doors = subscriptionDoors(passive(), { nowMs: NOW });
    assert.deepEqual(doors, [
        { path: 'accounts/codex', name: 'Codex', harness: 'codex', value: '1 of 2 have room', tone: '',
            meta: '2 in rotation · 1 at the limit · next reset in 5h · 1 out of rotation' },
        { path: 'accounts/claude', name: 'Claude Code', harness: 'claude', value: 'Has room', tone: '', meta: '1 in rotation' },
        general('Sign in, add an account or take one out of rotation.'),
    ]);
    assert.doesNotMatch(JSON.stringify(doors), /%/, 'no share of several accounts is said, so none is averaged');
    // Every account of a kind at its limit is the owner's to read (amber); the first to return is named.
    const spent = subscriptionDoors(passive({ quota: [windowOf('codex', 'work', 1, inHours(9)), windowOf('codex', 'home', 1, inHours(5)),
        windowOf('claude', 'max', 1, inHours(30))] }), { nowMs: NOW });
    assert.deepEqual(spent.slice(0, 2).map(({ value, tone, meta }) => [value, tone, meta]), [
        ['0 of 2 have room', 'warn', '2 in rotation · 2 at the limit · next reset in 5h · 1 out of rotation'],
        ['At its limit', 'warn', '1 in rotation · next reset in 30h']]);
    // An account whose window was not observed is counted as not checked, never as having room; so is a full window
    // whose reset has passed ("availability not proven").
    const partial = subscriptionDoors(passive({ quota: [windowOf('codex', 'work', 0.2, inHours(2))] }), { nowMs: NOW });
    assert.deepEqual(partial.slice(0, 2).map(({ value, tone, meta }) => [value, tone, meta]), [
        ['1 of 2 have room', '', '2 in rotation · 1 not checked · 1 out of rotation'], ['Usage not checked', '', '1 in rotation']]);
    const stale = subscriptionDoors(passive({ quota: [windowOf('codex', 'work', 0.2, inHours(2)), windowOf('codex', 'home', 1, inHours(-1)),
        windowOf('claude', 'max', 1, inHours(-1))] }), { nowMs: NOW });
    assert.deepEqual(stale.slice(0, 2).map(({ value, tone, meta }) => [value, tone, meta]), [
        ['1 of 2 have room', '', '2 in rotation · 1 not checked · 1 out of rotation'], ['Usage not checked', '', '1 in rotation']]);
    // A quota that was not read licenses no usage claim; a kind with every account out of rotation says so.
    assert.deepEqual(subscriptionDoors(passive({ reads: { catalog: 'not_read', accounts: 'ok', quota: 'failed' } }), { nowMs: NOW })
        .slice(0, 2).map(({ value, meta }) => [value, meta]), [['Limits not checked', '2 in rotation · 1 out of rotation'], ['Limits not checked', '1 in rotation']]);
    assert.deepEqual(subscriptionDoors(passive({ profiles: { profiles: [profile('cursor', 'c', 'C', false)] }, quota: [] }), { nowMs: NOW })[0],
        { path: 'accounts/cursor', name: 'Cursor', harness: 'cursor', value: 'None in rotation', tone: '', meta: '1 out of rotation' });
    // A roster that was not read lists no account.
    for (const unread of [passive({ reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' }, profiles: {}, quota: [] }),
        passive({ reads: { catalog: 'not_read', accounts: 'failed', quota: 'ok' } })]) {
        assert.deepEqual(subscriptionDoors(unread, { nowMs: NOW }), [general('The accounts could not be read now.')]);
    }
    assert.deepEqual(subscriptionDoors(null, { transportError: 'HTTP 502' }).map((door) => door.meta), ['The accounts could not be read now.']);
    assert.deepEqual(subscriptionDoors(passive({ profiles: { profiles: [] }, quota: [] }), { nowMs: NOW }).map((door) => door.meta), ['No subscription account yet.']);
});

test('Costs reads subscriptions only through the passive view, on show and Refresh, and its doors open Accounts with a way back', async (t) => {
    const { mount, routes, window, calls } = setup(t);
    const opened = [];
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/settings', response({ TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20 }));
    routes.set('/api/consciousness', response({ enabled: true, spent_24h_usd: 1, daily_usd: 20 }));
    // The tab reads against the real clock: these windows end after it.
    const ahead = (hours) => new Date(Date.now() + hours * 3_600_000).toISOString();
    routes.set('/api/claudexor/status?view=quota', response(passive({ quota: [windowOf('codex', 'work', 0.2, ahead(2)),
        windowOf('codex', 'home', 1, ahead(5)), windowOf('claude', 'max', 0.8, ahead(30))] })));
    initCosts({ mount, state: { activePage: 'dashboard' }, openSettings: (path, options) => { opened.push([path, options]); },
        openDashboardTab: (tab) => { opened.push([tab]); } });
    const block = mount.querySelector('.costs-subscriptions');
    assert.match(block.textContent, /Subscriptions/);
    assert.equal(calls.some((url) => url.startsWith('/api/claudexor')), false, 'nothing is read before the tab is shown');
    shown(window, 'costs');
    await settle();
    shown(window, 'logs');
    window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }));
    await settle();
    mount.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(calls.filter((url) => url.startsWith('/api/claudexor')), ['/api/claudexor/status?view=quota', '/api/claudexor/status?view=quota'],
        'the passive view alone, on show and on Refresh; a save elsewhere reads none');
    const doors = block.querySelectorAll('[data-costs-account]');
    assert.deepEqual(doors.map((door) => door.dataset.costsAccount), ['accounts/codex', 'accounts/claude', 'accounts']);
    // Each door leads with its glyph and says its value before its chevron; the general door has no value to say.
    assert.deepEqual(doors.map((door) => [door.hasAttribute('data-lead'), Boolean(door.querySelector('.dashboard-settings-glyph')),
        door.querySelector('.settings-row-value')?.textContent ?? null]), [[true, true, '1 of 2 have room'], [true, true, 'Has room'], [true, true, null]]);
    await block.fire('click', { target: doors[0] });
    await block.fire('click', { target: doors[2] });
    assert.deepEqual(opened, [['accounts/codex', undefined], ['accounts', undefined]], 'links need no return callback');
    doors[0].focus();
    shown(window, 'costs');
    await settle();
    assert.equal(block.querySelector('[data-costs-account="accounts/codex"]'), doors[0], 'return refresh retains the pressed row');
    assert.equal(document.activeElement, doors[0], 'its focus survives the asynchronous answer');
    // A read that fails keeps the one door, saying why.
    routes.set('/api/claudexor/status?view=quota', response({ error: 'down' }, 502));
    mount.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(block.querySelectorAll('[data-costs-account]').map((door) => [door.dataset.costsAccount, door.querySelector('.ui-row-meta').textContent]),
        [['accounts', 'The accounts could not be read now.']]);
    assert.equal(document.activeElement, doors[2], 'a vanished account returns to its surviving Accounts anchor');
    const elsewhere = mount.querySelector('#btn-refresh-costs');
    elsewhere.focus();
    shown(window, 'costs');
    await settle();
    assert.equal(document.activeElement, elsewhere, 'a refresh does not take focus from elsewhere');
});

test('Costs › Subscriptions whose status body never comes ends at the status reads\' lifetime as a read that failed; Refresh reads anew', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });  // before the fixture, so the timers it keeps are these
    const { mount, routes, window } = setup(t);
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/settings', response({ TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20 }));
    routes.set('/api/consciousness', response({ enabled: true, spent_24h_usd: 1, daily_usd: 20 }));
    // The engine answered the request and its body never comes, abort or not: a stalled read, not a host that is offline.
    let bodies = 0;
    routes.set('/api/claudexor/status?view=quota', async () => ({ ok: true, status: 200,
        json: () => { bodies += 1; return new Promise(() => {}); } }));
    initCosts({ mount, state: { activePage: 'dashboard' }, openSettings() {}, openDashboardTab() {} });
    const block = mount.querySelector('.costs-subscriptions');
    const doors = () => block.querySelectorAll('[data-costs-account]').map((door) => [door.dataset.costsAccount,
        door.querySelector('.ui-row-meta').textContent]);
    shown(window, 'costs');
    await settle();
    t.mock.timers.tick(STATUS_READ_LIFETIME_MS - 1);
    await settle();
    assert.deepEqual([bodies, doors()], [1, []], 'within its lifetime the read is still out');
    assert.match(block.querySelector('[data-costs-subscriptions]').textContent, /Reading/);
    t.mock.timers.tick(1);
    await settle();
    assert.deepEqual(doors(), [['accounts', 'The accounts could not be read now.']],
        'at its lifetime it is a read that failed, said as such: no engine is called stopped');
    routes.set('/api/claudexor/status?view=quota', response(passive()));
    mount.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(doors().map(([path]) => path), ['accounts/codex', 'accounts/claude', 'accounts'], 'Refresh reads anew');
});

test('Costs › Subscriptions cancels its old quota body on Refresh and on either exit without painting it', async (t) => {
    const { mount, routes, window } = setup(t);
    const state = { activePage: 'dashboard' };
    routes.set('/api/settings', response({}));
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/consciousness', response({}));
    const pending = [];
    routes.set('/api/claudexor/status?view=quota', (_url, { signal }) => {
        const body = deferred();
        pending.push({ signal, ...body });
        return { ok: true, status: 200, json: () => body.promise };
    });
    initCosts({ mount, state });
    const doors = () => mount.querySelectorAll('[data-costs-account]').map((door) => door.dataset.costsAccount);
    const noAccounts = passive({ profiles: { profiles: [] }, quota: [] });
    shown(window, 'costs');
    await settle();
    const [old] = pending;
    assert.equal(old.signal.aborted, false);
    await mount.querySelector('#btn-refresh-costs').fire('click');
    assert.equal(old.signal.aborted, true, 'Refresh released the old transport while its body was pending');
    assert.equal(pending.length, 2);
    pending[1].resolve(passive());
    old.resolve(noAccounts);  // a body reader that ignores abort
    await settle();
    const expected = ['accounts/codex', 'accounts/claude', 'accounts'];
    assert.deepEqual(doors(), expected);

    for (const leave of [() => shown(window, 'logs'), () => {
        state.activePage = 'chat';
        window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    }]) {
        shown(window, 'costs');
        await settle();
        const latest = pending.at(-1);
        assert.equal(latest.signal.aborted, false);
        leave();
        assert.equal(latest.signal.aborted, true, 'leaving Costs frees the transport');
        latest.resolve(noAccounts);
        await settle();
        assert.deepEqual(doors(), expected, 'the hidden panel keeps its last completed read');
    }
});

test('Costs › Subscriptions resumes an unfinished read when Dashboard is shown again', async (t) => {
    const { mount, routes, window } = setup(t);
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'costs' };
    routes.set('/api/settings', response({}));
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/consciousness', response({}));
    const pending = [];
    routes.set('/api/claudexor/status?view=quota', (_url, { signal }) => {
        const body = deferred();
        pending.push({ signal, ...body });
        return { ok: true, status: 200, json: () => body.promise };
    });
    initCosts({ mount, state });
    const host = mount.querySelector('[data-costs-subscriptions]');
    const doors = () => mount.querySelectorAll('[data-costs-account]').map((door) => door.dataset.costsAccount);
    shown(window, 'costs');
    await settle();
    assert.equal(pending.length, 1);
    assert.match(host.textContent, /Reading…/);

    state.activePage = 'chat';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    assert.equal(pending[0].signal.aborted, true, 'leaving Dashboard still releases the transport');
    assert.match(host.textContent, /Reading…/, 'the cancelled first read has not replaced its placeholder');

    state.activePage = 'dashboard';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } }));
    assert.equal(pending.length, 2, 'the page return starts one fresh read without a subtab event');
    pending[1].resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    await settle();
    assert.deepEqual(doors(), ['accounts']);
    assert.doesNotMatch(host.textContent, /Reading…/);

    pending[0].resolve(passive());
    await settle();
    assert.deepEqual(doors(), ['accounts'], 'the abandoned answer cannot repaint the resumed read');
});

// The Dashboard and Costs together, as app.js mounts them. The sheets draw one column's list over a hidden pane and a page
// opened as a level in it (ui.css @container sectionplace), which `getClientRects` stands for here as the existing tests do;
// the Costs page inside the pane has a box exactly while the pane has.
function costsInDashboard(t, { twoPanes = false, ws = undefined } = {}) {
    const { mount, routes, window, calls } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    window.CSS = { supports: () => twoPanes };
    routes.set('/api/settings', response({}));
    routes.set('/api/cost-breakdown', response({}, 503));
    routes.set('/api/consciousness', response({}));
    const pending = [];
    routes.set('/api/claudexor/status?view=quota', (_url, { signal }) => {
        const body = deferred();
        pending.push({ signal, ...body });
        return { ok: true, status: 200, json: () => body.promise };
    });
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    const layers = new Map();
    const dashboard = initDashboard({ state, ws, layers: { push: (layer) => layers.set(layer.id, layer), remove: (id) => layers.delete(id) } });
    const panes = dashboard.page.querySelector('.dashboard-panels');
    panes.getClientRects = () => (twoPanes || dashboard.page.dataset.sectionView === 'page' ? [{ width: 390 }] : []);
    const costs = initCosts({ mount: dashboard.page.querySelector('#dashboard-panel-costs'), state, ws, openSettings() {}, openDashboardTab() {} });
    dashboard.page.querySelector('#page-costs').getClientRects = () => (state.dashboardActiveSubtab === 'costs' ? panes.getClientRects() : []);
    const list = dashboard.page.querySelector('.dashboard-tabs');
    const open = () => list.fire('click', { target: list.querySelector('[data-dashboard-tab="costs"]') });
    const doors = () => mount.querySelectorAll('[data-costs-account]').map((door) => door.dataset.costsAccount);
    return { dashboard, layers, pending, window, state, open, doors, calls, routes, costs };
}

test('Costs › Subscriptions ends its pending read when one column’s Back or a second press of Dashboard leaves Costs, and reads anew when it is opened', async (t) => {
    const { dashboard, layers, pending, open, doors } = costsInDashboard(t);
    for (const [leave, how] of [[() => layers.get('dashboard-level').onClose(), 'Back'], [() => dashboard.pressedAgain(), 'Dashboard pressed again']]) {
        const before = pending.length;
        await open();
        await settle();
        assert.equal(pending.length, before + 1, `${how}: opening Costs reads`);
        assert.equal(layers.has('dashboard-level'), true, 'Costs is a level of the one column');
        const read = pending.at(-1);
        assert.equal(read.signal.aborted, false);
        leave();
        assert.equal(dashboard.page.dataset.sectionView, 'auto', `${how}: the list is back`);
        assert.equal(read.signal.aborted, true, `${how}: Costs is hidden, so its read is released as it is when another tab is chosen`);
    }
    await open();
    await settle();
    const [first, second, third] = pending;
    assert.equal(pending.length, 3, 'each opening starts its own read; none inherits the one left behind');
    third.resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    first.resolve(passive());
    second.resolve(passive());  // bodies that ignore their abort
    await settle();
    assert.deepEqual(doors(), ['accounts'], 'only the read of this showing paints');
    dashboard.destroy();
});

test('Costs › Subscriptions reads nothing when Dashboard returns as one column’s list, and reads anew when Costs is opened', async (t) => {
    const { dashboard, layers, pending, window, state, open, doors } = costsInDashboard(t);
    const show = (page) => {
        state.activePage = page;
        window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page } }));
    };
    await open();
    await settle();
    assert.equal(pending.length, 1, 'opening Costs reads');
    show('chat');
    assert.equal(pending[0].signal.aborted, true, 'Chat in front releases the read');
    show('dashboard');
    await settle();
    assert.equal(dashboard.page.dataset.sectionView, 'auto', 'one column: the list comes back, not Costs');
    assert.equal(pending.length, 1, 'no read starts for a page that is not on screen');
    await open();
    await settle();
    assert.equal(pending.length, 2, 'opening Costs reads anew, as a new showing');
    layers.get('dashboard-level').onClose();
    assert.equal(pending[1].signal.aborted, true, 'one column’s Back releases the read');
    show('chat');
    show('dashboard');
    await settle();
    assert.equal(pending.length, 2, 'a list that returns after Back reads nothing either');
    await open();
    await settle();
    assert.equal(pending.length, 3, 'each opening starts its own read');
    pending[2].resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    pending[0].resolve(passive());
    pending[1].resolve(passive());  // bodies that ignore their abort
    await settle();
    assert.deepEqual(doors(), ['accounts'], 'only the read of this showing paints');
    dashboard.destroy();
});

test('Costs › Subscriptions reads nothing while the arrow keys only select Costs in one column’s list, and reads when a press opens it', async (t) => {
    const { dashboard, pending, state, open } = costsInDashboard(t);
    const list = dashboard.page.querySelector('.dashboard-tabs');
    // One ArrowDown from Logs, Activity and Consciousness reaches Costs, which the list selects without opening its page.
    for (const row of list.querySelectorAll('[data-dashboard-tab]').slice(0, 3)) await list.fire('keydown', { target: row, key: 'ArrowDown', preventDefault() {} });
    await settle();
    assert.deepEqual([state.dashboardActiveSubtab, dashboard.page.dataset.sectionView], ['costs', 'auto'], 'the keys chose Costs on the list, not its page');
    assert.equal(pending.length, 0, 'a page that is not drawn reads nothing');
    await open();
    await settle();
    assert.equal(pending.length, 1, 'the press that opens the page reads');
    dashboard.destroy();
});

test('Costs › Subscriptions resumes at once when Dashboard returns with Costs beside its list, with no subtab event', async (t) => {
    const { dashboard, pending, window, state, doors } = costsInDashboard(t, { twoPanes: true });
    const announced = [];
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => announced.push(event.detail.tab));
    dashboard.activateTab('costs');
    await settle();
    assert.deepEqual([pending.length, announced], [1, ['costs']]);
    state.activePage = 'chat';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    assert.equal(pending[0].signal.aborted, true, 'leaving the Dashboard releases the read');
    state.activePage = 'dashboard';
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } }));
    assert.equal(pending.length, 2, 'the page keeps its box beside the list: the return reads at once');
    assert.deepEqual(announced, ['costs'], 'and no tab was announced for it');
    pending[1].resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    pending[0].resolve(passive());
    await settle();
    assert.deepEqual(doors(), ['accounts'], 'only the read of this showing paints');
    dashboard.destroy();
});

test('Costs › Subscriptions keeps its read while its page is on screen and ends it only when Costs’ own page is hidden', async (t) => {
    const two = costsInDashboard(t, { twoPanes: true });
    two.dashboard.activateTab('costs');
    await settle();
    two.dashboard.pressedAgain();
    assert.equal(two.pending.length, 1);
    assert.equal(two.pending[0].signal.aborted, false, 'beside its list the page keeps its box: nothing is hidden');
    two.window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-hidden', { detail: { tab: 'logs' } }));
    assert.equal(two.pending[0].signal.aborted, false, 'another tab’s page hidden is not Costs’ page');
    two.window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-hidden', { detail: { tab: 'costs' } }));
    assert.equal(two.pending[0].signal.aborted, true, 'Costs’ own page hidden ends the read');
    two.dashboard.destroy();
});

test('a Dashboard page that loses its box while the Dashboard stays is announced hidden, once, and never when the Dashboard itself is left', async (t) => {
    const { dashboard, window, state, pending } = costsInDashboard(t, { twoPanes: true });
    const hidden = [];
    window.addEventListener('ouro:dashboard-subtab-hidden', (event) => hidden.push(event.detail.tab));
    const pane = dashboard.page.querySelector('.dashboard-panels');
    let box = true;
    pane.getClientRects = () => (box ? [{ width: 390 }] : []);
    dashboard.activateTab('costs');
    await settle();
    assert.deepEqual(hidden, [], 'a page with its box is not hidden');
    box = false;  // the window narrowed to one column: the open page loses the pane beside the list
    dashboard.pressedAgain();
    dashboard.pressedAgain();
    assert.deepEqual(hidden, ['costs'], 'announced once, for the page that was open');
    assert.equal(pending[0].signal.aborted, true);
    box = true;
    dashboard.pressedAgain();
    state.activePage = 'chat';
    box = false;
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    assert.deepEqual(hidden, ['costs'], 'leaving the Dashboard is the page event’s to announce');
    dashboard.destroy();
});

// ---- Costs: every read waits for a drawn page and ends with it -------------------------------------------------------
// What Costs asks when it is shown, by address; the answers the tests give carry a transport signal and a body to resolve.
const COSTS = { totals: '/api/cost-breakdown', period: '/api/cost-breakdown?period=24h', settings: '/api/settings',
    allowance: '/api/consciousness', quota: '/api/claudexor/status?view=quota' };
const asked = (calls) => Object.values(COSTS).map((url) => calls.filter((call) => call === url).length);
// Whether each read's transport was ended: false while it is out, true once ended; a request sent with no way to end it, or
// never sent, says so.
const ended = (reads) => reads.map((read) => (!read ? 'not sent' : read.signal ? read.signal.aborted : 'no signal'));
const totalsBody = (spent) => ({ accounting: { available: true, settled_usd: spent, confirmed_usd: spent, estimated_usd: 0, reserved_usd: 0,
    unresolved_upper_bound_usd: 0, unknown_unmetered: 0, limit_usd: 100, cost_final: true }, total_calls: 1 });
const periodBody = (model) => ({ period: { available: true, models: [{ model, calls: 1, cost: 1 }], cache_writes: [] } });
const allowanceBody = (spent) => ({ enabled: true, spent_24h_usd: spent, daily_usd: 10 });
function heldReads(routes, urls) {
    const held = Object.fromEntries(urls.map((url) => [url, []]));
    for (const url of urls) {
        routes.set(url, (_url, { signal }) => {
            const body = deferred();
            held[url].push({ signal, ...body });
            return { ok: true, status: 200, json: () => body.promise };
        });
    }
    return held;
}
const arrowsToCosts = async (dashboard) => {
    const list = dashboard.page.querySelector('.dashboard-tabs');
    for (const row of list.querySelectorAll('[data-dashboard-tab]').slice(0, 3)) await list.fire('keydown', { target: row, key: 'ArrowDown', preventDefault() {} });
};

test('Costs starts none of its reads while the arrow keys only select it in one column’s list, and the press that opens it starts each once', async (t) => {
    const { dashboard, calls, state, open } = costsInDashboard(t);
    const base = calls.length;
    await arrowsToCosts(dashboard);
    await settle();
    assert.deepEqual([state.dashboardActiveSubtab, dashboard.page.dataset.sectionView], ['costs', 'auto'], 'the keys chose Costs on the list, not its page');
    assert.deepEqual(asked(calls.slice(base)), [0, 0, 0, 0, 0], 'a page that is not drawn asks for no totals, no 24 hours, no limits, no allowance and no subscriptions');
    await open();
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [1, 1, 1, 1, 1], 'the press that opens the page asks for each once');
    dashboard.destroy();
});

test('Costs ends every read it has out when one column’s Back hides it, so no late answer paints, and each opening reads anew', async (t) => {
    const { dashboard, layers, pending, routes, open, doors } = costsInDashboard(t);
    const held = heldReads(routes, [COSTS.totals, COSTS.period, COSTS.allowance]);
    const reads = (at) => [held[COSTS.totals][at], held[COSTS.period][at], held[COSTS.allowance][at], pending[at]];
    const painted = () => ['#cost-spent', '#costs-allowance-spent', '.costs-period-models'].map((selector) => dashboard.page.querySelector(selector).textContent).concat([doors()]);
    const back = () => layers.get('dashboard-level').onClose();
    // The first showing ends at Back: its four transports are released and its bodies come too late to paint.
    await open();
    await settle();
    const first = reads(0);
    assert.deepEqual(ended(first), [false, false, false, false], 'opened, the page has its four reads out, each with its own transport to end');
    back();
    assert.deepEqual(ended(first), [true, true, true, true], 'Back ends every one of them, not the subscriptions alone');
    first[0].resolve(totalsBody(11)); first[1].resolve(periodBody('late/first')); first[2].resolve(allowanceBody(1)); first[3].resolve(passive());
    await settle();
    assert.deepEqual(painted(), ['Loading…', 'Reading…', '', []], 'a page that was left paints no answer that comes after');
    // A second showing, ended by Back, and a third: whatever the earlier one answers, however late, the third's stands.
    await open();
    await settle();
    const second = reads(1);
    back();
    await open();
    await settle();
    const third = reads(2);
    assert.deepEqual([ended(second), ended(third)], [[true, true, true, true], [false, false, false, false]], 'each opening reads anew');
    third[0].resolve(totalsBody(7)); third[1].resolve(periodBody('current/model')); third[2].resolve(allowanceBody(2));
    third[3].resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    await settle();
    const current = painted();
    assert.deepEqual(current, [costDashboardPresentation(totalsBody(7)).spent,
        allowancePresentation(normalizeConsciousness(allowanceBody(2))).meter.spent, current[2], ['accounts']]);
    assert.match(current[2], /current\/model/);
    second[0].resolve(totalsBody(99)); second[1].resolve(periodBody('stale/model')); second[2].resolve(allowanceBody(9)); second[3].resolve(passive());
    await settle();
    assert.deepEqual(painted(), current, 'an earlier showing paints nothing over the one on screen');
    dashboard.destroy();
});

test('Costs reads each of its reads at once when two columns show it, and its visible Refresh reads again at once', async (t) => {
    const { dashboard, calls } = costsInDashboard(t, { twoPanes: true });
    const base = calls.length;
    dashboard.activateTab('costs');
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [1, 1, 1, 1, 1], 'beside its list the page asks for all five as it is shown');
    await dashboard.page.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [2, 2, 1, 2, 2], 'Refresh asks again for the totals, the 24 hours, the allowance and the subscriptions; the limits are not its');
    dashboard.destroy();
});

test('Costs ends its reads when the window narrows to one column and reads anew at once when it widens again', async (t) => {
    const observers = [];
    const before = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
    t.after(() => { globalThis.ResizeObserver = before; });
    const { dashboard, pending, routes } = costsInDashboard(t, { twoPanes: true });
    const held = heldReads(routes, [COSTS.totals, COSTS.period, COSTS.allowance]);
    const pane = dashboard.page.querySelector('.dashboard-panels');
    let box = true;
    pane.getClientRects = () => (box ? [{ width: 390 }] : []);
    const reflow = () => observers.forEach((fn) => fn([{ target: dashboard.page }]));
    const reads = (at) => [held[COSTS.totals][at], held[COSTS.period][at], held[COSTS.allowance][at], pending[at]];
    dashboard.activateTab('costs');
    await settle();
    const beside = reads(0);
    assert.deepEqual(ended(beside), [false, false, false, false], 'beside its list the page has four reads out');
    box = false;  // one column: the open page loses the pane beside the list
    reflow();
    assert.deepEqual(ended(beside), [true, true, true, true], 'the page without its box holds no read');
    box = true;
    reflow();
    await settle();
    const again = reads(1);
    assert.deepEqual(ended(again), [false, false, false, false], 'widened, it reads anew at once');
    dashboard.destroy();
});

test('Costs reads nothing for a setting saved while its page is not drawn, its own rows’ late saves included, and reads what each moves when it is', async (t) => {
    const { dashboard, calls, window, open } = costsInDashboard(t);
    const base = calls.length;
    const saved = (detail = { source: 'settings' }) => window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail }));
    const own = (row) => ({ source: 'settings-row', page: 'dashboard:costs', row });  // a save of Costs' own row that lands after Back
    await arrowsToCosts(dashboard);
    saved();
    saved(own('TOTAL_BUDGET'));
    saved(own('OUROBOROS_CONSCIOUSNESS_DAILY_USD'));
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [0, 0, 0, 0, 0], 'the list holds Costs undrawn: no save, elsewhere or its own, reads any of it');
    await open();
    await settle();
    saved();
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [2, 1, 2, 2, 1], 'drawn, a save elsewhere reads the totals, the limits and the allowance again');
    saved(own('TOTAL_BUDGET'));
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [3, 1, 2, 2, 1], 'a saved total reads the totals only');
    saved(own('OUROBOROS_CONSCIOUSNESS_DAILY_USD'));
    await settle();
    assert.deepEqual(asked(calls.slice(base)), [3, 1, 2, 3, 1], 'a saved allowance reads the allowance only');
    dashboard.destroy();
});

test('Costs reads each of its reads once when Dashboard returns with Costs beside its list after a real exit, answered or cut short, and the old paint stays until the new answers come', async (t) => {
    const show = (env, page) => {
        env.state.activePage = page;
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page } }));
    };
    const env = costsInDashboard(t, { twoPanes: true });
    const held = heldReads(env.routes, [COSTS.totals, COSTS.period, COSTS.allowance]);
    const painted = () => ['#cost-spent', '#costs-allowance-spent', '.costs-period-models'].map((selector) => env.dashboard.page.querySelector(selector).textContent).concat([env.doors()]);
    const answer = (at, spent, model) => {
        held[COSTS.totals][at].resolve(totalsBody(spent)); held[COSTS.period][at].resolve(periodBody(model));
        held[COSTS.allowance][at].resolve(allowanceBody(spent)); env.pending[at].resolve(passive({ profiles: { profiles: [] }, quota: [] }));
    };
    env.dashboard.activateTab('costs');
    await settle();
    answer(0, 11, 'old/model');
    await settle();
    const old = painted();
    assert.match(old[2], /old\/model/, 'the first showing is answered and painted');
    const base = asked(env.calls);
    // A real exit and a return, every read of the old visit answered: the return is a new visit and reads each once. The
    // return is told twice in one transition (commitPage: its page, then the tab its `enter` restores): the tab's tells
    // the visit nothing new.
    show(env, 'chat');
    show(env, 'dashboard');
    shown(env.window, 'costs');
    await settle();
    const returned = base.map((count) => count + 1);
    assert.deepEqual(asked(env.calls), returned, 'the return reads again though the last showing was answered, once, whatever tells it');
    assert.deepEqual(painted(), old, 'and the old paint stays until the new answers come');
    // A show after the transition is another: a visible page's show reads what is not out, as it always did.
    shown(env.window, 'costs');
    await settle();
    assert.deepEqual(asked(env.calls), returned.map((count, at) => count + (at === 2 ? 1 : 0)),
        'only what is not out (the answered limits) is read');
    // Refresh is never held back by the visit: it reads again at once.
    await env.dashboard.page.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(asked(env.calls), returned.map((count) => count + 1),
        'Refresh reads the totals, the 24 hours, the allowance and the subscriptions again at once (the limits’ one more is the show before)');
    answer(2, 22, 'new/model');
    await settle();
    const fresh = painted();
    assert.match(fresh[2], /new\/model/, 'the new answers replace the old paint');
    assert.deepEqual(fresh.slice(0, 2), [costDashboardPresentation(totalsBody(22)).spent, allowancePresentation(normalizeConsciousness(allowanceBody(22))).meter.spent]);
    env.dashboard.destroy();
    const cut = costsInDashboard(t, { twoPanes: true });  // its quota read never answers
    cut.dashboard.activateTab('costs');
    await settle();
    assert.deepEqual(asked(cut.calls), [1, 1, 1, 1, 1]);
    show(cut, 'chat');
    assert.equal(cut.pending[0].signal.aborted, true);
    show(cut, 'dashboard');
    await settle();
    assert.deepEqual(asked(cut.calls), [2, 2, 2, 2, 2], 'a read cut short is replaced with the others, once each, and no tab was announced');
    cut.dashboard.destroy();
});

test('Costs coalesces only the tab announcement of the transition that entered the Dashboard: one that comes later is a show of its own', async (t) => {
    const env = costsInDashboard(t, { twoPanes: true });
    const show = (page) => {
        env.state.activePage = page;
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page } }));
    };
    env.dashboard.activateTab('costs');
    await settle();
    env.pending[0].resolve(passive());
    await settle();
    const base = asked(env.calls);
    show('chat');
    show('dashboard');  // an ordinary entry into a Dashboard that shows Costs: no tab announcement follows it
    await settle();
    const entered = base.map((count) => count + 1);
    assert.deepEqual(asked(env.calls), entered, 'the entry reads each once');
    shown(env.window, 'costs');  // an announcement of the tab, later: not the pair of that entry
    await settle();
    assert.deepEqual(asked(env.calls), entered.map((count, at) => count + (at === 4 ? 0 : 1)), 'it reads what is not out (the quota read of the entry still is)');
    env.dashboard.destroy();
});

test('Costs gives back what it took when its owner destroys it: its socket, its page events, the reads it has out and its timer', async (t) => {
    forgetSettingRows();
    const ws = socket();
    const env = costsInDashboard(t, { twoPanes: true, ws });
    env.dashboard.activateTab('costs');
    await settle();
    assert.deepEqual(asked(env.calls), [1, 1, 1, 1, 1]);
    env.costs.destroy();
    assert.equal(env.pending[0].signal.aborted, true, 'the read it had out is ended');
    const base = asked(env.calls);
    ws.emit('open', { previouslyConnected: true });
    shown(env.window, 'costs');
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'dashboard' } }));
    env.window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings' } }));
    await settle();
    assert.deepEqual(asked(env.calls), base, 'nothing that reached it before reaches it now');
    env.dashboard.destroy();
});

test('the app that mounts Costs destroys it with its other pages when the window goes', () => {
    const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
    assert.ok(/const costs = initCosts\(\{ \.\.\.ctx, mount: document\.getElementById\('dashboard-panel-costs'\) \}\);/.test(app), 'the owner keeps what initCosts answers');
    assert.ok(/costs\.destroy\(\);/.test(app.slice(app.indexOf("window.addEventListener('pagehide'"))), 'and gives it back where the other pages are');
});

test('Costs does not read the allowance again at the time it gave while its page is not drawn, and does while it is', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });  // before the fixture, so the timers it keeps are these
    const { dashboard, calls, routes } = costsInDashboard(t, { twoPanes: true });
    const pane = dashboard.page.querySelector('.dashboard-panels');
    let box = true;
    pane.getClientRects = () => (box ? [{ width: 390 }] : []);
    routes.set(COSTS.allowance, () => response({ enabled: true, spent_24h_usd: 12, daily_usd: 10, allowance_status: 'exhausted',
        allowance_room_after: new Date(Date.now() + 60_000).toISOString() }));
    const allowance = () => calls.filter((call) => call === COSTS.allowance).length;
    dashboard.activateTab('costs');
    await settle();
    assert.equal(allowance(), 1);
    box = false;  // the page is not drawn when its time comes
    t.mock.timers.tick(61_001);
    await settle();
    assert.equal(allowance(), 1, 'a page that is not drawn is not read at the time it named');
    box = true;
    dashboard.activateTab('costs');
    await settle();
    assert.equal(allowance(), 2, 'shown again, it reads');
    t.mock.timers.tick(61_001);
    await settle();
    assert.equal(allowance(), 3, 'drawn at the time it named, it reads once more');
    dashboard.destroy();
});

// ---- The settings read: its signal is opt-in, and a read it ended costs no other read its place ---------------------------
// `/api/settings` answers as the test says: each read keeps its transport signal and a body to resolve (`answer.status`
// set before the body fails it). `applied` lists the TOTAL_BUDGET of every document announced as applied.
function heldSettings(t, { post = null } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const held = [];
    const applied = [];
    env.routes.set('/api/settings', (url, init = {}) => {
        if (init.method === 'POST' && post) return post(url, init);
        const body = deferred();
        const answer = { status: 200, get ok() { return this.status < 400; }, json: () => body.promise };
        held.push({ signal: init.signal, answer, ...body });
        return answer;
    });
    document.addEventListener('settings-document:applied', (event) => applied.push(event.detail.TOTAL_BUDGET));
    return { ...env, held, applied };
}

test('a settings read its own signal ended changes nothing when its answer comes: no document, no announcement, no mark of failure', async (t) => {
    const { held, applied, routes } = heldSettings(t);
    const first = readSettings();
    held[0].resolve({ TOTAL_BUDGET: 100 });
    await first;
    assert.deepEqual([settingsSnapshot().TOTAL_BUDGET, applied], [100, [100]], 'an ordinary read lands as it always did');
    const control = new AbortController();
    const endedRead = readSettings({ signal: control.signal }).then(() => 'landed', (error) => error.name);
    await settle();
    assert.equal(held[1].signal?.aborted, false, 'a read that is given a signal carries it to the transport');
    control.abort();
    assert.equal(held[1].signal?.aborted, true, 'and the end reaches the transport');
    held[1].resolve({ TOTAL_BUDGET: 999 });  // a transport that ignores the end answers anyway
    assert.equal(await endedRead, 'AbortError');
    assert.deepEqual([settingsSnapshot().TOTAL_BUDGET, applied], [100, [100]], 'the answer of an ended read puts no document back and announces none');
    forgetSettingRows();  // nothing read yet: an ended read that fails is no failure of the page to read
    const lost = new AbortController();
    const failed = readSettings({ signal: lost.signal }).then(() => 'landed', (error) => error.message);
    await settle();
    lost.abort();
    held[2].answer.status = 500;
    held[2].resolve({ error: 'offline' });
    assert.equal(await failed, 'offline');
    assert.equal(settingsReadMessage(), 'Reading…', 'a read its own signal ended marks no failure and paints none');
    const plain = readSettings().catch(() => 'failed');
    await settle();
    held[3].answer.status = 500;
    held[3].resolve({ error: 'offline' });
    await plain;
    assert.equal(settingsReadMessage(), 'Could not be read', 'the same failure of an ordinary read still does');
    // A transport that honours the end rejects at once with its own abort: that too is no failure to read.
    forgetSettingRows();
    routes.set('/api/settings', (_url, init = {}) => new Promise((_resolve, reject) => {
        const refuse = () => reject(new DOMException('aborted', 'AbortError'));
        if (init.signal.aborted) refuse(); else init.signal.addEventListener('abort', refuse);  // as fetch does
    }));
    const honoured = new AbortController();
    const rejected = readSettings({ signal: honoured.signal }).catch((error) => error.name);
    await settle();
    honoured.abort();
    assert.equal(await rejected, 'AbortError');
    assert.equal(settingsReadMessage(), 'Reading…');
    await assert.rejects(readSettings({ signal: honoured.signal }), { name: 'AbortError' }, 'a signal already ended starts no read');
});

test('a settings read its own signal ended confirms no write: a row unconfirmed stays so until a read that counts shows its value', async (t) => {
    const { held, mount } = heldSettings(t, { post: () => response({ error: 'answer lost' }, 500) });
    const row = rowById('TOTAL_BUDGET');
    const host = new NodeStub();
    mount.appendChild(host);
    host.innerHTML = settingsRow(row);
    const rows = bindSettingRows(host);
    const first = readSettings();
    held[0].resolve({ TOTAL_BUDGET: 100 });
    await first;
    const writing = commitSetting(row, 150);  // the answer is lost, and so is the read that would settle it
    await settle();
    held[1].answer.status = 500;
    held[1].resolve({ error: 'offline' });
    await writing;
    assert.equal(result(host, 'TOTAL_BUDGET'), 'Not confirmed yet');
    const control = new AbortController();
    const endedRead = readSettings({ signal: control.signal }).catch((error) => error.name);
    await settle();
    control.abort();
    held[2].resolve({ TOTAL_BUDGET: 150 });  // it shows the value that was sent
    assert.equal(await endedRead, 'AbortError');
    assert.equal(result(host, 'TOTAL_BUDGET'), 'Not confirmed yet', 'an ended read confirms nothing');
    const counted = readSettings();
    held[3].resolve({ TOTAL_BUDGET: 150 });
    await counted;
    assert.match(result(host, 'TOTAL_BUDGET'), /^Saved/, 'a read that counts does');
    rows.dispose();
});

test('a read its own signal can end takes its place in the order only when it lands: ended it costs the reads before it nothing, first it stands over them, overtaken it lands nothing', async (t) => {
    const { held, applied } = heldSettings(t, { post: () => response({ status: 'saved', changes: { TOTAL_BUDGET: { old: 6, new: 7, token: 'TOTAL_BUDGET@1' } } }) });
    const answer = (index, spent) => held[index].resolve({ TOTAL_BUDGET: spent });
    const start = (signal) => readSettings(signal ? { signal } : undefined).then((data) => data.TOTAL_BUDGET, (error) => error.name);
    const base = start();
    answer(0, 1);
    await base;
    // An ordinary read is out, Costs reads after it and is ended: the ordinary answer still lands over the snapshot.
    const ordinary = start();
    const visit = new AbortController();
    const endedRead = start(visit.signal);
    await settle();
    visit.abort();
    answer(1, 2);
    await ordinary;
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 2, 'the ordinary read lands: an ended read took no place in the order');
    answer(2, 99);
    assert.equal(await endedRead, 'AbortError');
    assert.deepEqual([settingsSnapshot().TOTAL_BUDGET, applied], [2, [1, 2]], 'the ended read changed and announced nothing');
    // An ordinary read is out, Costs reads after it and lands first: its document stands, and the older answer does not undo it.
    const older = start();
    const first = start(new AbortController().signal);
    await settle();
    answer(4, 4);
    assert.equal(await first, 4);
    answer(3, 3);
    await older;
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 4, 'the older ordinary answer, after the one that landed first, puts nothing over it');
    // The ordinary answer first, then Costs': each lands in the order it answered, the newer stands.
    const early = start();
    const late = start(new AbortController().signal);
    await settle();
    answer(5, 5);
    await early;
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 5);
    answer(6, 6);
    assert.equal(await late, 6);
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 6);
    // Costs reads, an ordinary read begins after it: Costs' answer is overtaken and lands nothing.
    const overtaken = start(new AbortController().signal);
    const newer = start();
    await settle();
    answer(7, 7);
    assert.equal(await overtaken, 'AbortError', 'a read a newer one overtook is no answer');
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 6);
    answer(8, 8);
    await newer;
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 8);
    // Costs reads, a write lands after it: the write is the newer authority, Costs' answer lands nothing.
    const beforeWrite = start(new AbortController().signal);
    await settle();
    await commitSetting(rowById('TOTAL_BUDGET'), 7);
    answer(9, 9);
    assert.equal(await beforeWrite, 'AbortError');
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 7, 'the write stands');
});

test('a read its own signal can end clears no standing failure: the pages left with "Could not be read" keep it until an answer, however the retry ends', async (t) => {
    const { held } = heldSettings(t);
    const plain = readSettings().catch(() => 'failed');
    held[0].answer.status = 500;
    held[0].resolve({ error: 'offline' });
    await plain;
    assert.equal(settingsReadMessage(), 'Could not be read', 'nothing was read and the newest read failed');
    const control = new AbortController();
    const retry = readSettings({ signal: control.signal }).catch((error) => error.name);
    await settle();
    assert.equal(settingsReadMessage(), 'Could not be read', 'a retry that may end clears the failure only by an answer');
    control.abort();
    held[1].resolve({ TOTAL_BUDGET: 5 });  // an answer that came after the end
    assert.equal(await retry, 'AbortError');
    assert.deepEqual([settingsReadMessage(), settingsSnapshot()], ['Could not be read', null], 'ended, it leaves the failure standing, and no page waits on a reader that is gone');
    const answered = readSettings({ signal: new AbortController().signal });
    held[2].resolve({ TOTAL_BUDGET: 6 });
    await answered;
    assert.deepEqual([settingsReadMessage(), settingsSnapshot().TOTAL_BUDGET], ['', 6], 'an answer clears it');
});

test('Costs’ retry after a failed read leaves that failure standing when one column’s Back ends it', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t);
    const held = [];
    let reads = 0;
    env.routes.set(COSTS.settings, (_url, init = {}) => {
        reads += 1;
        if (reads === 1) return response({ error: 'offline' }, 500);  // a Settings page's read: it fails with nothing read
        const body = deferred();
        held.push({ signal: init.signal, ...body });
        return { ok: true, status: 200, json: () => body.promise };
    });
    await readSettings().catch(() => {});
    assert.equal(settingsReadMessage(), 'Could not be read');
    await env.open();  // Costs reads again, a retry that its Back may end
    await settle();
    assert.equal(held.length, 1);
    assert.equal(settingsReadMessage(), 'Could not be read', 'the retry clears the failure only by an answer');
    env.layers.get('dashboard-level').onClose();
    assert.equal(held[0].signal.aborted, true);
    assert.equal(settingsReadMessage(), 'Could not be read', 'Back leaves the other pages their failure, not a "Reading…" with no reader');
    env.dashboard.destroy();
});

// A settings document as GET /api/settings answers it, to the fields Costs' limits card reads: the budget, what its limits
// cannot count (`known_spend_gaps`, `partly_counted_spend`) and the setup contract's note of how its limits count.
const limitsDoc = (budget, { gaps = [], partly = [], note = '' } = {}) => ({ TOTAL_BUDGET: budget,
    _meta: { known_spend_gaps: gaps, partly_counted_spend: partly, setup_contract: { budgetFields: note ? [{ note }] : [] } } });
// What the limits card shows of such a document: its Total budget field and, under the rows, the line of what is not
// counted and the note of how the limits count.
const limitsCard = (page) => ({
    total: () => page.querySelector('#setting-TOTAL_BUDGET').value,
    gaps: () => page.querySelector('#costs-gaps').textContent,
    help: () => page.querySelector('#costs-budget-help').textContent,
});

test('the limits’ notes are the accepted document’s, as the rows are: a read a newer read overtook paints neither, the newer document leads both', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t, { twoPanes: true });
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    const { total, gaps, help } = limitsCard(env.dashboard.page);
    const base = readSettings();  // a snapshot exists: held[0]
    held[0].resolve(limitsDoc(50, { note: 'Base note' }));
    await base;
    env.dashboard.activateTab('costs');  // Costs reads: held[1]
    await settle();
    const newer = readSettings();  // a Settings page reads after it: held[2]
    await settle();
    held[1].resolve(limitsDoc(111, { gaps: ['Stale tool'], note: 'Stale note' }));
    await settle();
    assert.deepEqual([gaps(), help(), total(), settingsSnapshot().TOTAL_BUDGET], ['', 'Base note', '50', 50],
        'a read the newer one overtook paints no note and puts nothing back');
    held[2].resolve(limitsDoc(60, { gaps: ['Current tool'], note: 'Current note' }));
    await newer;
    assert.deepEqual([total(), settingsSnapshot().TOTAL_BUDGET, gaps(), help()], ['60', 60, spendGapsWords(['Current tool']), 'Current note'],
        'the newer read’s document stands in the rows and, as one, in the notes under them');
    assert.equal(held.length, 3, 'no read was made for the notes: the three above are all');
    env.dashboard.destroy();
});

test('concurrent settings reads, in either order, leave the rows and the notes on the one document that was accepted', async (t) => {
    for (const order of ['the newer lands first', 'the older lands first']) {
        forgetSettingRows();
        const env = costsInDashboard(t, { twoPanes: true });
        const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
        const { total, gaps, help } = limitsCard(env.dashboard.page);
        env.dashboard.activateTab('costs');  // Costs reads first: held[0]
        await settle();
        const other = readSettings();  // a Settings page reads after it: held[1]
        await settle();
        const [older, newer] = [() => held[0].resolve(limitsDoc(111, { gaps: ['Older tool'], note: 'Older note' })),
            () => held[1].resolve(limitsDoc(222, { gaps: ['Newer tool'], note: 'Newer note' }))];
        for (const answer of order === 'the newer lands first' ? [newer, older] : [older, newer]) {
            answer();
            await settle();
        }
        await other;
        assert.deepEqual([total(), settingsSnapshot().TOTAL_BUDGET, gaps(), help()], ['222', 222, spendGapsWords(['Newer tool']), 'Newer note'], order);
        env.dashboard.destroy();
    }
});

test('a limit saved while Costs reads stays saved: the older document that comes after puts nothing back, and a later opening reads the notes anew', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t, { twoPanes: true });
    let server = limitsDoc(50, { gaps: ['Tool A'], note: 'Note A' });
    const held = [];
    env.routes.set(COSTS.settings, (_url, init = {}) => {
        if (init.method === 'POST') {
            const body = JSON.parse(init.body);
            server = { ...server, ...body };
            return response({ status: 'saved', changes: { TOTAL_BUDGET: { old: 50, new: body.TOTAL_BUDGET, token: 'TOTAL_BUDGET@1' } } });
        }
        const body = deferred();
        held.push({ signal: init.signal, ...body, asked: structuredClone(server) });  // the document as it stood when asked
        return { ok: true, status: 200, json: () => body.promise };
    });
    const { total, gaps, help } = limitsCard(env.dashboard.page);
    const base = readSettings();
    held[0].resolve(held[0].asked);
    await base;
    env.dashboard.activateTab('costs');  // Costs reads the document of 50: held[1]
    await settle();
    const saved = await commitSetting(rowById('TOTAL_BUDGET'), 120);
    assert.equal(saved.state, 'saved');
    held[1].resolve(held[1].asked);  // the older document comes after the write
    await settle();
    assert.deepEqual([total(), settingsSnapshot().TOTAL_BUDGET, gaps(), help()], ['120', 120, spendGapsWords(['Tool A']), 'Note A'],
        'the saved value stands, and the notes are still those of the accepted document');
    server = { ...server, _meta: { ...server._meta, known_spend_gaps: ['Tool B'] } };  // the server's own accounting moved on
    env.dashboard.activateTab('logs');
    env.dashboard.activateTab('costs');  // a later opening reads anew: held[2]
    await settle();
    held[2].resolve(held[2].asked);
    await settle();
    assert.deepEqual([total(), gaps()], ['120', spendGapsWords(['Tool B'])], 'the saved value is the server’s too, and the notes follow the new read');
    env.dashboard.destroy();
});

test('Costs hidden reads nothing for its notes and keeps what it showed; opened again it reads once, and the notes follow that read', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t, { twoPanes: true });
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    const { gaps, help } = limitsCard(env.dashboard.page);
    env.dashboard.activateTab('costs');
    await settle();
    held[0].resolve(limitsDoc(1, { gaps: ['First tool'], note: 'First note' }));
    await settle();
    assert.deepEqual([gaps(), help()], [spendGapsWords(['First tool']), 'First note']);
    env.dashboard.activateTab('logs');  // Costs is hidden
    await settle();
    assert.deepEqual([held.length, gaps(), help()], [1, spendGapsWords(['First tool']), 'First note'], 'hiding asks for nothing and clears nothing');
    const other = readSettings();  // a Settings page reads while Costs is hidden: held[1]
    await settle();
    held[1].resolve(limitsDoc(2, { gaps: ['Second tool'], note: 'Second note' }));
    await other;
    assert.deepEqual([held.length, gaps(), help()], [2, spendGapsWords(['Second tool']), 'Second note'], 'its rows and its notes alike follow the accepted document, with no read of its own');
    env.dashboard.activateTab('costs');  // opened again: held[2]
    await settle();
    assert.equal(held.length, 3, 'one read for the opening');
    held[2].resolve(limitsDoc(3, { gaps: ['Third tool'], note: 'Third note' }));
    await settle();
    assert.deepEqual([gaps(), help()], [spendGapsWords(['Third tool']), 'Third note']);
    env.dashboard.destroy();
});

test('Costs’ settings read is its page’s: Back ends it, no body that comes after puts a document back or paints a row, and each opening reads anew', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t);
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    const applied = [];
    document.addEventListener('settings-document:applied', (event) => applied.push(event.detail.TOTAL_BUDGET));
    const total = () => env.dashboard.page.querySelector('#setting-TOTAL_BUDGET').value;
    const back = () => env.layers.get('dashboard-level').onClose();
    const unread = total();
    await env.open();
    await settle();
    assert.deepEqual(ended([held[0]]), [false], 'opened, the page has its limits read out, with a transport to end');
    back();
    assert.deepEqual(ended([held[0]]), [true], 'Back ends it with the others');
    held[0].resolve({ TOTAL_BUDGET: 111 });
    await settle();
    assert.deepEqual([settingsSnapshot(), applied, total()], [null, [], unread], 'its late answer puts no document back and paints no row');
    // A second showing, ended by Back, and a third: the second's answer, however late, changes nothing over the third's.
    await env.open();
    await settle();
    back();
    await env.open();
    await settle();
    assert.deepEqual(ended([held[1], held[2]]), [true, false], 'each opening reads anew');
    held[2].resolve({ TOTAL_BUDGET: 333 });
    await settle();
    assert.deepEqual([settingsSnapshot().TOTAL_BUDGET, applied, total()], [333, [333], '333']);
    held[1].resolve({ TOTAL_BUDGET: 222 });
    await settle();
    assert.deepEqual([settingsSnapshot().TOTAL_BUDGET, applied, total()], [333, [333], '333'], 'an earlier showing’s answer changes nothing');
    env.dashboard.destroy();
});

// One socket for the Dashboard and Costs both, as the app has: the fixture's keeps the last listener of a kind only.
function socket() {
    const listeners = new Map();
    return {
        on(type, fn) {
            listeners.set(type, [...(listeners.get(type) || []), fn]);
            return () => listeners.set(type, (listeners.get(type) || []).filter((each) => each !== fn));
        },
        emit(type, data = {}) { (listeners.get(type) || []).forEach((fn) => fn(data)); },
    };
}

test('a reconnect starts no settings read for Costs while one column’s list holds it undrawn; drawn, Costs reads its limits once, as a read its Back ends', async (t) => {
    forgetSettingRows();
    const ws = socket();
    const env = costsInDashboard(t, { ws });
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    await arrowsToCosts(env.dashboard);
    await settle();
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(held.length, 0, 'a reconnect reads nothing for a page that the list holds undrawn');
    ws.emit('open', { previouslyConnected: false });  // the first connect is no reconnect
    await settle();
    assert.equal(held.length, 0);
    await env.open();
    await settle();
    assert.equal(held.length, 1, 'the press that opens the page reads its limits once');
    held[0].resolve({ TOTAL_BUDGET: 1 });
    await settle();
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(held.length, 2, 'drawn, a reconnect reads the limits once: Costs’ own read, and the Dashboard adds none beside it');
    assert.deepEqual(ended([held[1]]), [false], 'a read of the page, with a transport its Back can end');
    env.layers.get('dashboard-level').onClose();
    assert.deepEqual(ended([held[1]]), [true]);
    held[1].resolve({ TOTAL_BUDGET: 2 });
    await settle();
    assert.equal(settingsSnapshot().TOTAL_BUDGET, 1, 'and its late answer puts no document back');
    env.dashboard.destroy();
});

test('a reconnect still reads once for a neighbouring tab the Dashboard shows, and once for Costs beside its list in two columns', async (t) => {
    forgetSettingRows();
    const ws = socket();
    const env = costsInDashboard(t, { twoPanes: true, ws });
    const reads = () => env.calls.filter((call) => call === COSTS.settings).length;
    env.dashboard.activateTab('consciousness');
    await settle();
    const base = reads();
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(reads(), base + 1, 'Consciousness, drawn beside the list, is read for as the Dashboard always did');
    env.dashboard.activateTab('costs');
    await settle();
    const costs = reads();
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(reads(), costs + 1, 'Costs, drawn beside the list, is read for once');
    env.dashboard.destroy();
});

test('the notes are painted inside every settings read, so a document with malformed notes breaks neither that read nor the rows', async (t) => {
    forgetSettingRows();
    const env = costsInDashboard(t, { twoPanes: true });
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    const { total, gaps, help } = limitsCard(env.dashboard.page);
    for (const [meta, shown] of [[{ known_spend_gaps: 'a string', partly_counted_spend: { not: 'a list' }, setup_contract: { budgetFields: 'bad' } }, 7],
        [{ known_spend_gaps: [null, 3], partly_counted_spend: [null, 5], setup_contract: { budgetFields: [null, {}, { note: 42 }] } }, 8], [null, 9]]) {
        const read = readSettings();  // an ordinary read, as any page makes it
        await settle();
        held.at(-1).resolve({ TOTAL_BUDGET: shown, _meta: meta });
        assert.equal(await read.then(() => 'landed', (error) => `rejected: ${error.message}`), 'landed');
        assert.deepEqual([total(), settingsSnapshot().TOTAL_BUDGET], [String(shown), shown], 'the rows hold the document');
        assert.equal(typeof gaps(), 'string');
        assert.ok(help().length > 0, 'the note of how the limits count is never left empty');
    }
    env.dashboard.destroy();
});

test('a visible reconnect replaces a limits read that is still out: the old answer lands nothing, the new one paints rows and notes, whichever answers first', async (t) => {
    for (const order of ['the old answers first', 'the new answers first']) {
        forgetSettingRows();
        const ws = socket();
        const env = costsInDashboard(t, { twoPanes: true, ws });
        const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
        const { total, gaps, help } = limitsCard(env.dashboard.page);
        env.dashboard.activateTab('costs');  // Costs reads: held[0], asked on the connection that is about to drop
        await settle();
        assert.deepEqual(ended(held), [false], `${order}: one read is out`);
        ws.emit('open', { previouslyConnected: true });
        await settle();
        assert.equal(held.length, 2, `${order}: the reconnect asks again though the old read is still out`);
        assert.deepEqual(ended(held), [true, false], `${order}: the new read replaces the old, whose transport is ended`);
        const [old, now] = [() => held[0].resolve(limitsDoc(111, { gaps: ['Old tool'], note: 'Old note' })),
            () => held[1].resolve(limitsDoc(222, { gaps: ['New tool'], note: 'New note' }))];
        const answers = order === 'the old answers first' ? [old, now] : [now, old];
        answers[0]();
        await settle();
        if (answers[0] === old) {
            assert.deepEqual([settingsSnapshot(), gaps()], [null, ''], 'the ended read lands no document and paints no note');
        } else {
            assert.deepEqual([total(), gaps(), help()], ['222', spendGapsWords(['New tool']), 'New note'], 'the new answer paints rows and notes');
        }
        answers[1]();
        await settle();
        assert.deepEqual([total(), settingsSnapshot().TOTAL_BUDGET, gaps(), help()], ['222', 222, spendGapsWords(['New tool']), 'New note'],
            `${order}: what stands is the new read's document, in the rows and under them`);
        assert.equal(held.length, 2, 'and no other read was made');
        env.dashboard.destroy();
    }
});

test('a reconnect asks nothing for a Costs the Dashboard does not draw, and nothing once Costs is destroyed', async (t) => {
    forgetSettingRows();
    const ws = socket();
    const env = costsInDashboard(t, { twoPanes: true, ws });
    const held = heldReads(env.routes, [COSTS.settings])[COSTS.settings];
    env.dashboard.activateTab('costs');
    await settle();
    assert.deepEqual(ended(held), [false]);
    env.dashboard.activateTab('logs');  // Costs is hidden: the read it had out is ended with it
    await settle();
    assert.deepEqual(ended(held), [true]);
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(held.length, 1, 'a hidden Costs reads nothing on a reconnect');
    env.dashboard.activateTab('costs');
    await settle();
    assert.deepEqual([held.length, ended(held)], [2, [true, false]], 'opened, it reads once');
    ws.emit('open', { previouslyConnected: true });  // a read of the opening is out: the reconnect replaces it
    await settle();
    assert.deepEqual([held.length, ended(held)], [3, [true, true, false]]);
    env.costs.destroy();
    assert.deepEqual(ended(held), [true, true, true], 'destroyed, it ends what it has out');
    ws.emit('open', { previouslyConnected: true });
    await settle();
    assert.equal(held.length, 3, 'and a reconnect reaches it no more');
    env.dashboard.destroy();
});

test('in a wide window the Dashboard holds its list beside a page; an engine without container queries gets the list, then a page with Back', (t) => {
    // The sheets draw the two panes inside a container query. Where the engine has none, the module must not treat
    // the page as standing beside the list: it would hide Back and the list would never be seen.
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    NodeStub.prototype.getBoundingClientRect = () => ({ width: 1200 });
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    const start = (queries) => {
        window.CSS = { supports: () => queries };
        const layers = [];
        const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
        const dashboard = initDashboard({ state, openSettings: () => {},
            layers: { push: (layer) => layers.push(layer), remove: (id) => { layers.splice(0, layers.length, ...layers.filter((layer) => layer.id !== id)); } } });
        return { dashboard, layers };
    };
    const wide = start(true);
    assert.equal(wide.dashboard.page.dataset.sectionView, 'auto', 'two panes: a page stands beside the list at once');
    assert.equal(wide.dashboard.page.querySelector('.app-page-header').hasAttribute('data-controls'), false, 'a pane named by its row puts nothing in the band');
    assert.equal(wide.dashboard.page.querySelector('[data-dashboard-lead-name]').tagName.toLowerCase(), 'h2', 'the pane keeps its accessible heading');
    wide.dashboard.activateTab('costs');
    assert.deepEqual(wide.layers, [], 'and is no layer over it');
    wide.dashboard.destroy();
    const old = start(false);
    assert.equal(old.dashboard.page.dataset.sectionView, 'auto', 'one column: the list first');
    old.dashboard.activateTab('costs');
    assert.equal(old.dashboard.page.dataset.sectionView, 'page');
    assert.equal(old.layers[0].back.label, 'Dashboard', 'the page that replaced the list leads back to it');
    assert.equal(old.layers.length, 1, 'as a level a system Back closes');
    old.dashboard.destroy();
});

test('reopening the selected Dashboard row announces its page after the one-column list gives way', async (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    window.CSS = { supports: () => false };
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    const dashboard = initDashboard({ state });
    const shown = [];
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => shown.push([event.detail.tab, dashboard.page.dataset.sectionView]));
    const list = dashboard.page.querySelector('.dashboard-tabs');
    const row = list.querySelector('[data-dashboard-tab="logs"]');
    await list.fire('click', { target: row, preventDefault() {} });
    dashboard.pressedAgain();
    await list.fire('click', { target: row, preventDefault() {} });
    assert.deepEqual(shown, [['logs', 'page'], ['logs', 'page']], 'each reopening restores the same selected reader while visible');
    dashboard.destroy();
});

test('a root font that moves the two panes\' 40rem repaints the Dashboard: the pane and the list are watched beside its box', (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    NodeStub.prototype.getBoundingClientRect = () => ({ width: 735 });  // the owner's window: the place's box
    t.after(() => { delete NodeStub.prototype.getBoundingClientRect; });
    window.CSS = { supports: () => true };
    let font = 16;
    window.getComputedStyle = () => ({ fontSize: `${font}px` });
    const observers = [];
    const before = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(fn) { Object.assign(this, { fn, seen: [] }); observers.push(this); } observe(box) { this.seen.push(box); } disconnect() {} };
    t.after(() => { globalThis.ResizeObserver = before; });
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    const stack = new Map();
    const dashboard = initDashboard({ state, layers: { push: layer => stack.set(layer.id, layer), remove: id => stack.delete(id) } });
    const said = () => [dashboard.page.dataset.sectionView, [...stack.keys()], stack.get('dashboard-level')?.back.label];
    assert.equal(dashboard.page.dataset.sectionView, 'auto', '735 px hold the list beside a page at 16 px');
    dashboard.activateTab('costs');
    assert.deepEqual(said(), ['page', [], undefined], 'a page chosen beside the list has no Back: the list is the way to it');
    // The browser's font grows to 20 px: 40rem is 800 px, so the sheets draw one column while the box keeps its width.
    // The pane, which loses the list beside it, says so: the page now stands alone and gains the round Back to the
    // list, instead of a page with no way to it. Only the pane's observer can say it: the place's box kept its width.
    font = 20;
    const pane = dashboard.page.querySelector('.dashboard-panels');
    const heard = (box) => observers.find((observer) => observer.seen.includes(box));
    assert.ok(heard(pane), 'the pane is watched');
    assert.deepEqual(said(), ['page', [], undefined], 'nothing moves before the pane is heard');
    heard(pane).fn([{ target: pane }]);
    assert.deepEqual(said(), ['page', ['dashboard-level'], 'Dashboard'], 'one column: the page that replaced the list leads back to it');
    // Back to 16 px: two panes again. The list comes back beside the page as its rail, and its observer says so.
    font = 16;
    const list = dashboard.page.querySelector('.dashboard-tabs');
    assert.ok(heard(list), 'the list is watched');
    heard(list).fn([{ target: list }]);
    assert.deepEqual(said(), ['page', [], undefined], 'beside the list again, without Back');
    dashboard.destroy();
});

test('a page that gains its box as one column widens to two is announced shown, once, so Logs draws', (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.append(content);
    window.CSS = { supports: () => true };
    let width = 390;
    const before = { box: NodeStub.prototype.getBoundingClientRect, observer: globalThis.ResizeObserver };
    NodeStub.prototype.getBoundingClientRect = () => ({ width });
    const observers = [];
    globalThis.ResizeObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
    t.after(() => { NodeStub.prototype.getBoundingClientRect = before.box; globalThis.ResizeObserver = before.observer; });
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
    const dashboard = initDashboard({ state });
    // The sheets: one column's list hides the pane; from 40rem the pane stands beside it (ui.css @container sectionplace).
    dashboard.page.querySelector('.dashboard-panels').getClientRects = () => (width >= 640 ? [{ width }] : []);
    const shown = [];
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => shown.push(event.detail.tab));
    const resize = (value) => { width = value; observers.forEach((fn) => fn([{ target: dashboard.page }])); };
    try {
        resize(390);
        assert.deepEqual(shown, [], 'the list alone shows no page');
        resize(735);
        assert.deepEqual(shown, ['logs'], 'widened (900 -> 1087 px): the automatic page beside the list is shown');
        resize(800);
        assert.deepEqual(shown, ['logs'], 'a page already shown is not announced again');
        state.activePage = 'chat';
        resize(390); resize(735);
        assert.deepEqual(shown, ['logs'], 'another page in front: the Dashboard shows nothing');
    } finally { dashboard.destroy(); }
});

test('a page explicitly chosen beside the Dashboard list survives narrowing, while the automatic landing returns to the list', async (t) => {
    for (const gesture of ['same row', 'another row', 'keyboard']) await t.test(gesture, async (sub) => {
        const { mount, window, document } = setup(sub);
        const content = new NodeStub(); content.id = 'content'; mount.append(content);
        window.CSS = { supports: () => true };
        let width = 735;
        const before = { box: NodeStub.prototype.getBoundingClientRect, observer: globalThis.ResizeObserver };
        NodeStub.prototype.getBoundingClientRect = () => ({ width });
        const observers = [];
        globalThis.ResizeObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
        sub.after(() => { NodeStub.prototype.getBoundingClientRect = before.box; globalThis.ResizeObserver = before.observer; });
        const layers = new Map();
        const state = { activePage: 'dashboard', dashboardActiveSubtab: 'logs' };
        const dashboard = initDashboard({ state, layers: { push: layer => layers.set(layer.id, layer), remove: id => layers.delete(id) } });
        try {
            const list = dashboard.page.querySelector('.dashboard-tabs');
            const rows = list.querySelectorAll('[data-dashboard-tab]');
            const resize = value => { width = value; observers.forEach(fn => fn([{ target: dashboard.page }])); };
            resize(390);
            assert.deepEqual([dashboard.page.dataset.sectionView, layers.size], ['auto', 0], 'automatic landing has not chosen a page');
            // In one column the arrow keys move along the list; only a press (or the keyboard beside it) chooses a page.
            await list.fire('keydown', { target: rows[0], key: 'ArrowDown', preventDefault() {} });
            assert.deepEqual([state.dashboardActiveSubtab, dashboard.page.dataset.sectionView, layers.size], [rows[1].dataset.dashboardTab, 'auto', 0]);
            await list.fire('keydown', { target: rows[1], key: 'ArrowUp', preventDefault() {} });  // the press below is on the selected row
            assert.equal(state.dashboardActiveSubtab, rows[0].dataset.dashboardTab);
            resize(735);
            const chosen = gesture === 'same row' ? rows[0] : gesture === 'another row' ? rows[5] : rows[6];
            if (gesture === 'keyboard') await list.fire('keydown', { target: rows[0], key: 'End', preventDefault() {} });
            else await list.fire('click', { target: chosen, preventDefault() {} });
            assert.equal(state.dashboardActiveSubtab, chosen.dataset.dashboardTab);
            assert.equal(layers.size, 0);
            const control = new NodeStub('button'); dashboard.page.querySelector('.dashboard-panels').append(control); control.focus();
            resize(390);
            assert.deepEqual([dashboard.page.dataset.sectionView, [...layers.keys()]], ['page', ['dashboard-level']]);
            assert.equal(document.activeElement, control, 'resizing keeps the focused page control');
            resize(735);
            assert.deepEqual([dashboard.page.dataset.sectionView, layers.size], ['page', 0]);
            resize(390);
            layers.get('dashboard-level').onClose();
            assert.deepEqual([dashboard.page.dataset.sectionView, layers.size], ['auto', 0]);
            assert.equal(document.activeElement, chosen, 'Back returns to the chosen row');
        } finally { dashboard.destroy(); }
    });
});

test('a Dashboard deep link keeps one Back entry when it replaces the phone drawer', async () => {
    const { createLayerStack } = await import('../modules/layers.js');
    const { runInNewContext } = await import('node:vm');
    const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
    const door = app.slice(app.indexOf('async function openDashboardTab('), app.indexOf('// A line elsewhere'));
    for (const drawer of [false, true]) for (const allowed of [false, true]) {
        const historyCalls = [], shown = [], opened = [];
        const layers = createLayerStack({ win: null, doc: null, history: {
            pushState: () => historyCalls.push('push'), back: () => historyCalls.push('back'),
        } });
        try {
            if (drawer) layers.push({ id: 'drawer', focus: false });
            await Promise.resolve();
            const open = runInNewContext(`${door}\nopenDashboardTab`, { layers,
                showPage: async (name, options = {}) => {
                    shown.push(name);
                    if (!allowed) return false;
                    layers.batch(() => {
                        if (options.closeDrawer !== false) layers.remove('drawer');
                        options.enter?.();
                    });
                    return true;
                },
                dashboardControls: { activateTab: (name, options) => {
                    opened.push([name, options]); layers.push({ id: 'dashboard-level', page: true, focus: false });
                } },
                setMobileDrawerOpen: value => { assert.equal(value, false); layers.remove('drawer'); },
            });
            const options = { row: 'updates' };
            await open('updates', options);
            await Promise.resolve();
            assert.deepEqual(shown, ['dashboard']);
            assert.deepEqual(opened, allowed ? [['updates', options]] : []);
            assert.deepEqual(historyCalls, allowed || drawer ? ['push'] : [],
                'the handover never consumes the entry it will still need');
            if (allowed) {
                layers.remove('dashboard-level');
                await Promise.resolve();
                assert.deepEqual(historyCalls, ['push', 'back'], 'Back consumes exactly the one layer entry');
            }
        } finally { layers.destroy(); }
    }
});

test('a tab drawn before any read is no value; a read made elsewhere paints its rows and its own choice by its painter, before it is shown', async (t) => {
    const server = settingsServer(t, { OUROBOROS_UPDATE_CHANNEL: 'qa' });
    const host = new NodeStub();
    server.mount.appendChild(host);
    // Drawn at the app's start, as every tab is; Settings is shown, so the tab reads nothing itself.
    const state = { activePage: 'settings', dashboardActiveSubtab: 'logs' };
    const row = rowById('OUROBOROS_UPDATE_CHANNEL');
    const mounted = mountTabSettings(host, 'updates', { state, cards: [{ label: 'Which updates', solo: true }],
        onPaint: (settings) => host.querySelectorAll('[data-setting-value]').forEach((choice) =>
            choice.setAttribute('aria-pressed', String(choice.dataset.settingValue === String(settings?.OUROBOROS_UPDATE_CHANNEL ?? '')))),
        custom: () => row.control.options.map((option) => `<button data-setting-value="${option.value}" aria-pressed="false">${option.label}</button>`).join('') });
    const choices = host.querySelectorAll('[data-setting-value]');
    const facts = () => [choices.filter((choice) => choice.getAttribute('aria-pressed') === 'true').map((choice) => choice.dataset.settingValue),
        choices.map((choice) => choice.disabled), rowNode(host, 'OUROBOROS_UPDATE_CHANNEL').querySelector('[data-setting-meta]').textContent];
    assert.deepEqual(facts(), [[], [true, true, true], 'Reading…']);
    assert.deepEqual(server.calls, [], 'a tab not shown reads nothing');
    await readSettings();  // Settings read it for one of its pages
    assert.deepEqual(facts(), [['qa'], [false, false, false], '']);
    // A row of another page saved: the tab repaints from the last read, which it never made itself.
    server.window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { source: 'settings-row', page: 'models', row: 'OUROBOROS_MODEL' } }));
    assert.deepEqual(facts(), [['qa'], [false, false, false], '']);
    mounted.dispose();
});

test('a shown Dashboard tab recovers its failed first read on reconnect; first open, hidden tabs and disposal read nothing', async (t) => {
    const server = settingsServer(t, { OUROBOROS_UPDATE_CHANNEL: 'qa' });
    const content = new NodeStub(); content.id = 'content'; server.mount.appendChild(content);
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'updates' };
    const listeners = new Set();
    const ws = { on: (type, fn) => { assert.equal(type, 'open'); listeners.add(fn); return () => listeners.delete(fn); } };
    const reconnect = async (previouslyConnected = true) => { listeners.forEach((fn) => fn({ previouslyConnected })); await settle(); };
    const dashboard = initDashboard({ state, ws });
    const host = dashboard.page.querySelector('#dashboard-panel-updates');
    const original = server.routes.get('/api/settings');
    server.routes.set('/api/settings', () => response({ error: 'offline' }, 500));
    const settings = mountTabSettings(host, 'updates', { state, custom: (row) => row.control.options.map((option) =>
        `<button data-setting-value="${option.value}">${option.label}</button>`).join('') });
    await settle();
    const meta = host.querySelector('[data-setting-meta]');
    const choices = host.querySelectorAll('[data-setting-value]');
    assert.equal(meta.textContent, 'Could not be read');
    assert.ok(choices.every((button) => button.disabled));
    const reads = () => server.calls.filter((url) => url === '/api/settings').length;
    const failed = reads();
    server.routes.set('/api/settings', original);
    await reconnect(false);
    state.activePage = 'settings'; await reconnect();
    state.activePage = 'dashboard'; document.hidden = true; await reconnect();
    assert.equal(reads(), failed, 'no read at first open or while off screen');
    document.hidden = false;
    await reconnect();
    assert.equal(reads(), failed + 1);
    assert.equal(meta.textContent, '');
    assert.ok(choices.every((button) => !button.disabled), 'the shown tab recovered without reopening');
    state.dashboardActiveSubtab = 'logs'; await reconnect();
    assert.equal(reads(), failed + 1, 'a tab with no settings needs no read');
    dashboard.destroy(); settings.dispose();
    state.dashboardActiveSubtab = 'updates'; await reconnect();
    assert.equal(reads(), failed + 1, 'disposal releases the socket subscription');
});
