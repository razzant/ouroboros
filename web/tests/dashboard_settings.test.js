// The Dashboard tabs' own settings (dashboard.js mountTabSettings): rows of settings_rows.js that save themselves, drawn
// once outside every region a tab repaints, in the cards the approved picture gives them; the way back to Settings
// (settingsReturn); Costs' subscriptions from the passive quota read. The fake server below answers as the server does:
// the receipt (`changes`), `_meta.effects`, a token per key, a refused `if_unchanged` and `unset`.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { AUTONOMY_LEVELS } from '../modules/consciousness_autonomy.js';
import { cardWords, initConsciousnessDashboard, waitWords } from '../modules/consciousness_dashboard.js';
import { initCosts, subscriptionDoors } from '../modules/costs.js';
import { focusSettingRow, initDashboard, mountTabSettings, settingsReturn } from '../modules/dashboard.js';
import { SAFETY_LEVELS } from '../modules/consciousness_words.js';
import { initEvolution, whenMode, whenName, whenWords } from '../modules/evolution.js';
import { CODECS, GUARDED_WRITERS, HIDDEN_WHEN, bindSettingRows, commitSetting, forgetSettingRows, outcomeLine, readSettings, settingsRow } from '../modules/settings_controls.js';
import { ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { initStorage } from '../modules/storage.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
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
        'consciousness-allowance': '$20', 'consciousness-model': 'Main model',
        'consciousness-safety': SAFETY_LEVELS.find((level) => level.value === 'light').label, 'consciousness-evolution': 'Every 3 tasks' });
    assert.deepEqual(words({ OUROBOROS_BG_WAKEUP_MIN: 900, OUROBOROS_BG_WAKEUP_MAX: 900, OUROBOROS_CONSCIOUSNESS_DAILY_USD: '12.5',
        OUROBOROS_MODEL_CONSCIOUSNESS: 'anthropic::claude-test', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'off',
        OUROBOROS_POST_TASK_EVOLUTION: false, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:3' }), {
        OUROBOROS_CONSCIOUSNESS_AUTONOMY: '', OUROBOROS_BG_WAKEUP: 'It wakes every 15 minutes.',
        'consciousness-allowance': '$12.50', 'consciousness-model': 'claude-test', 'consciousness-safety': 'Off', 'consciousness-evolution': 'Off' });
    assert.deepEqual(words({}), { OUROBOROS_CONSCIOUSNESS_AUTONOMY: '', OUROBOROS_BG_WAKEUP: '', 'consciousness-allowance': '',
        'consciousness-model': 'Main model', 'consciousness-safety': '', 'consciousness-evolution': '' },
    'a value that was not read is not said: no "$0", no level and no "Off" the document did not carry');
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
        read: async () => ({ ...normalizeConsciousness(FIXTURE.consciousness), recentWakes: [] }),
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
    const range = rowNode(card, 'OUROBOROS_BG_WAKEUP');
    assert.deepEqual(range.querySelectorAll('[data-setting-control]').map((input) => input.value), ['300', '7200']);
    assert.equal(range.querySelectorAll('.settings-row-unit').map((unit) => unit.textContent).join(' '), 'to s', '[300] to [7200] s');
    assert.equal(range.querySelector('[data-row-caption]').textContent, 'It picks a moment between 5 minutes and 2 hours.');
    // One name for one limit: the door carries the name its row has on Costs (a rolling 24 hours), and the sum alone.
    assert.equal(rowNode(card, 'consciousness-allowance').querySelector('.ui-row-title').textContent, rowById('OUROBOROS_CONSCIOUSNESS_DAILY_USD').name);
    assert.equal(rowNode(card, 'consciousness-allowance').querySelector('[data-row-value]').textContent, '$20');
    assert.equal(rowNode(card, 'consciousness-model').querySelector('[data-row-value]').textContent, 'Main model');
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
    const [, path, { back }] = opened[1];
    assert.deepEqual([path, back.label], ['models/role/consciousness', 'Consciousness']);
    assert.deepEqual([opened.length, circles], [2, 1], 'the places door leaves through Circles alone');
    back.go();
    assert.deepEqual(opened[2], ['tab', 'consciousness', undefined]);
    // The Safety door opens its row in Permissions and checks, which leads back here; the evolution door opens its row on
    // the Evolution tab (a door inside the Dashboard asks for no Back).
    await card.fire('click', { target: rowNode(card, 'consciousness-safety') });
    await card.fire('click', { target: rowNode(card, 'consciousness-evolution') });
    const [, safetyPath, safetyOptions] = opened[3];
    assert.deepEqual([safetyPath, safetyOptions.row, safetyOptions.back.label], ['permissions', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'Consciousness']);
    assert.deepEqual(opened[4], ['tab', 'evolution', { row: 'OUROBOROS_POST_TASK_EVOLUTION' }]);
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
    const server = settingsServer(t, { OUROBOROS_POST_TASK_EVOLUTION: 'false', OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm',
        OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD: 0, OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE: '' });
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
    const { dispose } = initEvolution({ ws: { on: () => () => {} }, mount: server.mount,
        state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' } });
    await settle();
    const card = server.mount.querySelector('[data-evo-settings]');
    // The choice and the floor are one card's rows; the standing direction is a card of its own, its row drawn solo.
    const sections = card.querySelectorAll('section');
    assert.deepEqual(sections.map((section) => [section.getAttribute('aria-label'), section.hasAttribute('data-solo'),
        section.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow)]), [
        ['Cycles after your tasks', false, ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD']],
        ['What to work on', true, ['OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE']]]);
    assert.match(card.querySelector('#setting-OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE').getAttribute('placeholder'), /^For example: /);
    const caption = () => rowNode(card, 'OUROBOROS_POST_TASK_EVOLUTION').querySelector('[data-row-caption]').textContent;
    assert.equal(caption(), 'It starts no cycle by itself.');
    const choices = card.querySelectorAll('[data-evo-when]');
    const pressed = () => choices.filter((button) => button.getAttribute('aria-pressed') === 'true').map((button) => button.textContent);
    const count = card.querySelector('[data-evo-n]');
    assert.deepEqual(pressed(), ['Off']);
    assert.equal(count.parentElement.hidden, true, 'no count while it is off');
    count.value = '3';  // a browser reflects the field's value="3"; the fixture does not
    const click = (target) => card.fire('click', { type: 'click', target });
    const leave = (target) => card.fire('change', { type: 'change', target });
    await click(choices[2]);
    await settle();
    assert.deepEqual(server.posts, [{ OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:3' }]);
    assert.deepEqual(pressed(), ['Every N tasks']);
    assert.equal(count.parentElement.hidden, false);
    assert.equal(count.parentElement.querySelector('.settings-row-unit').textContent, 'N', 'the count is the N of its choice');
    assert.equal(caption(), 'After every 3 tasks, chats included, it decides whether one improvement is worth a cycle.');
    assert.equal(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), 'Saved · applies to new tasks · was Off · Undo');
    count.value = '7';
    await leave(count);
    await settle();
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:7' });
    count.value = '0';
    await leave(count);
    await settle();
    assert.equal(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), 'Not saved. Enter a whole number of tasks, 1 or more.');
    assert.equal(server.posts.length, 2, 'a count of 0 is never sent');
    await click(choices[2]);
    assert.equal(server.posts.length, 2, 'pressing the pressed choice sends nothing');
    await click(choices[1]);
    await settle();
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    assert.deepEqual(pressed(), ['After each task']);
    assert.equal(count.parentElement.hidden, true);
    dispose();
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
    assert.equal(caption.nextElementSibling, tasks.querySelector('[data-setting-result]'), 'the caption stands before the result line');
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

test('Back to Settings: a first row and one page layer; the press, Escape or Back blur the field first and open Settings', async (t) => {
    const { mount } = setup(t);
    const layers = [];
    const fakeLayers = { push: (layer) => layers.push(layer), remove: (id) => { layers.splice(0, layers.length, ...layers.filter((layer) => layer.id !== id)); } };
    const order = [];
    const back = settingsReturn({ layers: fakeLayers, openSettings: (path) => { order.push(`open ${path}`); } });
    const panel = new NodeStub('section');
    panel.innerHTML = '<div class="dashboard-body"><input data-setting-control></div>';
    mount.appendChild(panel);
    const field = panel.querySelector('[data-setting-control]');
    field.blur = () => { order.push('blur'); document.activeElement = null; };
    back.show(panel, 'models');
    assert.equal(panel.firstChild.className, 'settings-back-row dashboard-back', 'Back is the first row, above the scroller, drawn as a Settings page draws its own');
    assert.equal(panel.firstChild.textContent.trim(), 'Settings');
    const button = panel.firstChild.querySelector('[data-dashboard-back]');
    assert.deepEqual([button.className, button.getAttribute('aria-label')], ['settings-back ui-hit-area', 'Back to Settings']);
    assert.deepEqual(layers.map(({ id, el, page, modal, focus, opener }) => ({ id, el, page, modal, focus, opener })),
        [{ id: 'dashboard-back', el: panel, page: true, modal: false, focus: false, opener: null }]);
    field.focus();
    await panel.firstChild.querySelector('[data-dashboard-back]').fire('click');
    assert.deepEqual(order, ['blur', 'open models'], 'the field is left (and saved) before Settings opens');
    assert.equal(panel.querySelector('.dashboard-back'), null);
    assert.equal(layers.length, 0);
    // Escape and the system's Back pop the layer: its onClose goes the same way.
    back.show(panel, '');
    layers[0].onClose('escape');
    assert.deepEqual(order.at(-1), 'open ');
    assert.equal(panel.querySelector('.dashboard-back'), null);
    back.show(panel, 'accounts');
    back.clear();
    assert.deepEqual([panel.querySelector('.dashboard-back'), layers.length, order.length], [null, 0, 3], 'clearing opens nothing');
});

test('openDashboardTab: back gives the tab its Back, a row is brought into view and focused; another tab or page drops Back', async (t) => {
    const { mount, window } = setup(t);
    const content = new NodeStub(); content.id = 'content'; mount.appendChild(content);
    const opened = [];
    const layers = [];
    const state = { dashboardActiveSubtab: 'logs' };
    const dashboard = initDashboard({ state, openSettings: (path) => opened.push(path),
        layers: { push: (layer) => layers.push(layer), remove: (id) => { layers.splice(0, layers.length, ...layers.filter((layer) => layer.id !== id)); } } });
    const panel = (name) => dashboard.page.querySelector(`#dashboard-panel-${name}`);
    panel('costs').innerHTML = '<div class="dashboard-body"><details><summary>More</summary>'
        + '<div class="ui-row settings-row" data-setting-row="OUROBOROS_MAX_ROUNDS"><input data-setting-control></div></details></div>';
    const row = panel('costs').querySelector('[data-setting-row]');
    let scrolled = null;
    row.scrollIntoView = (options) => { scrolled = options; };
    dashboard.activateTab('costs', { row: 'OUROBOROS_MAX_ROUNDS', back: '' });
    assert.equal(panel('costs').firstChild.className, 'settings-back-row dashboard-back');
    assert.equal(row.closest('details').open, true, 'a row behind a fold opens it');
    assert.deepEqual(scrolled, { block: 'center' });
    assert.equal(document.activeElement, row.querySelector('[data-setting-control]'));
    // The system's Back returns to the Settings path the tab was opened from: '' is its home.
    layers[0].onClose('history');
    assert.deepEqual(opened, ['']);
    dashboard.activateTab('costs', { back: 'accounts' });
    assert.equal(layers.length, 1);
    await panel('costs').firstChild.querySelector('[data-dashboard-back]').fire('click');
    assert.deepEqual([opened, layers.length], [['', 'accounts'], 0], 'Back itself opens that page of Settings');
    dashboard.activateTab('costs', { back: 'accounts' });
    dashboard.activateTab('storage');
    assert.equal(panel('costs').querySelector('.dashboard-back'), null, 'another tab drops Back');
    assert.equal(layers.length, 0);
    dashboard.activateTab('costs');
    assert.equal(panel('costs').querySelector('.dashboard-back'), null, 'no back, no Back');
    dashboard.activateTab('costs', { back: '' });
    window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'chat' } }));
    assert.equal(panel('costs').querySelector('.dashboard-back'), null, 'leaving the Dashboard drops Back');
    assert.equal(layers.length, 0);
    assert.equal(focusSettingRow(panel('costs'), 'NO_SUCH_ROW'), false);
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
    assert.deepEqual(opened.map(([path, options]) => [path, options?.back?.label]), [['accounts/codex', 'Costs'], ['accounts', 'Costs']]);
    opened[0][1].back.go();
    assert.deepEqual(opened.at(-1), ['costs']);
    // A read that fails keeps the one door, saying why.
    routes.set('/api/claudexor/status?view=quota', response({ error: 'down' }, 502));
    mount.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(block.querySelectorAll('[data-costs-account]').map((door) => [door.dataset.costsAccount, door.querySelector('.ui-row-meta').textContent]),
        [['accounts', 'The accounts could not be read now.']]);
});
