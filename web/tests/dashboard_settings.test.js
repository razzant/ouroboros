// The Dashboard tabs' own settings (dashboard.js mountTabSettings): rows of settings_rows.js that save themselves, drawn
// once outside every region a tab repaints; the way back to Settings (settingsReturn); Costs' subscriptions from the
// passive quota read. INTEGRATED marks what needs another lane's half: the server's save receipt and `_meta.effects`
// (lane A) and lane C1's openSettings. The fakes below answer it until those land; the lane report lists each one.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { AUTONOMY_LEVELS } from '../modules/consciousness_autonomy.js';
import { cardWords, initConsciousnessDashboard, waitWords } from '../modules/consciousness_dashboard.js';
import { initCosts, subscriptionDoors } from '../modules/costs.js';
import { focusSettingRow, initDashboard, settingsReturn } from '../modules/dashboard.js';
import { initEvolution, whenMode } from '../modules/evolution.js';
import { CODECS, commitSetting, forgetSettingRows, outcomeLine, readSettings } from '../modules/settings_controls.js';
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

// INTEGRATED (lane A): when a key's change applies, as GET /api/settings `_meta.effects` says it.
const EFFECTS = { OUROBOROS_BG_WAKEUP_MIN: 'next_wake', OUROBOROS_BG_WAKEUP_MAX: 'next_wake', OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 'next_wake' };

/** /api/settings over `stored`: a read answers it with `_meta.effects`; a save stores its body and answers the receipt
 *  (INTEGRATED, lane A: `changes`; today's server answers none, and a row then says only "Saved"). */
function settingsServer(t, values = {}) {
    forgetSettingRows();
    const env = setup(t);
    const stored = { ...values };
    const posts = [];
    env.routes.set('/api/settings', (_url, init = {}) => {
        if (init.method !== 'POST') return response({ ...stored, _meta: { effects: EFFECTS } });
        const body = JSON.parse(init.body);
        posts.push(body);
        const changes = Object.fromEntries(Object.entries(body).map(([key, value]) => [key, { old: stored[key], new: value, token: `${key}@1` }]));
        Object.assign(stored, body);
        return response({ status: 'saved', changes });
    });
    return { ...env, stored, posts };
}

const MOVED = {
    'dashboard:consciousness': ['OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX',
        'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'OUROBOROS_PRESENCE_MAX_ACTIVE'],
    'dashboard:evolution': ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE',
        'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE'],
    'dashboard:updates': ['OUROBOROS_UPDATE_CHANNEL'],
    'dashboard:storage': ['OUROBOROS_GC_RETENTION_DAYS'],
};

test('each moved key is written by exactly one row, on its own tab; the two doors write none and are found elsewhere', () => {
    for (const [page, keys] of Object.entries(MOVED)) {
        for (const key of keys) assert.deepEqual(ROWS.filter((row) => row.keys.includes(key)).map((row) => row.page), [page], key);
        assert.deepEqual(rowsOf(page).flatMap((row) => row.keys).sort(), [...keys].sort(), `${page} writes nothing else`);
    }
    assert.deepEqual(rowById('OUROBOROS_BG_WAKEUP').keys, ['OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX'], 'the wake range is one row');
    assert.deepEqual(rowsOf('dashboard:consciousness').map((row) => row.id), ['OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'OUROBOROS_BG_WAKEUP',
        'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 'consciousness-allowance', 'consciousness-model', 'OUROBOROS_PRESENCE_MAX_ACTIVE']);
    assert.deepEqual(rowsOf('dashboard:consciousness', { fold: true }).map((row) => row.id), ['OUROBOROS_PRESENCE_MAX_ACTIVE']);
    for (const id of ['consciousness-allowance', 'consciousness-model']) {
        const door = rowById(id);
        assert.deepEqual([door.control.kind, door.keys, door.aliases], ['door', [], []], `${id}: search finds its setting in its own home`);
    }
    // The strip is the levels' one list: its labels and its More are AUTONOMY_LEVELS'.
    const autonomy = rowById('OUROBOROS_CONSCIOUSNESS_AUTONOMY');
    assert.deepEqual(autonomy.control.options, AUTONOMY_LEVELS.map(({ value, label }) => ({ value, label })));
    for (const level of AUTONOMY_LEVELS) assert.ok(autonomy.more.includes(`${level.label}: ${level.note}`), level.value);
    assert.equal(rowById('OUROBOROS_UPDATE_CHANNEL').hiddenWhen, 'managedDeployment', 'a managed deployment leaves the channel out');
});

test('the evolution choice is one row over two keys: one body carries both, and they read back as the choice', async (t) => {
    const server = settingsServer(t, { OUROBOROS_POST_TASK_EVOLUTION: 'false', OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'llm' });
    const row = rowById('OUROBOROS_POST_TASK_EVOLUTION');
    assert.deepEqual(row.keys, ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE']);
    assert.deepEqual(CODECS.evolutionWhen.read(await readSettings(), row), [false, 'llm']);
    const saved = await commitSetting(row, [true, 'every_n:5']);
    assert.deepEqual(server.posts, [{ OUROBOROS_POST_TASK_EVOLUTION: true, OUROBOROS_POST_TASK_EVOLUTION_CADENCE: 'every_n:5' }]);
    assert.equal(outcomeLine(row, saved).text, 'Saved · applies to new tasks · was Off', 'it names the choice it replaced');  // INTEGRATED (lane A)
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
});

test('the wake range is one row of two numbers; a first number above the second is refused before anything is sent', async (t) => {
    const server = settingsServer(t, { OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200 });
    const row = rowById('OUROBOROS_BG_WAKEUP');
    await readSettings();
    const refused = await commitSetting(row, [7200, 300]);
    assert.equal(outcomeLine(row, refused).text, 'Not saved. The first number is above the second.');
    assert.equal(server.posts.length, 0);
    const saved = await commitSetting(row, [600, 3600]);
    assert.deepEqual(server.posts, [{ OUROBOROS_BG_WAKEUP_MIN: 600, OUROBOROS_BG_WAKEUP_MAX: 3600 }], 'both numbers in one body');
    assert.equal(outcomeLine(row, saved).text, 'Saved · applies from the next wake · was 300 to 7200');  // INTEGRATED (lane A)
    await commitSetting(row, [900, 900]);
    assert.deepEqual(server.posts.at(-1), { OUROBOROS_BG_WAKEUP_MIN: 900, OUROBOROS_BG_WAKEUP_MAX: 900 }, 'one fixed wait is a range too');
    // The caption says the seconds in minutes or hours; the doors say their values.
    assert.deepEqual([60, 300, 90, 3600, 5400, 86400, 0, '', 'x'].map(waitWords),
        ['1 minute', '5 minutes', '1.5 minutes', '1 hour', '1.5 hours', '24 hours', '', '', '']);
    assert.deepEqual(cardWords({ OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20,
        OUROBOROS_MODEL_CONSCIOUSNESS: '' }), { range: '5 minutes to 2 hours', allowance: '$20.00', model: 'Main model' });
    assert.deepEqual(cardWords(null), { range: '', allowance: '', model: '' }, 'nothing read says nothing');
});

test('Consciousness: its card is drawn once beside the summary; a refresh leaves the card and a typed field alone', async (t) => {
    const server = settingsServer(t, { OUROBOROS_CONSCIOUSNESS_AUTONOMY: 'act', OUROBOROS_BG_WAKEUP_MIN: 300, OUROBOROS_BG_WAKEUP_MAX: 7200,
        OUROBOROS_CONSCIOUSNESS_MAX_TASKS: 2, OUROBOROS_PRESENCE_MAX_ACTIVE: 3, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20, OUROBOROS_MODEL_CONSCIOUSNESS: '' });
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
    assert.deepEqual(card.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow), rowsOf('dashboard:consciousness').map((row) => row.id));
    const fold = card.querySelector('.dashboard-settings-more');
    assert.equal(fold.querySelector('summary').textContent, 'More: conversations at once');
    assert.deepEqual(fold.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow), ['OUROBOROS_PRESENCE_MAX_ACTIVE']);
    assert.equal(rowNode(card, 'OUROBOROS_CONSCIOUSNESS_AUTONOMY').querySelector('.active').textContent, 'Acts');
    const range = rowNode(card, 'OUROBOROS_BG_WAKEUP');
    assert.deepEqual(range.querySelectorAll('[data-setting-control]').map((input) => input.value), ['300', '7200']);
    assert.equal(range.querySelectorAll('.settings-row-unit').map((unit) => unit.textContent).join(' '), 'to s', '[300] to [7200] s');
    assert.equal(range.querySelector('.settings-row-value').textContent, '5 minutes to 2 hours');
    assert.equal(rowNode(card, 'consciousness-allowance').querySelector('.settings-row-value').textContent, '$20.00');
    assert.equal(rowNode(card, 'consciousness-model').querySelector('.settings-row-value').textContent, 'Main model');
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
    assert.equal(result(card, 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS'), 'Saved · applies from the next wake · was 5 · Undo');  // INTEGRATED (lane A)
    // A range upside down is refused in its row; nothing is sent.
    const [shortest, longest] = range.querySelectorAll('[data-setting-control]');
    shortest.value = '7200';
    longest.value = '300';
    await card.fire('change', { target: longest });
    await settle();
    assert.equal(result(card, 'OUROBOROS_BG_WAKEUP'), 'Not saved. The first number is above the second.');
    assert.equal(server.posts.length, 1);
    // The allowance door opens its row in Costs; the model door its role in Models, which leads back here.
    await card.fire('click', { target: rowNode(card, 'consciousness-allowance') });
    await card.fire('click', { target: rowNode(card, 'consciousness-model') });
    assert.deepEqual(opened.slice(0, 1), [['tab', 'costs', { row: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD' }]]);
    const [, path, { back }] = opened[1];
    assert.deepEqual([path, back.label], ['models/role/consciousness', 'Consciousness']);  // INTEGRATED (lane C1's openSettings)
    back.go();
    assert.deepEqual(opened[2], ['tab', 'consciousness', undefined]);
    controller.destroy();
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
    const card = server.mount.querySelector('[aria-label="Cycles"]');
    assert.deepEqual(card.querySelectorAll('[data-setting-row]').map((node) => node.dataset.settingRow), rowsOf('dashboard:evolution').map((row) => row.id));
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
    assert.equal(result(card, 'OUROBOROS_POST_TASK_EVOLUTION'), 'Saved · applies to new tasks · was Off');  // INTEGRATED (lane A)
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
    assert.equal(panel.firstChild.className, 'dashboard-back', 'Back is the first row, above the scroller');
    assert.equal(panel.firstChild.textContent.trim(), 'Settings');
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
    const pages = [];
    const layers = [];
    const state = { dashboardActiveSubtab: 'logs' };
    const dashboard = initDashboard({ state, showPage: (name) => pages.push(name),
        layers: { push: (layer) => layers.push(layer), remove: (id) => { layers.splice(0, layers.length, ...layers.filter((layer) => layer.id !== id)); } } });
    const panel = (name) => dashboard.page.querySelector(`#dashboard-panel-${name}`);
    panel('costs').innerHTML = '<div class="dashboard-body"><details><summary>More</summary>'
        + '<div class="ui-row settings-row" data-setting-row="OUROBOROS_MAX_ROUNDS"><input data-setting-control></div></details></div>';
    const row = panel('costs').querySelector('[data-setting-row]');
    let scrolled = null;
    row.scrollIntoView = (options) => { scrolled = options; };
    dashboard.activateTab('costs', { row: 'OUROBOROS_MAX_ROUNDS', back: '' });
    assert.equal(panel('costs').firstChild.className, 'dashboard-back');
    assert.equal(row.closest('details').open, true, 'a row behind a fold opens it');
    assert.deepEqual(scrolled, { block: 'center' });
    assert.equal(document.activeElement, row.querySelector('[data-setting-control]'));
    // Until lane C1's openSettings lands, Back leads to the Settings page itself (INTEGRATED: openSettings(path)).
    layers[0].onClose('history');
    assert.deepEqual(pages, ['settings']);
    dashboard.activateTab('costs', { back: 'accounts' });
    assert.equal(layers.length, 1);
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

test('Subscriptions: one door per kind of account, each account its own window (never an average), then Accounts and keys', () => {
    const doors = subscriptionDoors(passive(), { nowMs: NOW });
    assert.deepEqual(doors, [
        { path: 'accounts/codex', name: 'Codex', meta: 'Work: 20% used · resets in 2h · Home: Limit reached · resets in 5h · 1 out of rotation' },
        { path: 'accounts/claude', name: 'Claude Code', meta: '80% used · resets in 30h' },
        { path: 'accounts', name: 'Accounts and keys', meta: 'Sign in, add an account or take one out of rotation.' },
    ]);
    assert.doesNotMatch(JSON.stringify(doors), /60%|50%/, 'no ratio of two accounts is averaged');
    // A quota that was not read licenses no usage claim; a roster that was not read lists no account.
    assert.match(subscriptionDoors(passive({ reads: { catalog: 'not_read', accounts: 'ok', quota: 'failed' } }), { nowMs: NOW })[1].meta, /^Limits not checked$/);
    for (const unread of [passive({ reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' }, profiles: {}, quota: [] }),
        passive({ reads: { catalog: 'not_read', accounts: 'failed', quota: 'ok' } })]) {
        assert.deepEqual(subscriptionDoors(unread, { nowMs: NOW }), [{ path: 'accounts', name: 'Accounts and keys', meta: 'The accounts could not be read now.' }]);
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
    routes.set('/api/claudexor/status?view=quota', response(passive()));
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
    await block.fire('click', { target: doors[0] });
    await block.fire('click', { target: doors[2] });
    assert.deepEqual(opened.map(([path, options]) => [path, options?.back?.label]), [['accounts/codex', 'Costs'], ['accounts', 'Costs']]);  // INTEGRATED (lane C1)
    opened[0][1].back.go();
    assert.deepEqual(opened.at(-1), ['costs']);
    // A read that fails keeps the one door, saying why.
    routes.set('/api/claudexor/status?view=quota', response({ error: 'down' }, 502));
    mount.querySelector('#btn-refresh-costs').fire('click');
    await settle();
    assert.deepEqual(block.querySelectorAll('[data-costs-account]').map((door) => [door.dataset.costsAccount, door.querySelector('.ui-row-meta').textContent]),
        [['accounts', 'The accounts could not be read now.']]);
});
