// Dashboard › Costs, its limits card: rows of settings_rows.js drawn and written by settings_controls.js, the one
// writer of a setting. Costs posts nothing to /api/settings itself and reads nothing of a save's answer.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN } from '../modules/api_client.js';
import { initCosts } from '../modules/costs.js';
import { forgetSettingRows } from '../modules/settings_controls.js';
import { rowsOf } from '../modules/settings_rows.js';
import { setup } from './dashboard_dom_fixture.js';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const costsJs = readFileSync(new URL('../modules/costs.js', import.meta.url), 'utf8');

// A module that writes settings itself or reads a save's answer in its own words: the shapes Costs had before its
// rows saved themselves (the excerpt below is from that Costs), each a second writer beside settings_controls.js.
const PRIVATE_WRITER = [/\/api\/settings['"`][^)]*method:\s*['"]POST/s, /jsonPost\(\s*['"`]\/api\/settings/, /saveSettings\(/,
    /\bimmediate_changed\b/, /\bnext_task_changed\b/, /\brestart_required\b/];
const privateWrites = (source) => PRIVATE_WRITER.filter((pattern) => pattern.test(source)).map(String);
const OLD_COSTS_EXCERPT = `
            const data = await jsonPost('/api/settings', body, { headers: APP_SETTINGS_ORIGIN });
            const saved = budgetSavePresentation(data || {}, { sent: Object.keys(body) });
        else if (data.immediate_changed && nextTask) {
        if (data.restart_required) text = 'Saved. Restart required.';`;

function costsPage(t, values = {}, meta = {}) {
    forgetSettingRows();
    const env = setup(t);
    const stored = { TOTAL_BUDGET: 200, OUROBOROS_PER_TASK_COST_USD: 50, OUROBOROS_CONSCIOUSNESS_DAILY_USD: 20,
        OUROBOROS_MAX_ROUNDS: 'unlimited', OUROBOROS_TASK_ABS_CEILING_SEC: 3600, ...values };
    const posts = [];
    env.routes.set('/api/cost-breakdown', response({}, 503));
    env.routes.set('/api/consciousness', response({ enabled: true, spent_24h_usd: 1, daily_usd: 20 }));
    env.routes.set('/api/settings', (_url, init = {}) => {
        if (init.method !== 'POST') {
            return response({ ...stored, _meta: { ...meta, effects: { TOTAL_BUDGET: 'now' }, setup_contract: { budgetFields: [
                { stateKey: 'totalBudget', settingKey: 'TOTAL_BUDGET', min: '0.01', note: 'Budget limits use known spend: the root task tree.' },
                { stateKey: 'perTaskCostUsd', settingKey: 'OUROBOROS_PER_TASK_COST_USD', min: '0.01' }] } } });
        }
        const body = JSON.parse(init.body);
        posts.push({ body, headers: init.headers });
        const changes = Object.fromEntries(Object.entries(body).map(([key, value]) => [key, { old: stored[key], new: value, token: `${key}@1` }]));
        Object.assign(stored, body);
        return response({ status: 'saved', changes });
    });
    initCosts({ mount: env.mount, state: { activePage: 'dashboard' } });
    const card = env.mount.querySelector('.costs-budget');
    return {
        ...env, stored, posts, card,
        field: (key) => env.mount.querySelector(`#setting-${key}`),
        line: (key) => env.mount.querySelector(`#setting-${key}-result`).textContent,
        async refresh() {
            env.window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'costs' } }));
            await settle();
        },
    };
}

test('the limits card is the data module\'s Costs rows, each with its name, its meta and the notes that describe it', (t) => {
    const page = costsPage(t);
    const ids = page.card.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow);
    assert.deepEqual(ids, rowsOf('dashboard:costs').map((row) => row.id));
    assert.deepEqual(rowsOf('dashboard:costs', { fold: true }).map((row) => row.id), ['OUROBOROS_MAX_ROUNDS', 'OUROBOROS_TASK_ABS_CEILING_SEC']);
    const fold = page.card.querySelectorAll('details').find((details) => details.querySelector('summary').textContent === 'More limits');
    assert.deepEqual(fold.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow), ['OUROBOROS_MAX_ROUNDS', 'OUROBOROS_TASK_ABS_CEILING_SEC']);
    for (const [key, label, meta, notes] of [
        ['TOTAL_BUDGET', 'Total budget', /Paid work stops/, ['costs-budget-help']],
        ['OUROBOROS_PER_TASK_COST_USD', 'One task', /A task and its helpers stop/, ['costs-budget-help']],
        ['OUROBOROS_CONSCIOUSNESS_DAILY_USD', 'Consciousness per 24 hours', /wake-ups, replies in places and the work they start/,
            ['costs-allowance-spend', 'costs-allowance-state', 'costs-allowance-help']]]) {
        const field = page.field(key);
        assert.equal(page.mount.querySelector(`[for="setting-${key}"]`).textContent, label);
        assert.equal(field.classList.contains('ui-control'), true);
        assert.deepEqual(field.getAttribute('aria-describedby').split(' '), [`setting-${key}-meta`, `setting-${key}-result`, ...notes]);
        assert.match(page.mount.querySelector(`#setting-${key}-meta`).textContent, meta);
        assert.equal(page.mount.querySelector(`#setting-${key}`).parentElement.querySelector('.settings-row-unit').textContent, '$');
    }
    const summaries = page.card.querySelectorAll('summary').map((summary) => summary.textContent);
    assert.deepEqual(summaries, ['How the total and per-task limits count', 'How the consciousness limit counts', 'More limits']);
    assert.match(page.mount.querySelector('#costs-budget-help').textContent, /root task tree/);
    assert.doesNotMatch(page.mount.querySelector('#costs-allowance-help').textContent, /root task tree/);
    assert.equal(page.card.querySelector('#btn-save-budget'), null, 'no Save: each row saves itself');
    assert.equal(page.card.querySelectorAll('[data-setting-no-limit]').length, 2, 'the two bounds that may be unlimited offer "No limit"');
});

test('one writer: Costs posts nothing to /api/settings itself and reads nothing of a save\'s answer', () => {
    assert.ok(privateWrites(OLD_COSTS_EXCERPT).length >= 3, 'the check sees the private writer Costs used to be');
    assert.deepEqual(privateWrites(costsJs), []);
    assert.match(costsJs, /import \{ bindSettingRows, readSettings, settingsRow \} from '\.\/settings_controls\.js'/);
    assert.doesNotMatch(costsJs, /budgetSavePresentation|btn-save-budget|budget-save-status|Saving…/);
});

test('leaving a limit\'s field saves it with the app\'s origin and the row says what happened; its spend is read again', async (t) => {
    const page = costsPage(t);
    await page.refresh();
    const reads = page.calls.filter((url) => url === '/api/cost-breakdown').length;
    const field = page.field('TOTAL_BUDGET');
    assert.equal(field.value, '200');
    field.value = '250';
    await page.card.fire('input', { target: field });
    assert.equal(page.posts.length, 0, 'typing saves nothing');
    await page.card.fire('change', { target: field });
    await settle();
    assert.deepEqual(page.posts, [{ body: { TOTAL_BUDGET: 250 }, headers: { 'Content-Type': 'application/json', ...APP_SETTINGS_ORIGIN } }]);
    assert.equal(page.line('TOTAL_BUDGET'), 'Saved · was 200 · Undo');
    assert.equal(page.calls.filter((url) => url === '/api/cost-breakdown').length, reads + 1, 'the total moves the known-spend headline');
    const noLimit = (key) => page.card.querySelector(`[data-setting-row="${key}"]`).querySelector('[data-setting-no-limit]');
    await page.card.fire('click', { target: noLimit('OUROBOROS_MAX_ROUNDS') });
    await settle();
    assert.equal(page.posts.length, 1, 'rounds are already unlimited: nothing to send');
    await page.card.fire('click', { target: noLimit('OUROBOROS_TASK_ABS_CEILING_SEC') });
    await settle();
    assert.deepEqual(page.posts.at(-1).body, { OUROBOROS_TASK_ABS_CEILING_SEC: 'unlimited' });
    assert.equal(page.field('OUROBOROS_TASK_ABS_CEILING_SEC').value, '', 'the field is empty and "No limit" is the pressed choice');
    assert.equal(noLimit('OUROBOROS_TASK_ABS_CEILING_SEC').getAttribute('aria-pressed'), 'true');
});

test('0 is the allowance\'s own choice; an emptied field is no number, and the total keeps the contract\'s floor', async (t) => {
    const page = costsPage(t);
    await page.refresh();
    const type = async (key, text) => {
        const field = page.field(key);
        field.value = text;
        await page.card.fire('input', { target: field });
        await page.card.fire('change', { target: field });
        await settle();
    };
    await type('OUROBOROS_CONSCIOUSNESS_DAILY_USD', '0');
    assert.deepEqual(page.posts.map((post) => post.body), [{ OUROBOROS_CONSCIOUSNESS_DAILY_USD: 0 }], 'it may not spend: a real choice');
    await type('OUROBOROS_CONSCIOUSNESS_DAILY_USD', ' ');
    await type('TOTAL_BUDGET', '0');
    assert.equal(page.posts.length, 1, 'neither is sent');
    assert.equal(page.line('OUROBOROS_CONSCIOUSNESS_DAILY_USD'), 'Not saved. Enter a number, 0 or more.');
    assert.equal(page.line('TOTAL_BUDGET'), 'Not saved. Enter a number, 0.01 or more.');
    assert.equal(page.field('TOTAL_BUDGET').value, '0', 'the typed text stays');
});

test('a refresh never rewrites a field the owner is in or typed into; the other fields take the fresh values', async (t) => {
    const page = costsPage(t);
    await page.refresh();
    const allowance = page.field('OUROBOROS_CONSCIOUSNESS_DAILY_USD');
    allowance.value = '15.25';
    await page.card.fire('input', { target: allowance });
    const perTask = page.field('OUROBOROS_PER_TASK_COST_USD');
    perTask.focus();
    Object.assign(page.stored, { OUROBOROS_CONSCIOUSNESS_DAILY_USD: 30, OUROBOROS_PER_TASK_COST_USD: 75, TOTAL_BUDGET: 400 });
    await page.refresh();
    assert.equal(allowance.value, '15.25', 'the typed text stays');
    assert.equal(perTask.value, '50', 'the focused field is left alone');
    assert.equal(page.field('TOTAL_BUDGET').value, '400', 'an untouched field shows the fresh value');
    allowance.value = '';
    await page.card.fire('input', { target: allowance });
    await page.refresh();
    assert.equal(allowance.value, '', 'an emptied field is not refilled with a default');
    assert.equal(page.posts.length, 0);
    page.window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'logs' } }));
    assert.equal(allowance.value, '', 'leaving the tab keeps an unsaved draft where it is');
});

test('under the limits one line names what they cannot count, in the page language; none when every route is priced', async (t) => {
    const { applyPayload, setMissTransport } = await import('../modules/i18n.js');
    setMissTransport(() => Promise.resolve(null));
    const view = costsPage(t, {}, { known_spend_gaps: ['Web search', 'Agents'] });
    await view.refresh();
    const line = view.card.querySelector('#costs-gaps');
    assert.equal(line.hidden, false);
    assert.equal(line.textContent, 'Not counted: Web search, Agents (price unknown).');
    // The words alone: without a document i18n.js leaves the fixture's page unwalked.
    const language = (payload) => { const kept = globalThis.document; globalThis.document = undefined; try { applyPayload(payload); } finally { globalThis.document = kept; } };
    language({ language: 'ru', english: false, revision: 61, entries: {
        'Not counted: {roles} (price unknown).': { text: 'Не учитываются: {roles} (цена неизвестна).' }, 'Web search': { text: 'Веб-поиск' } } });
    try {
        view.window.dispatchEvent(new CustomEvent('ouro:language-changed'));
        assert.equal(line.textContent, 'Не учитываются: Веб-поиск, Agents (цена неизвестна).');
    } finally {
        language({ language: '', english: true, revision: 62, entries: {} });
    }
    const priced = costsPage(t);
    await priced.refresh();
    assert.equal(priced.card.querySelector('#costs-gaps').hidden, true, 'every route has a price: the line is not drawn');
});
