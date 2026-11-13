// The plain pages of Settings (settings_ui.js; docs/DESIGN.md "Settings"): Permissions and checks, Your rules,
// Appearance, Notifications, Voice and calls, This computer with its Advanced and About. Their rows as data
// (settings_rows.js), the `choices` control they add to the row helper (settings_controls.js), and each page mounted
// the way the shell mounts it, against a fake settings server that answers like the server with its receipt.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { SAFETY_LEVELS } from '../modules/consciousness_words.js';
import { forgetSettingRows, readSettings, settingsRow } from '../modules/settings_controls.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { policyValueLabel } from '../modules/settings_policy.js';
import { HOME_ROWS, ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { forgetUnsavedRules, rulesSaveWords, spareWords } from '../modules/settings_safety.js';
import { tunePermissionRows, voiceFacts } from '../modules/settings_ui.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const MY_PAGES = ['permissions', 'rules', 'appearance', 'notifications', 'voice', 'computer', 'computer/advanced', 'computer/about'];
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 10) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const EFFECTS = { OUROBOROS_RUNTIME_MODE: 'restart', OUROBOROS_SERVER_HOST: 'restart', OUROBOROS_SAFETY_MODE: 'next_task',
    OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'next_task', OUROBOROS_TASK_REVIEW_MODE: 'next_task', OUROBOROS_NETWORK_PASSWORD: 'now',
    OUROBOROS_SAFETY_OWNER_RULES: 'next_task', OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: 'now', OUROBOROS_UI_DIR: 'restart',
    OUROBOROS_MAX_WORKERS: 'now', OUROBOROS_EXTRA_CA_BUNDLE: 'next_task' };

/** A settings server: the document (the password masked, as the server masks it), the owner endpoints with their
 *  receipts (a secret's withheld), what a read derives from them, and Show's read of the stored password. */
function fakeServer(routes, values = {}) {
    const server = { posts: [], refuse: new Map(), running: { OUROBOROS_RUNTIME_MODE: 'pro', OUROBOROS_SERVER_HOST: '127.0.0.1', OUROBOROS_UI_DIR: '' },
        values: { OUROBOROS_RUNTIME_MODE: 'pro', OUROBOROS_SAFETY_MODE: 'full', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'light',
            OUROBOROS_SAFETY_OWNER_RULES: 'Ask before paying.', OUROBOROS_TASK_REVIEW_MODE: 'auto',
            OUROBOROS_REVIEW_MAX_CYCLES: '2', OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: 'false', OUROBOROS_SERVER_HOST: '127.0.0.1',
            OUROBOROS_NETWORK_PASSWORD: '', OUROBOROS_MAX_WORKERS: '', OUROBOROS_UI_DIR: '', OUROBOROS_EXTRA_CA_BUNDLE: '', ...values } };
    server.document = () => {
        const { values: now, running } = server;
        const pending = Object.keys(running).filter((key) => String(now[key]) !== String(running[key])).sort();
        return { ...structuredClone(now), OUROBOROS_NETWORK_PASSWORD: now.OUROBOROS_NETWORK_PASSWORD ? '***set***' : '',
            _meta: { effects: EFFECTS, bind_host: running.OUROBOROS_SERVER_HOST, reachability: 'loopback_only',
            policy_state: { access: { configured: now.OUROBOROS_RUNTIME_MODE, current_process: running.OUROBOROS_RUNTIME_MODE,
                effective: running.OUROBOROS_RUNTIME_MODE, restart_required: now.OUROBOROS_RUNTIME_MODE !== running.OUROBOROS_RUNTIME_MODE } },
            restart_state: { restart_keys: pending, running: Object.fromEntries(pending.map((key) => [key, running[key]])) } } };
    };
    const write = (changed) => Object.fromEntries(Object.entries(changed).filter(([key, value]) => String(server.values[key]) !== String(value))
        .map(([key, value]) => {
            const old = server.values[key];
            server.values[key] = value;
            return [key, key === 'OUROBOROS_NETWORK_PASSWORD' ? { withheld: 'secret' } : { old, new: value }];
        }));
    const route = (path, answer) => routes.set(path, async (_url, init = {}) => {
        if (init.method !== 'POST') return response(server.document());
        const body = JSON.parse(init.body);
        server.posts.push([path, body]);
        if (server.refuse.has(path)) { const refusal = server.refuse.get(path); server.refuse.delete(path); return response(...refusal); }
        return response(answer(body));
    });
    route('/api/settings', (body) => ({ status: 'saved', changes: write(body) }));
    route('/api/owner/runtime-mode', ({ mode }) => ({ ok: true, runtime_mode: mode, changes: write({ OUROBOROS_RUNTIME_MODE: mode }) }));
    route('/api/owner/safety-mode', ({ mode, kind }) => ({ ok: true, safety_mode: mode,
        changes: write({ [kind === 'consciousness' ? 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE' : 'OUROBOROS_SAFETY_MODE']: mode }) }));
    route('/api/owner/safety-rules', ({ rules }) => ({ ok: true, rules, changes: write({ OUROBOROS_SAFETY_OWNER_RULES: rules }) }));
    route('/api/owner/auto-grant', ({ enabled }) => ({ ok: true, enabled, changes: write({ OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: String(enabled) }) }));
    server.revealed = [];
    routes.set('/api/settings/secret', async (_url, init) => {
        server.revealed.push(JSON.parse(init.body));
        return response({ value: server.values[JSON.parse(init.body).key] });
    });
    routes.set('/api/logs/events?limit=2000&view=compact', response({ entries: [] }));
    routes.set('/api/desktop/autostart', response({ state: 'on' }));
    routes.set('/api/desktop/background', response({ state: 'off' }));
    routes.set('/api/state', response({ served_ui: { source: 'builtin' } }));
    routes.set('/api/extensions', response({ live: { settings_sections: [] }, skills: [] }));
    server.writes = (path) => server.posts.filter(([posted]) => posted === path).map(([, body]) => body);
    return server;
}

/** Mount `path` the way the shell does and show it; `asked` collects the questions the page asked first. */
async function open(t, path, { values = {}, answer = true } = {}) {
    forgetSettingRows();
    forgetUnsavedRules();
    const env = setup(t);
    const server = fakeServer(env.routes, values);
    const asked = [];
    const opened = [];
    const container = new NodeStub();
    const ctx = { readSettings, openSettings: (to) => opened.push(to), ask: async (dialog) => { asked.push(dialog); return answer; } };
    const handle = await settingsPageFor(path).mount(container, { path, row: '', ctx });
    const shown = async () => { env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path } })); await settle(); };
    await shown();
    t.after(() => handle?.unmount?.());
    const row = (id) => container.querySelectorAll('[data-setting-row]').find((node) => node.dataset.settingRow === id);
    return { ...env, server, asked, opened, container, handle, shown, row,
        meta: (id) => row(id).querySelector('[data-setting-meta]').textContent,
        line: (id) => row(id).querySelector('[data-setting-result]').textContent,
        option: (id, value) => row(id).querySelectorAll('[data-setting-value]').find((button) => button.dataset.settingValue === value),
        async press(target) { await container.fire('click', { target }); await settle(); } };
}

test('every key of the plain pages has exactly one row, on a page that is registered, and the module stays data', () => {
    const mine = ROWS.filter((row) => MY_PAGES.includes(row.page));
    const keys = mine.flatMap((row) => row.keys);
    assert.deepEqual(keys.filter((key, i) => keys.indexOf(key) !== i), [], 'no key is written by two rows');
    for (const key of ['OUROBOROS_RUNTIME_MODE', 'OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'OUROBOROS_SAFETY_OWNER_RULES',
        'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', 'OUROBOROS_TASK_REVIEW_MODE', 'OUROBOROS_REVIEW_MAX_CYCLES',
        'OUROBOROS_UI_LANGUAGE', 'OUROBOROS_DESKTOP_KEEP_RUNNING', 'OUROBOROS_SERVER_HOST', 'OUROBOROS_NETWORK_PASSWORD', 'OUROBOROS_MAX_WORKERS',
        'OUROBOROS_UI_DIR', 'OUROBOROS_EXTRA_CA_BUNDLE']) {
        assert.equal(ROWS.filter((row) => row.keys.includes(key)).length, 1, `${key} has one row in the whole module`);
        assert.ok(MY_PAGES.includes(ROWS.find((row) => row.keys.includes(key)).page), key);
    }
    for (const path of MY_PAGES) {
        assert.equal(settingsPageFor(path)?.prefix, path, `${path} is a registered page of its own`);
        assert.ok(rowsOf(path).length, `${path} has rows`);
    }
    // Every door of these pages leads to a page that exists here, to a page of another lane's family, or out by a door id.
    const families = new Set(['models', 'agents', 'skill']);
    for (const row of mine.filter((each) => each.control.kind === 'door')) {
        const to = row.control.to;
        assert.ok(MY_PAGES.includes(to) || families.has(to.split('/')[0]) || /^(dashboard|page|main):/.test(to), `${row.id} leads to ${to}`);
    }
    // The owner writers: each key takes its own endpoint, and what asks first is named.
    assert.deepEqual(['OUROBOROS_RUNTIME_MODE', 'OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS',
        'OUROBOROS_SERVER_HOST'].map((id) => [rowById(id).writer, rowById(id).confirm]),
    [['runtimeMode', 'raiseAccess'], ['safetyMode', 'lowerSafety'], ['safetyMode', 'lowerSafety'], ['autoGrant', 'autoGrantOn'], ['generic', 'openNetwork']]);
    assert.deepEqual(JSON.parse(JSON.stringify(ROWS)), ROWS);
    // A page's rare rows sit behind its one fold: the fold rows of each page are few and named.
    assert.deepEqual(mine.filter((row) => row.fold).map((row) => row.id), ['OUROBOROS_REVIEW_MAX_CYCLES', 'appearance:translations', 'computer:reset']);
});

test('the Safety scale reads Off, Outgoing, + Commands wherever it is drawn, and stores off, light, full', async (t) => {
    const words = SAFETY_LEVELS.map(({ value, label, note }) => ({ value, label, note }));
    assert.deepEqual(words.map((level) => [level.value, level.label]), [['off', 'Off'], ['light', 'Outgoing'], ['full', '+ Commands']]);
    // The rows restate the scale as data: the same values, labels and lines, for both levels.
    for (const id of ['OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE']) assert.deepEqual(rowById(id).control.options, words, id);
    assert.deepEqual(['off', 'light', 'full'].map((value) => policyValueLabel(value, 'coverage')), ['Off', 'Outgoing', '+ Commands']);
    // Drawn: the strip's words, and the chosen level's line as the row's one line.
    const page = await open(t, 'permissions');
    for (const id of ['OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE']) {
        assert.deepEqual(page.row(id).querySelectorAll('[data-setting-value]').map((button) => button.textContent), ['Off', 'Outgoing', '+ Commands']);
    }
    assert.equal(page.meta('OUROBOROS_SAFETY_MODE'), 'Shell commands and scripts are read first too. File edits are not.');
    assert.equal(page.meta('OUROBOROS_CONSCIOUSNESS_SAFETY_MODE'), 'Messages, posts, skills, connectors and starting work are read first.');
    // No module of the client spells the old labels of this scale beside its stored values.
    for (const name of fs.readdirSync(new URL('../modules/', import.meta.url)).filter((file) => file.endsWith('.js'))) {
        const source = fs.readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8');
        assert.doesNotMatch(source, /label: 'Full'|full: 'Full'|Off · Light · Full|Light or Full|Light \/ Full/, name);
    }
    // The Access levels and the autonomy words keep their names.
    assert.deepEqual(rowById('OUROBOROS_RUNTIME_MODE').control.options.map((option) => option.label), ['Light', 'Advanced', 'Pro', 'Cyber Pro']);
});

test('Access is four choices, each with its line; its meta says both values while a restart waits and one line otherwise', async (t) => {
    const page = await open(t, 'permissions');
    const access = page.row('OUROBOROS_RUNTIME_MODE');
    assert.equal(access.dataset.kind, 'choices');
    const options = access.querySelectorAll('[role="radio"]');
    assert.deepEqual(options.map((button) => button.querySelector('.settings-choice-name').textContent), ['Light', 'Advanced', 'Pro', 'Cyber Pro']);
    assert.equal(options[3].querySelector('.settings-choice-note').textContent, 'Full say over itself and this computer. Asks for reviews when it judges.');
    assert.equal(access.querySelector('[role="radiogroup"]').getAttribute('aria-labelledby'), 'setting-OUROBOROS_RUNTIME_MODE-name');
    assert.deepEqual(options.map((button) => button.getAttribute('aria-checked')), ['false', 'false', 'true', 'false']);
    assert.deepEqual(options.map((button) => button.tabIndex), [-1, -1, 0, -1], 'one tab stop: the chosen level');
    assert.equal(page.meta('OUROBOROS_RUNTIME_MODE'), 'How far it reaches into its own code and this computer.');
    assert.deepEqual(access.querySelectorAll('[data-setting-mark]').map((mark) => mark.textContent), ['', '', '', ''], 'nothing waits: no mark');
    // The setup wizard says the same four sentences: its contract, as the server builds it for a fresh install.
    const { contract } = JSON.parse(fs.readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
    assert.deepEqual(contract.runtimeModes.map((mode) => [mode.value, mode.label, mode.copy]),
        rowById('OUROBOROS_RUNTIME_MODE').control.options.map((option) => [option.value, option.label, option.note]));

    // Raising asks first; declined, nothing is sent and nothing changes.
    page.asked.length = 0;
    const declined = await open(t, 'permissions', { answer: false });
    await declined.press(declined.option('OUROBOROS_RUNTIME_MODE', 'cyber_pro'));
    assert.equal(declined.asked[0].title, 'Raise Access');
    assert.deepEqual(declined.server.writes('/api/owner/runtime-mode'), []);
    assert.equal(declined.option('OUROBOROS_RUNTIME_MODE', 'pro').getAttribute('aria-checked'), 'true');

    const raised = await open(t, 'permissions');
    await raised.press(raised.option('OUROBOROS_RUNTIME_MODE', 'cyber_pro'));
    assert.match(raised.asked[0].body, /^Raise Access from Pro to Cyber Pro\?/);
    assert.deepEqual(raised.server.writes('/api/owner/runtime-mode'), [{ mode: 'cyber_pro' }]);
    assert.equal(raised.meta('OUROBOROS_RUNTIME_MODE'), 'Running: Pro · after restart: Cyber Pro', 'both values, from the read after the write');
    assert.equal(raised.line('OUROBOROS_RUNTIME_MODE'), 'Saved · applies after restart');
    const marks = raised.row('OUROBOROS_RUNTIME_MODE').querySelectorAll('[data-setting-mark]').map((mark) => mark.textContent);
    assert.deepEqual(marks, ['', '', 'Now', 'After restart'], 'the level in force and the chosen one, in words');
    assert.equal(raised.option('OUROBOROS_RUNTIME_MODE', 'cyber_pro').getAttribute('aria-checked'), 'true');
    // Lowering back to the level in force asks nothing, and nothing waits any more.
    raised.asked.length = 0;
    await raised.press(raised.option('OUROBOROS_RUNTIME_MODE', 'pro'));
    assert.deepEqual(raised.asked, []);
    assert.equal(raised.meta('OUROBOROS_RUNTIME_MODE'), 'How far it reaches into its own code and this computer.');
    assert.deepEqual(raised.row('OUROBOROS_RUNTIME_MODE').querySelectorAll('[data-setting-mark]').map((mark) => mark.textContent), ['', '', '', '']);
});

test('a choices group is one radio group: the arrows, Home and End move between its options and only a press chooses', async (t) => {
    const page = await open(t, 'permissions');
    const key = async (from, name) => {
        let prevented = false;
        await page.container.fire('keydown', { target: from, key: name, preventDefault: () => { prevented = true; } });
        return [page.document.activeElement?.dataset?.settingValue, prevented];
    };
    const option = (value) => page.option('OUROBOROS_RUNTIME_MODE', value);
    assert.deepEqual(await key(option('pro'), 'ArrowDown'), ['cyber_pro', true]);
    assert.deepEqual(await key(option('cyber_pro'), 'ArrowDown'), ['light', true], 'it wraps');
    assert.deepEqual(await key(option('light'), 'ArrowUp'), ['cyber_pro', true]);
    assert.deepEqual(await key(option('pro'), 'Home'), ['light', true]);
    assert.deepEqual(await key(option('pro'), 'End'), ['cyber_pro', true]);
    page.document.activeElement = null;
    assert.deepEqual(await key(option('pro'), 'a'), [undefined, false], 'any other key is left alone');
    // A strip is not a radio group: its buttons take no arrows.
    assert.deepEqual(await key(page.option('OUROBOROS_SAFETY_MODE', 'full'), 'ArrowDown'), [undefined, false]);
    assert.deepEqual(page.server.posts, [], 'moving the focus writes nothing');
    // The control carries no hover title and no inline style, like the strip.
    const html = settingsRow(rowById('OUROBOROS_RUNTIME_MODE'));
    assert.doesNotMatch(html, / title=| style=/);
});

test('lowering a Safety level asks first and raising one does not', async (t) => {
    const page = await open(t, 'permissions');
    await page.press(page.option('OUROBOROS_SAFETY_MODE', 'light'));
    assert.deepEqual(page.asked.map((dialog) => [dialog.title, dialog.danger]), [['Lower a safety level', true]]);
    assert.match(page.asked[0].body, /^Lower the safety level for work you gave it from \+ Commands to Outgoing\?/);
    assert.deepEqual(page.server.writes('/api/owner/safety-mode'), [{ mode: 'light' }]);
    assert.equal(page.line('OUROBOROS_SAFETY_MODE'), 'Saved · applies to new tasks · was + Commands');
    assert.equal(page.meta('OUROBOROS_SAFETY_MODE'), 'Messages, posts, skills, connectors and starting work are read first.', 'the line follows the level');
    // Raising: the other level goes from Outgoing to + Commands with no question.
    page.asked.length = 0;
    await page.press(page.option('OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'full'));
    assert.deepEqual(page.asked, []);
    assert.deepEqual(page.server.writes('/api/owner/safety-mode')[1], { mode: 'full', kind: 'consciousness' });
    // A lowering the owner declines sends nothing and leaves the level.
    const declined = await open(t, 'permissions', { answer: false });
    await declined.press(declined.option('OUROBOROS_SAFETY_MODE', 'off'));
    assert.equal(declined.asked.length, 1);
    assert.deepEqual(declined.server.writes('/api/owner/safety-mode'), []);
    assert.equal(declined.option('OUROBOROS_SAFETY_MODE', 'full').getAttribute('aria-pressed'), 'true');
});

test('Permissions and checks: three questions and what it is trusted with, doors that say their fact, one fold, no hover title', async (t) => {
    const page = await open(t, 'permissions', { values: { OUROBOROS_MODEL_SAFETY: 'anthropic::claude-opus-5.5' } });
    assert.deepEqual(page.container.querySelectorAll('.settings-card-title').map((title) => title.textContent),
        ['What it may change', 'Who watches its actions', 'Who checks its results', 'Trust it without asking']);
    assert.deepEqual(page.container.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow), [
        'OUROBOROS_RUNTIME_MODE', 'OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'permissions:rules', 'permissions:supervisor',
        'permissions:second-check', 'OUROBOROS_TASK_REVIEW_MODE', 'permissions:reviewers',
        'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', 'OUROBOROS_REVIEW_MAX_CYCLES']);
    const folds = page.container.querySelectorAll('details[data-settings-fold]');
    assert.equal(folds.length, 1);
    assert.equal(folds[0].querySelector('summary').textContent, 'More: review rounds');
    assert.deepEqual(folds[0].querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow), ['OUROBOROS_REVIEW_MAX_CYCLES']);
    assert.equal(page.meta('permissions:supervisor'), 'claude-opus-5.5');
    assert.equal(page.meta('permissions:second-check'), 'Off');
    assert.equal(page.container.querySelector('[data-safety-skips]').textContent, 'No safety checks let through in the last 24 h.',
        'the counter is read when the page is shown');
    assert.equal(page.container.querySelector('[data-safety-spares]').textContent,
        'No safety check went to a spare model in the last 24 h. 0 checks with a recorded verdict in the same span.',
        'the spare-model line is read from the same tail');
    for (const [id, to] of [['permissions:rules', 'rules'], ['permissions:supervisor', 'models/role/safety'], ['permissions:second-check', 'agents'],
        ['permissions:reviewers', 'agents']]) {
        await page.press(page.row(id));
        assert.equal(page.opened.at(-1), to, id);
    }
    // Turning the grant on asks first; turning it off does not.
    await page.press(page.row('OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS'));
    const grant = page.row('OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS').querySelector('[data-setting-control]');
    grant.checked = true;
    await page.container.fire('change', { target: grant });
    await settle();
    assert.equal(page.asked.at(-1).title, 'Give reviewed skills their keys?');
    assert.deepEqual(page.server.writes('/api/owner/auto-grant'), [{ enabled: true }]);
    assert.doesNotMatch(settingsPageSource(), / title="/, 'what a row needs to say is in its meta or its result line');
});

const settingsPageSource = () => fs.readFileSync(new URL('../modules/settings_ui.js', import.meta.url), 'utf8')
    + fs.readFileSync(new URL('../modules/settings_safety.js', import.meta.url), 'utf8');

test('what a managed deployment locks is said in its row, and an unmanaged one leaves the rows as described', () => {
    const rows = () => rowsOf('permissions').map((row) => structuredClone(row));
    const row = (list, id) => list.find((each) => each.id === id);
    const managed = tunePermissionRows(rows(), { OUROBOROS_MODEL_SAFETY: '', _meta: { policy_state: {
        access: { configured: 'pro', current_process: 'pro' },
        deployment: { access_max: 'pro', min_coverage: 'full', checker: { model: 'anthropic@safety::checker' } } } } });
    assert.equal(row(managed, 'OUROBOROS_RUNTIME_MODE').meta, 'Your organization allows up to Pro');
    assert.deepEqual(row(managed, 'OUROBOROS_RUNTIME_MODE').control.options.map((option) => option.value), ['light', 'advanced', 'pro']);
    assert.equal(row(managed, 'OUROBOROS_SAFETY_MODE').meta, 'Set by your organization: + Commands');
    assert.equal(row(managed, 'permissions:supervisor').meta, 'checker · Set by your organization');
    assert.deepEqual([row(managed, 'permissions:supervisor').control.kind, row(managed, 'permissions:second-check').control.kind], ['custom', 'custom'],
        'an organization\'s Supervisor is said, never offered for change');
    // Cyber Pro in force and saved, unmanaged: the Supervisor stays a door and no review row is drawn.
    const cyber = tunePermissionRows(rows(), { _meta: { policy_state: { access: { configured: 'cyber_pro', current_process: 'cyber_pro' } } } });
    assert.equal(row(cyber, 'permissions:supervisor').control.kind, 'door');
    assert.equal(row(cyber, 'OUROBOROS_REVIEW_ENFORCEMENT'), undefined);
    // Unmanaged and not Cyber Pro: the rows as described.
    const plain = tunePermissionRows(rows(), { _meta: { policy_state: { access: { configured: 'pro', current_process: 'pro' } } } });
    assert.deepEqual(row(plain, 'OUROBOROS_TASK_REVIEW_MODE').control, rowById('OUROBOROS_TASK_REVIEW_MODE').control);
    assert.deepEqual(row(plain, 'OUROBOROS_RUNTIME_MODE').control, rowById('OUROBOROS_RUNTIME_MODE').control);
});

test('Permissions and checks: under an organization\'s Supervisor the spare-model line is hidden; the ordinary page shows it', async (t) => {
    const page = await open(t, 'permissions');
    const spares = () => page.container.querySelector('[data-safety-spares]');
    assert.equal(spares().hidden, false);
    assert.match(spares().textContent, /^No safety check went to a spare model in the last 24 h\. /);
    // Its checker lists a route of its own without a credential before any ask, and its own answer is then a walk's.
    const ordinary = page.server.document;
    page.server.document = () => {
        const document = ordinary();
        document._meta.policy_state.deployment = { access_max: 'pro', min_coverage: 'full', checker: { model: 'anthropic@safety::checker' } };
        return document;
    };
    await page.shown();
    assert.equal(page.meta('permissions:supervisor'), 'checker · Set by your organization');
    assert.equal(spares().hidden, true, 'no count of the checker\'s own answers as a spare\'s');
    page.server.document = ordinary;
    await page.shown();
    assert.equal(spares().hidden, false);
});

test('Your rules: its own Save, the server\'s refusal on the page with the text kept, and what a saved text applies to', async (t) => {
    const page = await open(t, 'rules');
    const field = page.container.querySelector('#settings-rules');
    const save = page.container.querySelector('[data-editor-save]');
    const result = page.container.querySelector('[data-editor-result]');
    const type = async (text) => { field.value = text; await field.fire('input', { target: field }); };
    assert.equal(field.value, 'Ask before paying.');
    assert.match(page.container.textContent, /Running work keeps the text it started with\./);
    await type('Ask before paying.\nNever email my clients.');
    assert.deepEqual(page.server.posts, [], 'typing sends nothing');
    page.server.refuse.set('/api/owner/safety-rules', [{ error: 'Rules are limited to 64,000 bytes.', saved: false }, 400]);
    await save.fire('click', { target: save });
    await settle();
    assert.equal(result.textContent, 'Not saved. Rules are limited to 64,000 bytes.');
    assert.equal(result.dataset.tone, 'danger');
    assert.equal(field.value, 'Ask before paying.\nNever email my clients.', 'the refused text stays in the field');
    assert.equal(page.server.values.OUROBOROS_SAFETY_OWNER_RULES, 'Ask before paying.');
    // A refused text survives leaving the page, and says it is not saved.
    page.handle.unmount();
    const again = new NodeStub();
    const handle = await settingsPageFor('rules').mount(again, { path: 'rules', row: '', ctx: { readSettings } });
    await settle();
    assert.equal(again.querySelector('#settings-rules').value, 'Ask before paying.\nNever email my clients.');
    assert.equal(again.querySelector('[data-editor-result]').textContent, 'Not saved yet');
    await again.querySelector('[data-editor-save]').fire('click', {});
    await settle();
    assert.equal(again.querySelector('[data-editor-result]').textContent, 'Saved · applies from the next task or turn');
    assert.deepEqual(page.server.writes('/api/owner/safety-rules').at(-1), { rules: 'Ask before paying.\nNever email my clients.' });
    handle.unmount();
    // The words of every outcome: an unknown one is never called saved.
    assert.deepEqual(rulesSaveWords({ state: 'unknown', message: 'timeout' }), { text: 'Not confirmed yet', tone: 'warn', details: '' });
    assert.equal(rulesSaveWords({ state: 'saved_then_failed', message: 'The audit step failed afterwards.' }).text, 'Saved. The audit step failed afterwards.');
});

test('Notifications keeps the notifier\'s own switches, as switches that act at once, and its two status lines', async (t) => {
    const page = await open(t, 'notifications');
    const prefs = page.container.querySelectorAll('[data-notify-pref]');
    assert.deepEqual(prefs.map((input) => input.dataset.notifyPref), ['enabled', 'needs_answer', 'task_done', 'important', 'main_reply', 'sound', 'show_text']);
    for (const input of prefs) assert.deepEqual([input.getAttribute('role'), input.parentElement.className], ['switch', 'ui-switch']);
    assert.equal(page.container.querySelectorAll('.ui-checkbox').length, 0);
    for (const hook of ['[data-notify-volume]', '[data-notify-volume-value]', '[data-notify-preview]', '[data-notify-test]']) {
        assert.equal(page.container.querySelectorAll(hook).length, 1, hook);
    }
    // The first row's line is the notifier's sentence about this system; the test row's is its attention line.
    assert.ok(page.row('notify:enabled').querySelector('[data-setting-meta]').hasAttribute('data-notify-status'));
    assert.match(page.meta('notify:enabled'), /Notifications are off|This device blocks storage/);
    assert.ok(page.row('notify:test').querySelector('[data-setting-meta]').hasAttribute('data-notify-attention-status'));
    assert.deepEqual(page.container.querySelectorAll('.settings-card-title').map((title) => title.textContent), ['Notify me about', 'How']);
    await page.container.querySelector('[data-notify-doors]').fire('click', { target: page.row('notify:places') });
    assert.deepEqual(page.opened, ['page:circles'], 'what a place is sent is set in Circles');
    assert.deepEqual(page.server.posts, [], 'nothing of this page is a server setting');
});

test('Voice and calls says the saved choices and opens the form of whoever registered a live conversation', async (t) => {
    const hub = ({ engine = { key: '' }, reader = null, voice = '' }) => ({ selected: () => engine, readAloud: () => reader, voiceId: () => voice });
    const store = (items) => ({ providers: (kind) => items.filter((item) => item.provides === kind) });
    const title = (skill) => ({ some_call_skill: 'Calls (beta)' }[skill] || skill);
    assert.deepEqual(voiceFacts({ speech: hub({}), store: store([]), forms: [], title }),
        { dictation: 'Not chosen', dictationNote: '', readAloud: 'No voice on this device', readAloudNote: '', calls: [] });
    const voices = [{ id: 'milena', label: 'Milena' }];
    const talking = store([{ provides: 'conversation', skill: 'some_call_skill', skill_label: 'some_call_skill', label: 'Live Conversation' },
        { provides: 'conversation', skill: 'some_call_skill', label: 'A second entry of the same skill' }, { provides: 'stt', skill: 'whisper-voice' },
        { provides: 'conversation', skill: 'formless_skill', label: 'Talks, with no form of its own' }]);
    const facts = voiceFacts({
        speech: hub({ engine: { key: 'whisper-voice:stt', label: 'Whisper', available: true }, voice: 'milena',
            reader: { key: 'device:tts', label: 'This device', available: true, impl: { voices: () => voices } } }),
        store: talking, forms: [{ skill: 'some_call_skill' }, { skill: 'whisper-voice' }], title,
    });
    assert.deepEqual([facts.dictation, facts.readAloud], ['Whisper', 'Milena · This device']);
    assert.deepEqual(facts.calls, [{ skill: 'some_call_skill', name: 'Calls (beta)' }],
        'one door per skill that registered a live conversation and a form, named by the form\'s own title');
    assert.deepEqual(voiceFacts({ speech: hub({}), store: talking, forms: [], title }).calls, [], 'no form known, no door to open');
    // An engine that cannot run here says so, with the reason as the row's line.
    const broken = voiceFacts({ speech: hub({ engine: { key: 'x:stt', label: 'Whisper', available: false, reason: 'The chosen engine has not started.' },
        reader: { key: 'y:tts', label: 'Cloud voice', available: false, reason: 'Read aloud is unavailable here.' } }), store: store([]), forms: [], title });
    assert.deepEqual([broken.dictation, broken.dictationNote, broken.readAloud, broken.readAloudNote],
        ['Whisper · not ready here', 'The chosen engine has not started.', 'Cloud voice · not ready here', 'Read aloud is unavailable here.']);
    // The page has no control of its own, and core code names no skill: the door's target comes from the two stores.
    const page = await open(t, 'voice');
    assert.deepEqual(page.container.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow), ['voice:dictation', 'voice:read-aloud']);
    assert.equal(page.container.querySelectorAll('[data-setting-control], input, select').length, 0);
    assert.equal(page.meta('voice:dictation'), 'Chosen at the message field: +, then Add dictation.');
    // Each row says its value and leads to Main, where the choice is made; a row of another page stays a row.
    for (const id of ['voice:dictation', 'voice:read-aloud']) {
        assert.equal(page.row(id).dataset.kind, 'door', id);
        assert.ok(page.row(id).querySelector(`[data-fact="${id}"]`), `${id} says its value`);
        await page.press(page.row(id));
        assert.equal(page.opened.at(-1), 'main:context', id);
    }
    assert.ok(page.calls.includes('/api/extensions'), 'the skills are read when the page is shown');
    assert.doesNotMatch(settingsPageSource(), /live_conversation|skill\/[a-z]/, 'no skill id is written in the page');
});

test('This computer: the host\'s two switches, who can reach it with the password in the opening write, and doors to its pages', async (t) => {
    const page = await open(t, 'computer');
    assert.deepEqual(page.container.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow), ['computer:autostart',
        'OUROBOROS_DESKTOP_KEEP_RUNNING', 'OUROBOROS_SERVER_HOST', 'OUROBOROS_NETWORK_PASSWORD', 'computer:address', 'computer:advanced', 'computer:about']);
    // The host switches are the host's: read from it, each saying its state in its own row.
    const startup = page.row('computer:autostart');
    assert.equal(page.container.querySelector('[data-autostart-settings]').hidden, false);
    assert.equal(startup.querySelector('[data-autostart-toggle]').checked, true);
    assert.ok(startup.querySelector('[data-setting-result]').hasAttribute('data-autostart-status'));
    assert.equal(page.row('OUROBOROS_DESKTOP_KEEP_RUNNING').hidden, false);
    assert.equal(page.row('OUROBOROS_DESKTOP_KEEP_RUNNING').querySelector('[data-background-toggle]').checked, false);
    // Only this computer reaches it: no address row, and the password row says none is set.
    assert.deepEqual(page.row('OUROBOROS_SERVER_HOST').querySelectorAll('[data-setting-value]').map((b) => [b.dataset.settingValue, b.textContent]),
        [['127.0.0.1', 'Only here'], ['0.0.0.0', 'My network']]);
    assert.equal(page.meta('OUROBOROS_SERVER_HOST'), 'Only this computer.');
    assert.equal(page.row('computer:address').hidden, true);
    assert.match(page.meta('OUROBOROS_NETWORK_PASSWORD'), /^Not set\./);
    // The password typed here travels with the opening write, which asks first.
    const password = page.row('OUROBOROS_NETWORK_PASSWORD').querySelector('[data-setting-control]');
    password.value = 'correct horse battery';
    await page.container.fire('input', { target: password });
    await page.container.events.get('change').reduce((done, fn) => done.then(() => fn({ target: password })), Promise.resolve());
    await settle();
    await page.press(page.option('OUROBOROS_SERVER_HOST', '0.0.0.0'));
    assert.equal(page.asked.at(-1).title, 'Open to the network');
    const hostWrites = page.server.writes('/api/settings').filter((body) => 'OUROBOROS_SERVER_HOST' in body);
    assert.deepEqual(hostWrites, [{ OUROBOROS_SERVER_HOST: '0.0.0.0', OUROBOROS_NETWORK_PASSWORD: 'correct horse battery' }]);
    assert.equal(page.meta('OUROBOROS_SERVER_HOST'), 'Running: Only here · after restart: My network');
    assert.equal(page.line('OUROBOROS_SERVER_HOST'), 'Saved · applies after restart');
    assert.equal(page.meta('OUROBOROS_NETWORK_PASSWORD'), 'Asked on every other device.');
    assert.equal(page.row('OUROBOROS_NETWORK_PASSWORD').querySelector('[data-secret-state]').textContent, 'Saved');
    // Closing it again asks nothing and carries no password.
    page.asked.length = 0;
    await page.press(page.option('OUROBOROS_SERVER_HOST', '127.0.0.1'));
    assert.deepEqual(page.asked, []);
    assert.deepEqual(page.server.writes('/api/settings').at(-1), { OUROBOROS_SERVER_HOST: '127.0.0.1' });
    for (const [id, to] of [['computer:advanced', 'computer/advanced'], ['computer:about', 'computer/about']]) {
        await page.press(page.row(id));
        assert.equal(page.opened.at(-1), to);
    }
});

test('a stored network password is a stored key\'s row: Replace, Show the stored value, Remove asked first; a refusal is said', async (t) => {
    const page = await open(t, 'computer', { values: { OUROBOROS_NETWORK_PASSWORD: 'old horse' } });
    const part = (hook, at = page) => at.row('OUROBOROS_NETWORK_PASSWORD').querySelector(hook);
    assert.deepEqual([part('[data-secret-state]').textContent, part('[data-secret-open]').textContent], ['Saved', 'Replace']);
    assert.equal(part('[data-secret-field]').hidden, true);
    assert.equal(part('[data-setting-control]').value, '', 'the stored value is never put into the field');
    // Replace opens the field: Show and Remove stand beside it. Show reads the stored value on the press.
    await page.press(part('[data-secret-open]'));
    assert.equal(part('[data-secret-field]').hidden, false);
    assert.deepEqual([part('[data-secret-show]').hidden, part('[data-secret-remove]').hidden], [false, false]);
    await part('[data-secret-show]').fire('click', {});
    await settle();
    assert.deepEqual(page.server.revealed, [{ key: 'OUROBOROS_NETWORK_PASSWORD' }]);
    assert.equal(part('.settings-secret-value').textContent, 'old horse');
    // A typed new one replaces it, says Saved alone, offers no Undo, and puts the field away.
    const field = part('[data-setting-control]');
    field.value = 'new horse';
    await page.container.fire('input', { target: field });
    await page.container.fire('change', { target: field });
    await settle();
    assert.deepEqual(page.server.writes('/api/settings'), [{ OUROBOROS_NETWORK_PASSWORD: 'new horse' }]);
    assert.equal(page.server.values.OUROBOROS_NETWORK_PASSWORD, 'new horse');
    assert.equal(page.line('OUROBOROS_NETWORK_PASSWORD'), 'Saved');
    assert.equal(part('[data-setting-undo]'), null);
    assert.equal(part('[data-secret-field]').hidden, true);

    // Remove asks first; declined, nothing is sent.
    const declined = await open(t, 'computer', { values: { OUROBOROS_NETWORK_PASSWORD: 'new horse' }, answer: false });
    await declined.press(part('[data-secret-open]', declined));
    await declined.press(part('[data-secret-remove]', declined));
    assert.deepEqual(declined.asked.map((dialog) => dialog.title), ['Remove Password?']);
    assert.deepEqual(declined.server.posts, []);
    // Still open to the network, the server refuses to clear it: its sentence is the row's line, and the password stays.
    const shut = await open(t, 'computer', { values: { OUROBOROS_NETWORK_PASSWORD: 'new horse' } });
    const refusal = 'Cannot clear Network Password while the running server is still bound to a non-localhost interface.';
    await shut.press(part('[data-secret-open]', shut));
    shut.server.refuse.set('/api/settings', [{ saved: false, error: refusal }, 400]);
    await shut.press(part('[data-secret-remove]', shut));
    assert.deepEqual(shut.server.writes('/api/settings'), [{ OUROBOROS_NETWORK_PASSWORD: '' }], 'Remove writes the empty value');
    assert.equal(shut.line('OUROBOROS_NETWORK_PASSWORD'), `Not saved. ${refusal}`);
    assert.equal(shut.server.values.OUROBOROS_NETWORK_PASSWORD, 'new horse');
    // Accepted, it is cleared, and the row offers Add.
    await shut.press(part('[data-secret-remove]', shut));
    assert.equal(shut.server.values.OUROBOROS_NETWORK_PASSWORD, '');
    assert.equal(shut.line('OUROBOROS_NETWORK_PASSWORD'), 'Saved');
    assert.deepEqual([part('[data-secret-open]', shut).textContent, part('[data-secret-remove]', shut).hidden], ['Add', true]);
    assert.match(shut.meta('OUROBOROS_NETWORK_PASSWORD'), /^Not set\./);
});

test('Advanced holds what an installation rarely changes, and About the reset in its fold, asked first', async (t) => {
    const advanced = await open(t, 'computer/advanced');
    assert.match(advanced.container.textContent, /^Most installations never change these\./);
    assert.deepEqual(advanced.container.querySelectorAll('[data-setting-row]').map((row) => row.dataset.settingRow),
        ['OUROBOROS_MAX_WORKERS', 'OUROBOROS_UI_DIR', 'OUROBOROS_EXTRA_CA_BUNDLE']);
    await advanced.shown();
    assert.equal(advanced.meta('OUROBOROS_UI_DIR'), 'Empty serves the built-in interface.');
    const served = advanced.row('OUROBOROS_UI_DIR').querySelector('[data-served-ui]');
    assert.deepEqual([served.textContent, served.dataset.tone], ['Serving now: the built-in interface.', 'muted'],
        'what the core serves now stands in the row, after its line');
    // A folder the core refused is a warning with the core's own sentence.
    advanced.routes.set('/api/state', response({ served_ui: { source: 'builtin', error: "OUROBOROS_UI_DIR '/srv/ui' was not applied: no index.html" } }));
    await advanced.shown();
    assert.deepEqual([served.textContent, served.dataset.tone],
        ["Serving now: the built-in interface. OUROBOROS_UI_DIR '/srv/ui' was not applied: no index.html", 'warn']);
    // An empty value is said in the row's own word, in the line that says what it replaced and in the two-value meta.
    const type = async (id, text) => {
        const field = advanced.row(id).querySelector('[data-setting-control]');
        field.value = text;
        await advanced.container.fire('input', { target: field });
        await advanced.container.fire('change', { target: field });
        await settle();
    };
    await type('OUROBOROS_EXTRA_CA_BUNDLE', '/etc/ssl/extra.pem');
    assert.equal(advanced.line('OUROBOROS_EXTRA_CA_BUNDLE'), 'Saved · applies to new tasks · was none');
    await type('OUROBOROS_UI_DIR', '/srv/ui');
    assert.equal(advanced.line('OUROBOROS_UI_DIR'), 'Saved · applies after restart');
    assert.equal(advanced.meta('OUROBOROS_UI_DIR'), 'Running: the built-in interface · after restart: /srv/ui');
    // The cap on acting tasks is optional and acts at once: the server's blank is the row's No limit, both ways.
    const cap = advanced.row('OUROBOROS_MAX_WORKERS');
    const noLimit = cap.querySelector('[data-setting-no-limit]');
    assert.deepEqual([cap.querySelector('[data-setting-control]').value, noLimit.getAttribute('aria-pressed')], ['', 'true']);
    await type('OUROBOROS_MAX_WORKERS', '4');
    assert.deepEqual(advanced.server.writes('/api/settings').at(-1), { OUROBOROS_MAX_WORKERS: 4 });
    assert.equal(advanced.line('OUROBOROS_MAX_WORKERS'), 'Saved · was no limit');
    assert.equal(noLimit.getAttribute('aria-pressed'), 'false');
    await advanced.press(noLimit);
    assert.deepEqual(advanced.server.writes('/api/settings').at(-1), { OUROBOROS_MAX_WORKERS: '' }, 'No limit is sent as the server\'s blank');
    assert.equal(advanced.line('OUROBOROS_MAX_WORKERS'), 'Saved · was 4');
    assert.deepEqual([cap.querySelector('[data-setting-control]').value, noLimit.getAttribute('aria-pressed')], ['', 'true']);
    const about = await open(t, 'computer/about');
    assert.ok(about.container.querySelector('#about-version'));
    const fold = about.container.querySelector('details[data-settings-fold]');
    assert.equal(fold.querySelector('summary').textContent, 'More: reset all data');
    assert.ok(fold.querySelector('[data-reset-all]'));
    assert.deepEqual(about.server.posts, []);
});

test('the home opens every one of these pages by the path its row names', () => {
    for (const path of ['rules', 'permissions', 'appearance', 'notifications', 'voice', 'computer']) {
        assert.ok(HOME_ROWS.some((row) => row.to === path), path);
        assert.equal(settingsPageFor(path).prefix, path);
    }
    assert.equal(settingsPageFor('computer/advanced').prefix, 'computer/advanced', 'the longest prefix serves a page under This computer');
});

test('a switch a page draws by hand is stamped as a switch, so a narrow row keeps it at its end', async (t) => {
    // settings.css keeps a door, a switch, a stored key and a short number at a narrow row's end; a custom row's
    // control goes under its words. A page that draws its own switch says so in the row's description (`as`).
    const notifications = await open(t, 'notifications');
    for (const id of ['notify:enabled', 'notify:needs_answer', 'notify:task_done', 'notify:important', 'notify:main_reply', 'notify:sound', 'notify:show_text']) {
        assert.equal(notifications.row(id).dataset.kind, 'switch', id);
    }
    assert.equal(notifications.row('notify:test').dataset.kind, 'custom', 'a button of a page stays a custom row');
    const computer = await open(t, 'computer');
    for (const id of ['computer:autostart', 'OUROBOROS_DESKTOP_KEEP_RUNNING']) assert.equal(computer.row(id).dataset.kind, 'switch', id);
    const css = fs.readFileSync(new URL('../settings.css', import.meta.url), 'utf8');
    assert.match(css, /@container \(max-width: 28rem\) \{\s*\.settings-row:is\(\[data-kind="door"\], \[data-kind="switch"\]/);
});

test('Permissions and checks: the checks a spare model answered are counted by model, beside the misses and the checks with a verdict', () => {
    const walk = (model) => ({ type: 'safety_route_fallback', answered: model ? { model, role: 'fallback:0' } : null });
    assert.equal(spareWords([], 7, 'in the last 24 h'),
        'No safety check went to a spare model in the last 24 h. 7 checks with a recorded verdict in the same span.');
    assert.equal(spareWords([walk('openrouter@openrouter-anthropic::anthropic/claude-opus-5.5'), walk('anthropic/claude-opus-5.5'),
        walk('anthropic/claude-fable-5.2'), walk(null)], 412, 'since 09:12 (the latest 2,000 events)'),
    '3 safety checks were answered by a spare model since 09:12 (the latest 2,000 events), because the Supervisor\'s own'
        + ' model gave no answer: anthropic/claude-opus-5.5 ×2, anthropic/claude-fable-5.2 ×1. 1 check went to the spare'
        + ' models and got no answer. 412 checks with a recorded verdict in the same span.');
    assert.equal(spareWords([walk(null)], 1, 'in the last 24 h'),
        '1 check went to the spare models in the last 24 h and got no answer. 1 check with a recorded verdict in the same span.');
});
