// The home test of web/modules/ui_primitives.js, the portable leaf served to author kits with ui.css: the five
// field/status exports keep their signatures, the result-of-an-action helpers hold DESIGN "Status and chips"
// lifetime in both directions, and the shared button role and result/action composition have one owner. The Costs
// budget save (the first consumer of the lifetime rule) is checked here on the real panel.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { readFile } from 'node:fs/promises';
import * as portable from '../modules/ui_primitives.js';
import * as existing from '../modules/ui_helpers.js';
import { escapeHtmlAttr } from '../modules/utils.js';
import { budgetSavePresentation, initCosts } from '../modules/costs.js';
import { setup } from './dashboard_dom_fixture.js';
import { domDouble } from './dom_double.js';

const settingsCss = await readFile(new URL('../settings.css', import.meta.url), 'utf8');
const onboardingCss = await readFile(new URL('../onboarding.css', import.meta.url), 'utf8');
const sharedCss = await readFile(new URL('../ui.css', import.meta.url), 'utf8');
const documents = await Promise.all(['index.html', 'onboarding_template.html'].map(name => readFile(new URL('../' + name, import.meta.url), 'utf8')));
const settingsJs = await readFile(new URL('../modules/settings.js', import.meta.url), 'utf8');
const settingsUiJs = await readFile(new URL('../modules/settings_ui.js', import.meta.url), 'utf8');
const controlsJs = await readFile(new URL('../modules/settings_controls.js', import.meta.url), 'utf8');
const connectionsJs = await readFile(new URL('../modules/settings_connections.js', import.meta.url), 'utf8');
const catalogJs = await readFile(new URL('../modules/settings_catalog.js', import.meta.url), 'utf8');
const primitivesJs = await readFile(new URL('../modules/ui_primitives.js', import.meta.url), 'utf8');

// ---- the leaf -------------------------------------------------------------------------------------------------------

test('host consumers and author kit share the same portable functions', () => {
    for (const name of ['renderSafeField', 'collectSafeFieldValues', 'normalizeTone', 'setInlineStatus']) {
        assert.equal(existing[name], portable[name]);
    }
    assert.equal(escapeHtmlAttr, portable.escapeHtmlAttr);
});

test('portable source imports in an isolated document without host dependencies', async () => {
    const source = fs.readFileSync(new URL('../modules/ui_primitives.js', import.meta.url), 'utf8');
    assert.doesNotMatch(source, /^\s*import\b/m);
    const exported = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
    assert.equal(exported.normalizeTone('error'), 'danger');
    const result = exported.renderSafeField({ name: 'city', label: '<City>', default: '" value' });
    assert.match(result, /&lt;City&gt;/);
    assert.match(result, /value="&quot; value"/);
    assert.match(result, /class="ui-control"/);
    // The additions are exported from the same leaf and nothing in it touches document or window at import.
    for (const name of ['setResult', 'clearResultOnEdit', 'errorWords', 'SAVE_ACK_MS']) assert.ok(name in exported, name);
    assert.doesNotMatch(source, /^\s*(document|window)\./m);
});

test('field rendering escapes values and preserves secret/non-secret semantics', () => {
    const field = portable.renderSafeField({ name: 'token', type: 'password', label: 'Token', default: 'not-a-default', help: '<help>' }, { token: 'not-a-draft' });
    assert.match(field, /type="password"/);
    assert.match(field, /value=""/);
    assert.doesNotMatch(field, /not-a-/);
    assert.match(field, /&lt;help&gt;/);
    const textarea = portable.renderSafeField({ name: 'text', type: 'textarea' }, { text: '</textarea><script>x</script>' });
    assert.doesNotMatch(textarea, /<script>/);
    assert.match(textarea, /&lt;\/textarea&gt;/);
    const select = portable.renderSafeField({ name: 'source', type: 'select', options: ['one', 'two'] }, { source: 'two' });
    assert.match(select, /value="two" selected/);
    const checkbox = portable.renderSafeField({ name: 'enabled', type: 'checkbox' }, { enabled: true });
    assert.match(checkbox, /class="ui-checkbox"/);
    assert.match(checkbox, / checked/);
});

test('collection follows field type and never silently persists passwords when excluded', () => {
    const inputs = { enabled: { checked: false, value: 'on' }, n: { value: '2' }, key: { value: 'secret' } };
    const form = { elements: { namedItem: name => inputs[name] } };
    const fields = [{ name: 'enabled', type: 'checkbox' }, { name: 'n', type: 'number' }, { name: 'key', type: 'password' }];
    assert.deepEqual(portable.collectSafeFieldValues(form, fields), { enabled: false, n: '2', key: 'secret' });
    assert.deepEqual(portable.collectSafeFieldValues(form, fields, { includePasswords: false }), { enabled: false, n: '2' });
});

test('status updates text and tone without replacing the host or resetting identical text', () => {
    let writes = 0; let text = 'ready';
    const el = { dataset: {}, get textContent() { return text; }, set textContent(v) { text = v; writes++; } };
    portable.setInlineStatus(el, 'ready', 'success');
    assert.equal(writes, 0); assert.equal(el.dataset.tone, 'ok');
    portable.setInlineStatus(el, 'failed', 'error');
    assert.equal(writes, 1); assert.equal(el.textContent, 'failed'); assert.equal(el.dataset.tone, 'danger');
});

test('safe controls name the field independently of options, help and secret values', () => {
    for (const type of ['text', 'textarea', 'select', 'checkbox']) {
        const html = portable.renderSafeField({ name: 'view', label: 'View', type, options: ['List', 'Grid'], help: 'Choose <one>' });
        assert.match(html, /aria-label="View"/);
        assert.match(html, /aria-description="Choose &lt;one&gt;"/);
    }
});

// ---- the result of an action -----------------------------------------------------------------------------------------

const resultNode = () => {
    const { el } = domDouble();
    const subject = el('form');
    const node = el('span', { className: 'ui-result' });
    subject.append(node);
    return { el, subject, node, words: () => [node.textContent, node.dataset.tone ?? ''] };
};

test('setResult writes the words and the tone, keeps raw text behind one Details disclosure, and an empty result clears', () => {
    const { node } = resultNode();
    portable.setResult(node, { text: "Couldn't save.", tone: 'error', details: '<script>raw</script> 500' });
    assert.equal(node.dataset.tone, 'danger');
    assert.match(node.textContent, /^Couldn't save\./);
    const disclosure = node.children.find((child) => child.tagName.toLowerCase() === 'details');
    assert.equal(disclosure.className, 'ui-disclosure');
    assert.deepEqual(disclosure.children.map((child) => [child.tagName.toLowerCase(), child.textContent]), [['summary', 'Details'], ['pre', '<script>raw</script> 500']]);
    assert.equal(disclosure.children[1].className, 'ui-result-details');
    portable.setResult(node, { text: 'Saved.', tone: 'ok' });
    assert.equal(node.children.length, 0, 'the previous details are gone with the previous result');
    portable.setResult(node, {});
    assert.deepEqual([node.textContent, node.dataset.tone], ['', undefined]);
});

test('a plain acknowledgement ends by itself after SAVE_ACK_MS; a warning, a restart or a failure never does', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const { node, words } = resultNode();
    portable.setResult(node, { text: 'Saved.', tone: 'ok', ends: true });
    t.mock.timers.tick(portable.SAVE_ACK_MS - 1);
    assert.deepEqual(words(), ['Saved.', 'ok'], 'still there just before it ends');
    t.mock.timers.tick(1);
    assert.deepEqual(words(), ['', ''], 'ended by itself, its tone with it');
    for (const [text, tone] of [['Saved. Restart required.', 'warn'], ['Could not save.', 'error']]) {
        portable.setResult(node, { text, tone, ends: true });
        t.mock.timers.tick(10 * portable.SAVE_ACK_MS);
        assert.deepEqual(words(), [text, portable.normalizeTone(tone)], `${tone} stays whatever the caller asked`);
    }
    portable.setResult(node, { text: 'Saved.', tone: 'ok' });
    t.mock.timers.tick(10 * portable.SAVE_ACK_MS);
    assert.deepEqual(words(), ['Saved.', 'ok'], 'without `ends` even an acknowledgement is a record');
});

test('one timer owner per node: an older acknowledgement never erases a newer result', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const { node, words } = resultNode();
    portable.setResult(node, { text: 'Saved.', tone: 'ok', ends: true });
    t.mock.timers.tick(portable.SAVE_ACK_MS - 1000);
    portable.setResult(node, { text: 'Offline.', tone: 'error' });
    t.mock.timers.tick(portable.SAVE_ACK_MS);
    assert.deepEqual(words(), ['Offline.', 'danger'], 'the first timer took nothing away');
    portable.setResult(node, { text: 'Saved.', tone: 'ok', ends: true });
    t.mock.timers.tick(portable.SAVE_ACK_MS - 1000);
    portable.setResult(node, { text: 'Saved again.', tone: 'ok', ends: true });
    t.mock.timers.tick(1000);
    assert.deepEqual(words(), ['Saved again.', 'ok'], 'the second acknowledgement counts its own full time');
    t.mock.timers.tick(portable.SAVE_ACK_MS - 1000);
    assert.deepEqual(words(), ['', '']);
});

test('an edit in the same subject clears the result; an edit inside the result or after disposal does not', () => {
    const { el, subject, node, words } = resultNode();
    const field = el('input');
    subject.append(field);
    const dispose = portable.clearResultOnEdit(subject, node);
    portable.setResult(node, { text: 'Saved. Restart required.', tone: 'warn' });
    node.emit('input');
    assert.deepEqual(words(), ['Saved. Restart required.', 'warn'], 'opening Details is not an edit');
    field.emit('input');
    assert.deepEqual(words(), ['', ''], 'the edit cleared the stale result');
    portable.setResult(node, { text: 'Saved.', tone: 'ok' });
    field.emit('change');
    assert.deepEqual(words(), ['', ''], 'a change clears too');
    portable.setResult(node, { text: 'Saved.', tone: 'ok' });
    dispose();
    field.emit('input');
    assert.deepEqual(words(), ['Saved.', 'ok'], 'disposed: the subject is no longer watched');
    assert.equal(typeof portable.clearResultOnEdit(null, node), 'function', 'no subject: a no-op disposer');
});

test('errorWords says the server\'s own reason when it gave one, generic words otherwise, the raw text as details', () => {
    const network = new TypeError('Failed to fetch');
    assert.deepEqual(portable.errorWords(network), { text: "Couldn't reach Ouroboros.", tone: 'danger', details: 'Failed to fetch' });
    const typedWithResponse = Object.assign(new TypeError('Failed to fetch'), { response: {} });
    assert.equal(portable.errorWords(typedWithResponse).text, "Didn't work.", 'a TypeError with a response is not a network failure');
    // An answer with a reason for a person: those are the words (a 409 is not guessed to be "changed elsewhere").
    const exists = { status: 409, body: { error: "project 'x' already exists — pick another name/id", error_code: 'project_exists' } };
    assert.deepEqual(portable.errorWords(exists), { text: "Project 'x' already exists — pick another name/id.", tone: 'danger',
        details: "project 'x' already exists — pick another name/id" });
    assert.equal(portable.errorWords({ status: 422, body: { error: 'Budget values must be at least 0.01.' } }).text,
        'Budget values must be at least 0.01.', 'one stop, never two');
    const refused = portable.errorWords({ status: 400, body: { reason_code: 'runtime_mode_protected', error: 'Protected path.' } });
    assert.deepEqual(refused, { text: 'Protected path.', tone: 'danger', details: 'runtime_mode_protected: Protected path.' });
    const importing = { status: 503, body: { error: 'the task store is importing; retry shortly', reason_code: 'migrating' } };
    assert.equal(portable.errorWords(importing).text, 'The task store is importing; retry shortly.', 'a "not yet" is not a refusal');
    assert.equal(portable.errorWords({ status: 503, body: { error: 'Offline.' } }).text, 'Offline.');
    const structured = { status: 400, body: { error: { code: 'bad_path', message: 'That folder does not exist' } } };
    assert.deepEqual(portable.errorWords(structured), { text: 'That folder does not exist.', tone: 'danger',
        details: '{"code":"bad_path","message":"That folder does not exist"}' });
    // An answer without a reason for a person: the generic words of its status (a code is not a sentence, nor is the
    // reader's placeholder for a body it could not read).
    assert.equal(portable.errorWords(Object.assign(new Error('x'), { status: 503 })).text, "Ouroboros couldn't do this.");
    assert.equal(portable.errorWords({ status: 503, body: { reason_code: 'migrating' } }).text, "Ouroboros couldn't do this.");
    assert.equal(portable.errorWords({ status: 502, body: { error: 'non-json response (HTTP 502)' } }).text, "Ouroboros couldn't do this.");
    assert.equal(portable.errorWords({ status: 409, body: { error: 'card_stale', reason_code: 'card_stale' } }).text, 'Ouroboros refused this.');
    const echo = { status: 503, body: { error: 'continue refused: queue_not_restored', reason_code: 'queue_not_restored' } };
    assert.equal(portable.errorWords(echo).text, "Ouroboros couldn't do this.", 'a line that repeats its code is not a reason');
    assert.equal(portable.errorWords({ ...echo, status: 409 }).text, 'Ouroboros refused this.');
    assert.equal(portable.errorWords({ status: 403 }).text, 'Not allowed here.');
    assert.equal(portable.errorWords({ status: 404 }).text, "Ouroboros couldn't find this.");
    assert.equal(portable.errorWords({ status: 409 }).text, "Ouroboros didn't accept this.");
    // The words never come from an error's own message: a message that talks of a network or a 404 changes nothing.
    assert.equal(portable.errorWords(new Error('network error: 404 not found')).text, "Didn't work.");
    assert.equal(portable.errorWords(new Error('network error: 404 not found')).details, 'network error: 404 not found');
    assert.equal(portable.errorWords({ status: 500, body: 'Internal Server Error' }).text, "Ouroboros couldn't do this.");
    assert.equal(portable.errorWords({ status: 500, body: 'Internal Server Error' }).details, 'Internal Server Error');
    assert.equal(portable.errorWords({ status: 500, body: { code: 7 } }).details, '{"code":7}');
    assert.equal(portable.errorWords(undefined).text, "Didn't work.");
});

// ---- the shared button role and the result/action composition --------------------------------------------------------

test('neutral controls have one shared button role in both UI shells', () => {
    assert.match(sharedCss, /\.btn-default[^{]*\{/);
    assert.match(sharedCss, /\.btn:focus-visible\s*\{/);
    // The app shell loads its main-interface /static; the wizard always loads the core's own tree.
    const [indexHtml, wizardHtml] = documents;
    assert.match(indexHtml, /href="\/static\/ui\.css"/);
    assert.match(wizardHtml, /href="\/onboarding\/static\/ui\.css"/);
    assert.doesNotMatch(onboardingCss, /(?:^|\n)\.btn-default\s*\{/);
    assert.match(sharedCss, /--control-h:\s*34px/);
    assert.doesNotMatch(settingsCss, /settings-ghost-btn/);
    assert.doesNotMatch(onboardingCss, /settings-ghost-btn/);
});

test('result/action composition has one shared owner, independent of page and viewport', async () => {
    // Real-container reflow and readability are exercised by the browser consumer.
    const pageCss = await readFile(new URL('../style.css', import.meta.url), 'utf8');
    // Settings composes the shared row (the footer, the catalog refresh, the LAN hint); the private row stays only for
    // the screens that still read it (Skills › Browse, Updates) and never reaches the shared sheet.
    assert.doesNotMatch(settingsUiJs, /settings-action-row/);
    assert.doesNotMatch(sharedCss, /\.settings-action-row\s*\{/);
    assert.match(settingsUiJs, /<div class="settings-footer ui-action-row" data-draft="clean">/);
    assert.doesNotMatch(pageCss, /\.skills-card-action-row\s*\{[^}]*(?:display:|grid-template|flex:)/);
    // The shared parts: the action row keeps a reserved result line, the result shares the status sentence's tones.
    assert.match(sharedCss, /\.ui-action-row\s*\{[^}]*flex-wrap:\s*wrap/);
    assert.match(sharedCss, /\.ui-result\s*\{[^}]*min-height:\s*calc\(var\(--type-meta\) \* var\(--line-meta\)\)/);
    assert.match(sharedCss, /\.ui-result\[data-tone="warn"\]/);
    assert.match(sharedCss, /details\.ui-disclosure > summary::before/);
    assert.match(sharedCss, /> \[data-primary\] ~ \[data-primary\] \{/);
});

test('async actions expose the same busy and status semantics', () => {
    // Save and every API key's Test share one busy helper and one status vocabulary.
    assert.match(controlsJs, /export function setButtonBusy\(button, busy\)/);
    assert.match(controlsJs, /setAttribute\('aria-busy', 'true'\)/);
    assert.match(settingsJs, /setButtonBusy\(saveButton, true\)/);
    assert.match(connectionsJs, /setButtonBusy\(button, true\)/);
    assert.match(connectionsJs, /setInlineStatus\(status, 'Testing…', 'muted'\)/);
    assert.match(connectionsJs, /providerTestStatusText\(data\), data\?\.ok \? 'ok' : 'danger'/);
    assert.match(settingsJs, /refreshModelCatalog\(\{ button: byId\('btn-refresh-model-catalog'\) \}\)/);
    assert.match(connectionsJs, /setInlineStatus\(row\.querySelector\('\[data-connection-test-status\]'\), '', 'muted'\)/);
    assert.match(primitivesJs, /if \(el\.textContent !== next\) el\.textContent = next/);
    // The catalog's result is the shared result line: its count ends by itself, a read gap stays.
    assert.match(catalogJs, /import \{ setResult \} from '\.\/ui_primitives\.js'/);
    assert.match(catalogJs, /setResult\(statusEl, \{ text, tone, ends: tone === 'ok' \}\)/);
    assert.doesNotMatch(catalogJs, /setInlineStatus/);
    assert.match(catalogJs, /refreshModelCatalog\(\{ button \} = \{\}\)/);
    assert.match(catalogJs, /buttonRefreshes\.get\(button\) === refreshSeq/);
});

// ---- the Costs budget save: the lifetime rule on the real panel -------------------------------------------------------
// A plain "saved" ends by itself; a restart still required, a warning and an error stay until the next save replaces
// them, and one timer owner means an older acknowledgement never erases a newer save's words or its tone.

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const settle = () => new Promise((resolve) => setImmediate(resolve));
const ACK = 4000;

test('a save says what happened and only a plain acknowledgement may end by itself', () => {
    const cases = [
        [{ no_changes: true }, 'No changes.', 'ok', true],
        [{ immediate_changed: true }, 'Saved. Took effect immediately.', 'ok', true],
        [{ next_task_changed: true }, 'Saved. Applies on the next task.', 'ok', true],
        [{ immediate_changed: true, next_task_changed: true },
            'Saved. Some changes took effect immediately; others apply on the next task.', 'ok', true],
        [{ restart_required: true }, 'Saved. Restart required.', 'warn', false],
        // Nothing changed, but a setting saved earlier still waits for a restart (gateway/settings.py restart_state).
        [{ no_changes: true, restart_required: true }, 'No changes. Restart still required.', 'warn', false],
        [{ immediate_changed: true, warnings: ['Limit is below known spend.', 'Cap lowered.'] },
            'Saved. Took effect immediately. Warnings: Limit is below known spend. | Cap lowered.', 'warn', false],
    ];
    for (const [data, text, tone, transient] of cases) {
        assert.deepEqual(budgetSavePresentation(data), { text, tone, transient }, JSON.stringify(data));
    }
    assert.equal(ACK, portable.SAVE_ACK_MS, 'the panel and the leaf agree on the acknowledgement time');
});

// The real panel, its real click handler and the shared HTTP client on the dashboard DOM fixture, with mocked time.
function costsPanel(t, answers) {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = setup(t);
    const posts = [];
    env.routes.set('/api/settings', async (_url, init) => {
        posts.push(JSON.parse(init.body));
        const next = answers.shift();
        if (next instanceof Error) throw next;
        return next;
    });
    initCosts({ mount: env.mount, state: {} });
    const status = env.mount.querySelector('#budget-save-status');
    const save = env.mount.querySelector('#btn-save-budget');
    env.mount.querySelector('#s-budget').value = '200';
    env.mount.querySelector('#s-per-task-cost').value = '50';
    const press = async () => { await save.fire('click', { currentTarget: save }); await settle(); };
    // The result's sentence; its raw text, if any, is the disclosure after it (setResult).
    const words = () => {
        const all = status.textContent;
        const details = status.querySelector('.ui-result-details');
        return [details ? all.slice(0, all.indexOf('Details')) : all, status.dataset.tone ?? ''];
    };
    const raw = () => status.querySelector('.ui-result-details')?.textContent ?? '';
    return { ...env, status, save, posts, press, words, raw, tick: (ms) => t.mock.timers.tick(ms) };
}

test('the budget form is the reference action row: Save, then its result as the next sibling in one row', (t) => {
    const panel = costsPanel(t, []);
    const row = panel.save.parentElement;
    assert.ok(row.classList.contains('ui-action-row'), 'Save stands in the shared action row');
    assert.equal(panel.save.nextElementSibling, panel.status, 'the result comes right after its button, never under it');
    assert.ok(panel.status.classList.contains('ui-result'));
    assert.equal(row.children.length, 2, 'nothing else sits between the button and its result');
    assert.equal(panel.save.textContent, 'Save');
});

test('a plain save acknowledges, then clears its words and its tone after the acknowledgement time', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true })]);
    await panel.press();
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately.', 'ok']);
    panel.tick(ACK - 1);
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately.', 'ok'], 'still there just before it ends');
    panel.tick(1);
    assert.deepEqual(panel.words(), ['', ''], 'the acknowledgement ended by itself, its tone with it');
    assert.equal(panel.save.disabled, false);
});

test('a restart the save requires, a warning and an error are never erased by time', async (t) => {
    const panel = costsPanel(t, [reply({ restart_required: true }), reply({ no_changes: true, restart_required: true }),
        reply({ immediate_changed: true, warnings: ['Limit is below known spend.'] }),
        reply({ error: 'Settings file is read-only.' }, 500)]);
    await panel.press();
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn']);
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn'], 'the restart stays required, so its words stay');
    await panel.press();  // the same values again: nothing changes, and the restart is still owed
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['No changes. Restart still required.', 'warn'], 'it says why it stays');
    await panel.press();
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Saved. Took effect immediately. Warnings: Limit is below known spend.', 'warn']);
    await panel.press();
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Not saved. Settings file is read-only.', 'danger'],
        "a failed save is not an acknowledgement, and the server's own reason is its words");
    assert.equal(panel.raw(), 'Settings file is read-only.', 'the raw text stays one disclosure away');
});

test('an edit of the budget fields clears the result that time never clears; a failed reach is owner words', async (t) => {
    const panel = costsPanel(t, [reply({ restart_required: true }), new TypeError('Load failed')]);
    await panel.press();
    panel.tick(10 * ACK);
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn']);
    const card = panel.save.closest('.costs-budget');
    const field = panel.mount.querySelector('#s-budget');
    await card.fire('input', { target: panel.status });
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn'], 'its own disclosure toggling is no edit');
    await card.fire('input', { target: field });
    assert.deepEqual(panel.words(), ['', ''], 'a new edit starts a new subject');
    await panel.press();
    assert.deepEqual(panel.words(), ["Not saved. Couldn't reach Ouroboros.", 'danger']);
    assert.equal(panel.raw(), 'Load failed', "the browser's transport text is never the sentence");
    assert.doesNotMatch(panel.words()[0], /Error:|Load failed/);
    field.value = '0';
    await panel.press();
    assert.deepEqual(panel.words(), ['Budget values must be at least 0.01.', 'danger'], 'nothing is sent for an invalid value');
    assert.equal(panel.posts.length, 2);
});

test('one timer owner: an older acknowledgement never erases what a newer save says', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true }), reply({ error: 'Offline.' }, 503),
        reply({ immediate_changed: true }), reply({ restart_required: true })]);
    await panel.press();
    panel.tick(ACK - 1000);
    await panel.press();  // a failing save inside the window: the first timer must not take it away
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Not saved. Offline.', 'danger']);
    assert.equal(panel.raw(), 'Offline.');
    // Two plain saves: the second one's own full time counts, and the first one's timer is gone.
    await panel.press();
    panel.tick(ACK - 1000);
    await panel.press();  // answered by the restart-required reply
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Saved. Restart required.', 'warn']);
});

test('a validation message after an acknowledgement is not erased by the acknowledgement\'s timer either', async (t) => {
    const panel = costsPanel(t, [reply({ immediate_changed: true })]);
    await panel.press();
    panel.mount.querySelector('#s-budget').value = '0';
    panel.tick(ACK - 1);
    await panel.press();
    panel.tick(ACK);
    assert.deepEqual(panel.words(), ['Budget values must be at least 0.01.', 'danger'], 'never in the acknowledgement\'s tone');
    assert.equal(panel.posts.length, 1, 'an invalid value is never sent');
});
