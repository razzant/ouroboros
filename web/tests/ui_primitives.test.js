// The home test of web/modules/ui_primitives.js, the portable leaf served to author kits with ui.css: the five
// field/status exports keep their signatures, the result-of-an-action helpers hold DESIGN "Status and chips"
// lifetime in both directions, and the shared button role and result/action composition have one owner.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { readFile } from 'node:fs/promises';
import * as portable from '../modules/ui_primitives.js';
import * as existing from '../modules/ui_helpers.js';
import { escapeHtmlAttr } from '../modules/utils.js';
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

test('a read-only multiline value is whole, selectable text that collection still reaches; editable ones stay controls', () => {
    const long = "Restoring the previous owner's menu button: Telegram did not confirm the restored menu button.";
    const field = { name: 'miniapp_status', label: 'Mini App', type: 'textarea', disabled: true };
    const html = portable.renderSafeField(field, { miniapp_status: long });
    assert.equal(html, '<label class="ui-field widget-field"><span>Mini App</span><output class="ui-control ui-control-readonly" '
        + 'name="miniapp_status" aria-label="Mini App" tabindex="0">Restoring the previous owner&#39;s menu button: Telegram '
        + 'did not confirm the restored menu button.</output></label>');
    assert.doesNotMatch(html, /<textarea|disabled|rows=/, 'no inert control, no fixed rows, no disabled ink');
    const hostile = portable.renderSafeField({ name: 's', type: 'textarea', disabled: true }, { s: '</output><script>x</script>' });
    assert.doesNotMatch(hostile, /<script>/);
    assert.match(hostile, /&lt;\/output&gt;&lt;script&gt;/);
    // An editable textarea, and one disabled only for a while (a busy widget, an unread form until Edit anyway), stay controls.
    assert.match(portable.renderSafeField({ name: 'notes', type: 'textarea' }), /<textarea class="ui-control" name="notes"/);
    assert.match(portable.renderSafeField({ name: 'notes', type: 'textarea' }, {}, { disabled: true }),
        /<textarea class="ui-control" name="notes"[^>]* disabled><\/textarea>/);
    // A disabled single-line field keeps its input: only the multiline read-only value changed.
    assert.match(portable.renderSafeField({ name: 'proxy', type: 'text', disabled: true }), /<input class="ui-control" type="text" name="proxy"[^>]* disabled>/);
    // The output keeps the field's name and its text as `.value`, so the form collects what it shows.
    const form = { elements: { namedItem: (name) => (name === 'miniapp_status' ? { type: 'output', value: long } : null) } };
    assert.deepEqual(portable.collectSafeFieldValues(form, [field]), { miniapp_status: long });
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
    // Settings composes the shared row (the catalog refresh, the LAN hint) and has no page-wide Save bar; the private
    // row stays only for the screens that still read it (Skills › Browse, Updates) and never reaches the shared sheet.
    assert.doesNotMatch(settingsUiJs, /settings-action-row/);
    assert.doesNotMatch(sharedCss, /\.settings-action-row\s*\{/);
    assert.doesNotMatch(settingsUiJs + settingsJs + settingsCss, /settings-footer|btn-save-settings/);
    assert.doesNotMatch(pageCss, /\.skills-card-action-row\s*\{[^}]*(?:display:|grid-template|flex:)/);
    // The shared parts: the action row keeps a reserved result line, the result shares the status sentence's tones.
    assert.match(sharedCss, /\.ui-action-row\s*\{[^}]*flex-wrap:\s*wrap/);
    assert.match(sharedCss, /\.ui-result\s*\{[^}]*min-height:\s*calc\(var\(--type-meta\) \* var\(--line-meta\)\)/);
    assert.match(sharedCss, /\.ui-result\[data-tone="warn"\]/);
    assert.match(sharedCss, /details\.ui-disclosure > summary::before/);
    assert.match(sharedCss, /> \[data-primary\] ~ \[data-primary\] \{/);
});

test('async actions expose the same busy and status semantics', () => {
    // Every API key's Test shares the one busy helper and one status vocabulary.
    assert.match(controlsJs, /export function setButtonBusy\(button, busy\)/);
    assert.match(controlsJs, /setAttribute\('aria-busy', 'true'\)/);
    assert.match(connectionsJs, /setButtonBusy\(button, true\)/);
    assert.match(connectionsJs, /setInlineStatus\(status, 'Testing…', 'muted'\)/);
    assert.match(connectionsJs, /providerTestStatusText\(data\), data\?\.ok \? 'ok' : 'danger'/);
    assert.match(connectionsJs, /setInlineStatus\(row\.querySelector\('\[data-connection-test-status\]'\), '', 'muted'\)/);
    assert.match(primitivesJs, /if \(el\.textContent !== next\) el\.textContent = next/);
    // The catalog's result is the shared result line: its count ends by itself, a read gap stays.
    assert.match(catalogJs, /import \{ setResult \} from '\.\/ui_primitives\.js'/);
    assert.match(catalogJs, /setResult\(statusEl, \{ text, tone, ends: tone === 'ok' \}\)/);
    assert.doesNotMatch(catalogJs, /setInlineStatus/);
    assert.match(catalogJs, /refreshModelCatalog\(\{ button \} = \{\}\)/);
    assert.match(catalogJs, /buttonRefreshes\.get\(button\) === refreshSeq/);
});
