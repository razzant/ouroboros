import assert from 'node:assert/strict';
import test from 'node:test';
import { syncPolicyState } from '../modules/settings_policy.js';
import { collectCustomSecretDraft, paintSettingsFieldErrors, settingsWriteFailure, webSearchPreviewWords } from '../modules/settings_controls.js';

const draft = (overrides = {}) => ({
    key: '', value: '', appliedValue: '', originalKey: '', clear: false, removed: false,
    ...overrides,
});

test('custom-key collection returns every invalid row without changing the draft', () => {
    const rows = [draft({ value: 'keep this draft' }), draft({ key: '!bad', value: 'second' }),
        draft({ key: 'VALID_KEY', value: 'accepted' })].map(Object.freeze);
    const before = JSON.stringify(rows);
    const result = collectCustomSecretDraft(rows);
    assert.deepEqual(result.values, { VALID_KEY: 'accepted' });
    assert.deepEqual(result.errors.map(({ index, field }) => [index, field]), [[0, 'key'], [1, 'key']]);
    assert.equal(JSON.stringify(rows), before);
    const corrected = collectCustomSecretDraft([
        draft({ key: 'first_key', value: 'keep this draft' }),
        draft({ key: 'SECOND_KEY', value: 'second' }), rows[2],
    ]);
    assert.deepEqual(corrected.errors, []);
    assert.deepEqual(corrected.values, { FIRST_KEY: 'keep this draft', SECOND_KEY: 'second', VALID_KEY: 'accepted' });
});

test('custom-key masks, explicit clear and remove retain their different meanings', () => {
    const saved = draft({ key: 'SAVED_KEY', originalKey: 'SAVED_KEY', value: 'abcdefgh...', appliedValue: 'abcdefgh...' });
    assert.deepEqual(collectCustomSecretDraft([saved]), { values: {}, errors: [] });
    assert.deepEqual(collectCustomSecretDraft([{ ...saved, clear: true, value: '' }]), { values: { SAVED_KEY: '' }, errors: [] });
    assert.deepEqual(collectCustomSecretDraft([{ ...saved, removed: true, key: 'RENAMED_KEY' }]), { values: { SAVED_KEY: '' }, errors: [] });
    assert.deepEqual(collectCustomSecretDraft([draft({ removed: true, key: '!unfinished' })]), { values: {}, errors: [] });
    // Literal ellipses are valid new secret bytes, not a mask heuristic.
    assert.deepEqual(collectCustomSecretDraft([draft({ key: 'NEW_KEY', value: 'literal...inside' })]).values,
        { NEW_KEY: 'literal...inside' });
});

test('renaming a saved key cannot pretend its masked value was copied', () => {
    const renamed = draft({ key: 'NEW_KEY', originalKey: 'SAVED_KEY', value: 'abcdefgh...', appliedValue: 'abcdefgh...' });
    const invalid = collectCustomSecretDraft([renamed]);
    assert.equal(invalid.errors[0].field, 'value');
    assert.deepEqual(invalid.values, {});
    const entered = collectCustomSecretDraft([{ ...renamed, value: 'new value' }]);
    assert.deepEqual(entered, { values: { NEW_KEY: 'new value' }, errors: [] });
    assert.equal('SAVED_KEY' in entered.values, false, 'a new value does not implicitly delete the old key');
});

test('duplicate and built-in names cannot silently overwrite another field', () => {
    const result = collectCustomSecretDraft([
        draft({ key: 'same_key', value: 'one' }), draft({ key: 'SAME_KEY', value: 'two' }),
        draft({ key: 'OUROBOROS_PRIVATE', value: 'three' }), draft({ key: 'TOTAL_BUDGET', value: 'four' }),
    ], ['TOTAL_BUDGET']);
    assert.deepEqual(new Set(result.errors.map(({ index }) => index)), new Set([0, 1, 2, 3]));
    assert.equal(result.values.TOTAL_BUDGET, undefined);
    assert.equal(result.values.OUROBOROS_PRIVATE, undefined);
});

test('field validation clears the old error and keeps its accessible association', () => {
    const attributes = new Map([['aria-describedby', 'custom-key-help']]);
    const hint = { id: 'custom-key-error', hidden: true, textContent: '' };
    const input = { id: 'custom-key', dataset: {},
        getAttribute: (name) => attributes.get(name),
        setAttribute: (name, value) => attributes.set(name, value),
        removeAttribute: (name) => attributes.delete(name) };
    const root = {
        querySelectorAll: () => input.dataset.settingsValidation ? [input] : [],
        querySelector: () => hint,
    };
    paintSettingsFieldErrors(root, [{ input, message: 'Invalid key' }]);
    assert.equal(attributes.get('aria-invalid'), 'true');
    assert.equal(attributes.get('aria-describedby'), 'custom-key-help custom-key-error');
    assert.equal(hint.hidden, false);
    paintSettingsFieldErrors(root, []);
    assert.equal(attributes.has('aria-invalid'), false);
    assert.equal(hint.hidden, true);
    assert.equal(hint.textContent, '');
    assert.equal(attributes.get('aria-describedby'), 'custom-key-help custom-key-error');
});

test('settings write errors preserve saved, unsaved and unknown receipts', () => {
    const error = new Error('stage refused');
    error.body = { saved: true };
    assert.deepEqual(settingsWriteFailure(error, 'Runtime mode'), {
        unknown: false, text: 'Runtime mode was saved, but a later step failed: stage refused',
    });
    error.body = { saved: false };
    assert.match(settingsWriteFailure(error).text, /was not changed/);
    for (const body of [{ saved: null }, {}, undefined]) {
        error.body = body;
        const result = settingsWriteFailure(error);
        assert.equal(result.unknown, true);
        assert.match(result.text, /outcome is unknown/);
    }
});

test('the web search preview names the sources once in order; models, gaps and credentials are Details', () => {
    const names = { openai: 'OpenAI', openrouter: 'OpenRouter', anthropic: 'Anthropic', ddgs: 'DuckDuckGo' };
    const name = (id) => names[id] || id;
    const words = webSearchPreviewWords({
        legs: [{ source: 'openai', model: 'gpt-5.2' }, { source: 'openrouter', model: 'openai/gpt-5.2', model_source: 'provider_default' },
            { source: 'ddgs' }],
        unavailable: [{ source: 'anthropic', unavailable_reason: 'package_missing' }],
    }, name);
    assert.equal(words.main, 'Uses OpenAI, then OpenRouter, then DuckDuckGo.');
    assert.equal(words.details, 'OpenAI · gpt-5.2; OpenRouter · openai/gpt-5.2 (recommended). '
        + 'Not available now: Anthropic (package not installed). Uses saved account credentials.');
    // No id, arrow or model in the main line; the gap that changes nothing stays out of it.
    assert.doesNotMatch(words.main, /openai|ddgs|→|gpt|Anthropic/);
    // A saved model a source ignores changes what the owner does next, so it stays in the main line.
    assert.match(webSearchPreviewWords({ legs: [{ source: 'ddgs' }], unapplied_model: 'x-1' }, name).main,
        /^Uses DuckDuckGo\. No built-in search serves the saved model x-1/);
    assert.deepEqual(webSearchPreviewWords({ legs: [] }, name), { main: 'No search source is ready.', details: 'Uses saved account credentials.' });
    assert.deepEqual(webSearchPreviewWords({ error: 'Search routes are not ready' }), { main: 'Search routes are not ready', details: '' });
});

test('a policy line says one value once and spells the facts out only where they differ', () => {
    const nodes = {};
    const root = { querySelector: (selector) => (selector.startsWith('[data-policy-state') ? (nodes[selector] ||= { textContent: '' }) : null) };
    const line = (key) => nodes[`[data-policy-state="${key}"]`].textContent;
    const same = { configured: 'advisory', current_process: 'advisory', next_task: 'advisory' };
    syncPolicyState(root, { policy_state: { access: { configured: 'pro', current_process: 'pro' }, supervisor: same,
        review: { ...same, active_task_snapshot: true } } });
    assert.equal(line('access'), 'Pro now and after restart');
    assert.equal(line('supervisor'), 'Advisory now and for new tasks');
    assert.equal(line('review'), 'Advisory now and for new tasks; the running task keeps the value it started with');
    assert.doesNotMatch(line('review'), /Advisory.*Advisory/, 'never the same value three times');
    syncPolicyState(root, { policy_state: { access: { configured: 'cyber_pro', current_process: 'light', restart_required: true },
        supervisor: { ...same, next_task: 'blocking' }, review: { ...same, configured: 'blocking', next_task: 'blocking', active_task_snapshot: true } } });
    assert.equal(line('access'), 'Saved: Cyber Pro · Current process: Light · After restart: Cyber Pro · Restart required');
    assert.equal(line('supervisor'), 'Current process: Advisory · Next task: Blocking');
    assert.match(line('review'), /^Saved: Blocking · Current process: Advisory · Next task: Blocking · Current task keeps its start snapshot · /);
    assert.match(line('review'), /Blocking would not be enforced under the selected Cyber Pro$/, 'the Cyber Pro note still follows');
});
