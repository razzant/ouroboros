import assert from 'node:assert/strict';
import test from 'node:test';
import { syncPolicyState } from '../modules/settings_policy.js';
import { collectCustomSecretDraft, deepseekSearchKey, paintSettingsFieldErrors, searchModelAfterSourceSwitch, searchModelForKey,
    settingsWriteFailure, syncSearchKey, syncSearchModelWords, webSearchPreviewWords } from '../modules/settings_controls.js';

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

test('a DeepSeek preview names the key it uses; every other source keeps its own words', () => {
    const name = (id) => ({ deepseek: 'DeepSeek', openai: 'OpenAI', anthropic: 'Anthropic' })[id] || id;
    const leg = { source: 'deepseek', backend: 'deepseek_search_tool', model: '', model_source: 'provider_default' };
    assert.deepEqual(webSearchPreviewWords({ source: 'deepseek', legs: [{ ...leg, connection: 'work' }] }, name),
        { main: 'Uses DeepSeek.', details: 'DeepSeek · connection work. Uses saved account credentials.' });
    assert.deepEqual(webSearchPreviewWords({ source: 'deepseek', legs: [leg] }, name),
        { main: 'Uses DeepSeek.', details: 'DeepSeek · built-in key. Uses saved account credentials.' });
    // A saved model this Source ignores changes what the owner does next, so it stays in the main line.
    assert.equal(webSearchPreviewWords({ source: 'deepseek', legs: [leg], unapplied_model: 'gpt-5.5' }, name).main,
        'Uses DeepSeek. DeepSeek takes no model, so the saved model gpt-5.5 is not applied.');
    // A named connection that is missing, disabled or of another kind is named as itself, never as the built-in key.
    const gap = (extra) => webSearchPreviewWords({ source: 'deepseek', legs: [],
        unavailable: [{ ...leg, ...extra, unavailable_reason: 'no_credentials' }] }, name);
    assert.deepEqual(gap({ connection: 'work' }), { main: 'No search source is ready.',
        details: 'Not available now: DeepSeek connection work (not configured). Uses saved account credentials.' });
    assert.equal(gap({}).details, 'Not available now: DeepSeek built-in key (not configured). Uses saved account credentials.');
    // Auto and the sources with a model say what they said before DeepSeek: no connection beside a model.
    assert.deepEqual(webSearchPreviewWords({ source: 'auto', unapplied_model: 'x-1',
        legs: [{ source: 'openai', model: 'gpt-5.5', model_source: 'selection', connection: 'work' }],
        unavailable: [{ source: 'anthropic', unavailable_reason: 'no_credentials' }] }, name), {
        main: 'Uses OpenAI. No built-in search serves the saved model x-1; Auto uses the recommended models.',
        details: 'OpenAI · gpt-5.5. Not available now: Anthropic (no API key). Uses saved account credentials.' });
});

test('the search Model help and placeholder change with the Source only, so typing never rewrites a translation', () => {
    const writes = [];
    const help = { set innerHTML(value) { writes.push(value); } };
    const field = { set placeholder(value) { writes.push(`placeholder: ${value}`); } };
    const select = { dataset: { modelHelp: 'Page help', modelPlaceholder: 'Recommended for each source' },
        selectedOptions: [{ dataset: {} }] };
    syncSearchModelWords(select, field, help);  // the page rendered the select's own words: nothing to write
    syncSearchModelWords(select, field, help);  // a keystroke
    assert.deepEqual(writes, []);
    select.selectedOptions = [{ dataset: { modelHelp: 'DeepSeek help', modelPlaceholder: 'Built-in DeepSeek key' } }];
    syncSearchModelWords(select, field, help);
    syncSearchModelWords(select, field, help);  // typing under DeepSeek writes nothing more
    select.selectedOptions = [{ dataset: {} }];
    syncSearchModelWords(select, field, help);  // back to a source with a model: the page's own English again
    assert.deepEqual(writes, ['DeepSeek help', 'placeholder: Built-in DeepSeek key', 'Page help',
        'placeholder: Recommended for each source']);
    syncSearchModelWords(null, field, help);  // a page without the section changes nothing
    assert.equal(writes.length, 4);
});

test('a DeepSeek key is chosen from the keys of Accounts and spelled only by the editor', () => {
    // The saved Model as the Key select reads it: a key, or null for a misspelt head the resolver refuses.
    assert.deepEqual(['', 'deepseek::', 'deepseek@default::', 'deepseek@work::', 'gpt-5.5', 'openai@work::gpt-5.5']
        .map(deepseekSearchKey), ['deepseek', 'deepseek', 'deepseek', 'deepseek@work', 'deepseek', 'deepseek']);
    assert.deepEqual(['deepseek@work', 'deepseek@work:', 'DeepSeek@work::', 'deepseek@work:: (local)', 'deepseek::deepseek-chat']
        .map(deepseekSearchKey), [null, null, null, null, null]);
    assert.deepEqual([searchModelForKey('api:deepseek', 'gpt-5.5'), searchModelForKey('api:deepseek@work', ''),
        searchModelForKey('saved', 'deepseek@work')], ['', 'deepseek@work::', 'deepseek@work']);
    // Leaving DeepSeek drops only the spelling written for it; another source's model stays the owner's.
    assert.deepEqual([searchModelAfterSourceSwitch('auto', 'deepseek@work::'), searchModelAfterSourceSwitch('openai', 'gpt-5.5'),
        searchModelAfterSourceSwitch('deepseek', 'deepseek@work::'), searchModelAfterSourceSwitch('auto', 'deepseek@work')],
    ['', 'gpt-5.5', 'deepseek@work::', 'deepseek@work']);

    const key = (id, label, extra = {}) => ({ id, label, kind: id.split('@')[0], name: id.split('@')[1] || '', enabled: true,
        configured: true, ...extra });
    const connections = [key('deepseek', 'DeepSeek'), key('deepseek@work', 'DeepSeek · Work'),
        key('deepseek@old', 'DeepSeek · old', { enabled: false }), key('openai', 'OpenAI')];
    const keys = { connections, providers: connections.filter((entry) => entry.enabled).map(({ id, label }) => ({ id, label })) };
    const label = () => ({ hidden: false });
    const page = (saved, source = 'deepseek') => ({ source: { value: source }, field: { value: saved, labels: [label()] },
        select: { innerHTML: '', value: '', labels: [label()] } });
    const shown = (select) => [...select.innerHTML.matchAll(/<option value="([^"]*)"( selected)?( disabled)?>([^<]*)</g)]
        .map(([, value, selected, disabled, text]) => `${text}${selected ? ' *' : ''}${disabled ? ' -' : ''}`);
    for (const [saved, expected] of [
        ['', ['DeepSeek *', 'DeepSeek · Work', 'Add a key in Accounts for more -']],
        ['deepseek@work::', ['DeepSeek', 'DeepSeek · Work *', 'Add a key in Accounts for more -']],
        ['deepseek@old::', ['DeepSeek', 'DeepSeek · Work', 'DeepSeek · old (off) *', 'Add a key in Accounts for more -']],
        ['deepseek@gone::', ['DeepSeek', 'DeepSeek · Work', 'DeepSeek · gone (removed) *', 'Add a key in Accounts for more -']],
        ['deepseek@work', ['Saved choice (not readable) *', 'DeepSeek', 'DeepSeek · Work', 'Add a key in Accounts for more -']],
    ]) {
        const { source, field, select } = page(saved);
        syncSearchKey(source, field, select, keys);
        assert.deepEqual(shown(select), expected, saved);
        assert.equal(select.value, saved === 'deepseek@work' ? 'saved' : `api:${deepseekSearchKey(saved)}`);
        assert.deepEqual([field.hidden, field.labels[0].hidden, select.hidden, select.labels[0].hidden], [true, true, false, false]);
        assert.equal(field.value, saved);  // showing a choice never rewrites the saved Model
    }
    const { source, field, select } = page('gpt-5.5', 'openai');  // every other Source keeps its Model field
    syncSearchKey(source, field, select, keys);
    assert.deepEqual([field.hidden, select.hidden, field.value], [false, true, 'gpt-5.5']);
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
