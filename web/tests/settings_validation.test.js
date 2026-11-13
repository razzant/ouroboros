import assert from 'node:assert/strict';
import test from 'node:test';
import { collectCustomSecretDraft, deepseekSearchKey, searchModelAfterSourceSwitch, searchModelForKey, settingsWriteFailure,
    syncSearchKey, webSearchPreviewWords } from '../modules/settings_controls.js';

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

test('a DeepSeek key is chosen from the keys of Accounts and spelled only by the editor', () => {
    // The saved Model as the Key select reads it: a key, or null for a misspelt head the resolver refuses.
    assert.deepEqual(['', 'deepseek::', 'deepseek@default::', 'deepseek@work::', 'gpt-5.5', 'openai@work::gpt-5.5',
        ' deepseek@k-2_b:: ', `deepseek@${'w'.repeat(32)}::`]
        .map(deepseekSearchKey), ['deepseek', 'deepseek', 'deepseek', 'deepseek@work', 'deepseek', 'deepseek',
        'deepseek@k-2_b', `deepseek@${'w'.repeat(32)}`]);
    assert.deepEqual(['deepseek@work', 'deepseek@work:', 'DeepSeek@work::', 'deepseek@work:: (local)', 'deepseek::deepseek-chat']
        .map(deepseekSearchKey), [null, null, null, null, null]);
    // A name outside the key grammar (provider_models._CONNECTION_NAME) as written, which the resolver refuses as
    // invalid: `@default` reads as the built-in key only as the whole name, never after another one.
    assert.deepEqual(['deepseek@::', 'deepseek@Work::', 'deepseek@DEFAULT::', 'deepseek@-work::', 'deepseek@wo rk::',
        'deepseek@work@x::', `deepseek@${'w'.repeat(33)}::`, 'deepseek@work@default::', 'deepseek@default@work::']
        .map(deepseekSearchKey), [null, null, null, null, null, null, null, null, null]);
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
        // A name outside the key grammar is not readable: no second "DeepSeek · Work", no nameless key said to be removed.
        ['deepseek@Work::', ['Saved choice (not readable) *', 'DeepSeek', 'DeepSeek · Work', 'Add a key in Accounts for more -']],
        ['deepseek@::', ['Saved choice (not readable) *', 'DeepSeek', 'DeepSeek · Work', 'Add a key in Accounts for more -']],
        ['deepseek@work@default::', ['Saved choice (not readable) *', 'DeepSeek', 'DeepSeek · Work', 'Add a key in Accounts for more -']],
    ]) {
        const { source, field, select } = page(saved);
        syncSearchKey(source, field, select, keys);
        assert.deepEqual(shown(select), expected, saved);
        assert.equal(select.value, deepseekSearchKey(saved) === null ? 'saved' : `api:${deepseekSearchKey(saved)}`);
        assert.deepEqual([field.hidden, field.labels[0].hidden, select.hidden, select.labels[0].hidden], [true, true, false, false]);
        assert.equal(field.value, saved);  // showing a choice never rewrites the saved Model
    }
    const { source, field, select } = page('gpt-5.5', 'openai');  // every other Source keeps its Model field
    syncSearchKey(source, field, select, keys);
    assert.deepEqual([field.hidden, select.hidden, field.value], [false, true, 'gpt-5.5']);
});
