// What the Model capsule knows, as data (chat_model_routes.js; DESIGN "Composer owner controls"): the sources that
// can lead a work, one row per (source, model) however many accounts list it, what a route is called, and a search
// across the sources. The catalog suggests and never admits: a route it does not list is still described and can
// still be typed.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
    buildModelIndex, compareNewerFirst, describeRoute, normalizeChoice, routeKey, routeSource, sameRoute, searchModels,
    shortName, sourceUnlisted, typedRoute, windowWords,
} from '../modules/chat_model_routes.js';

const ASTRA = 'claudexor::codex=gpt-6-astra';
const SOL = 'claudexor::codex=gpt-6.1-sol';
const codex = (model, profile, extra = {}) => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id: model,
    name: model, value: `claudexor::codex=${model}`, credential_profile_id: profile, ...extra });
const api = (provider, id, extra = {}) => ({ provider_id: provider, provider, id, name: id,
    value: provider === 'openrouter' ? id : `${provider}::${id}`, ...extra });
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [
        codex('gpt-6-astra', 'work', { availability: 'unavailable', context_window: 400000, max_context_window: 872000 }),
        codex('gpt-6-astra', 'home', { availability: 'available', context_window: 400000 }),
        codex('gpt-6.1-sol', 'work', { availability: 'unavailable', max_context_window: 1000000 }),
        codex('gpt-6.1-sol', 'home', { availability: 'unavailable' }),
        codex('gpt-5-old', 'work', { availability: 'unknown' }),
        api('openrouter', 'z-ai/glm-5.3'), api('openrouter', 'anthropic/claude-opus-5.5'), api('openrouter', 'x-ai/grok-4.7'),
        api('openrouter@work', 'z-ai/glm-5.3', { provider_label: 'OpenRouter · Work' }),
        api('openrouter@work', 'moonshotai/kimi-k3', { provider_label: 'OpenRouter · Work' }),
        api('openai', 'gpt-6'),
        { provider_id: 'local', provider: 'Local', id: 'qwen3.gguf', name: 'qwen3.gguf', value: 'local::qwen3.gguf', use_local: true, context_window: 32768 },
    ],
    errors: [{ provider_id: 'anthropic', error: 'HTTP 401 For more information check: https://x' },
        { provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'old', error: 'sign in again' },
        { provider_id: 'extra_ca_bundle', error: 'bundle unreadable' }],
};

test('a room follows Main or holds one route: nothing else is a choice', () => {
    assert.deepEqual(normalizeChoice({ kind: 'model', model: ` ${ASTRA} `, use_local: 'yes' }), { kind: 'model', model: ASTRA, use_local: false });
    assert.deepEqual(normalizeChoice({ kind: 'model', model: 'qwen', use_local: true }), { kind: 'model', model: 'qwen', use_local: true });
    for (const unreadable of [null, undefined, {}, { kind: 'fallback' }, { kind: 'model', model: ' ' }, 'main']) {
        assert.deepEqual(normalizeChoice(unreadable), { kind: 'main' });
    }
    assert.ok(sameRoute({ model: 'a', use_local: false }, { model: 'a' }));
    assert.ok(!sameRoute({ model: 'a', use_local: true }, { model: 'a', use_local: false }), 'the local lane is another route');
    assert.notEqual(routeKey({ model: 'a', use_local: true }), routeKey({ model: 'a', use_local: false }));
});

test('the sources are the ones that can lead: subscriptions, API keys that answered or failed, the local model', () => {
    const index = buildModelIndex({ catalog: CATALOG });
    assert.deepEqual(index.sources.map((source) => [source.id, source.label, source.pay, source.count]), [
        ['subscription:codex', 'Codex', 'subscription', 3],
        ['openrouter', 'OpenRouter', 'paid by key', 3],
        ['openrouter@work', 'OpenRouter · Work', 'paid by key', 2],
        ['openai', 'OpenAI', 'paid by key', 1],
        ['anthropic', 'Anthropic', 'paid by key', 0],
        ['local', 'Local', 'on this computer', 1],
    ]);
    assert.deepEqual(index.source('anthropic').problems, ['HTTP 401'], 'a list that failed is still a source, and says so');
    assert.deepEqual(index.source('subscription:codex').problems, ['sign in again']);
    assert.deepEqual(index.problems, ['bundle unreadable'], 'a failure of no one source is the catalog\'s');
    assert.equal(buildModelIndex({ catalog: {} }).sources.length, 0, 'an unread catalog names no source');
});

test('one row per source and model: many accounts are one row, available when any of them can serve it', () => {
    const index = buildModelIndex({ catalog: CATALOG });
    const row = (model, local = false) => index.row({ model, use_local: local });
    assert.equal(index.rows.filter((entry) => entry.source === 'subscription:codex').length, 3, 'three models, five listings');
    assert.deepEqual([row(ASTRA).available, row(SOL).available, row('claudexor::codex=gpt-5-old').available], [true, false, null]);
    assert.equal(row(ASTRA).window, 872000, 'the widest window any account reports');
    assert.equal(row('z-ai/glm-5.3').available, null, 'an API row claims nothing about availability');
    assert.ok(!('credential_profile_id' in row(ASTRA)), 'a row never names an account');
    // A local model's row is its artifact with `use_local`; the catalog's `value` for it is a label, never a route.
    assert.deepEqual(row('qwen3.gguf', true).route, { model: 'qwen3.gguf', use_local: true });
    assert.equal(row('local::qwen3.gguf'), null);
    // Each source's models read newer first.
    assert.deepEqual(index.sourceRows('subscription:codex').map((entry) => entry.id), ['gpt-6.1-sol', 'gpt-6-astra', 'gpt-5-old']);
});

test('a route is called by its short name, its source, how the source is paid and its window when one is known', () => {
    const index = buildModelIndex({ catalog: CATALOG });
    const said = describeRoute(index, { model: ASTRA });
    assert.deepEqual([said.short, said.sourceLabel, said.pay, said.window, said.facts],
        ['gpt-6-astra', 'Codex', 'subscription', '872K', 'Codex · subscription · 872K window']);
    const glm = describeRoute(index, { model: 'z-ai/glm-5.3' });
    assert.deepEqual([glm.short, glm.id, glm.facts], ['glm-5.3', 'z-ai/glm-5.3', 'OpenRouter · paid by key'], 'an unknown window is omitted');
    assert.equal(describeRoute(index, { model: 'qwen3.gguf', use_local: true }).facts, 'Local · on this computer · 33K window');
    // The catalog never admits: a route it does not list is described from the route itself.
    const typed = describeRoute(buildModelIndex({ catalog: {} }), { model: 'claudexor::codex=gpt-9' });
    assert.deepEqual([typed.short, typed.sourceLabel, typed.listed, typed.harness], ['gpt-9', 'Codex', false, 'codex']);
    assert.equal(describeRoute(index, null), null);
    assert.deepEqual([windowWords(1000000), windowWords(1050000), windowWords(272000), windowWords(0), windowWords(undefined)],
        ['1M', '1.1M', '272K', '', '']);
    assert.equal(shortName({ model: 'openai::gpt-6' }), 'gpt-6');
    assert.equal(routeSource({ model: 'openrouter@work::z-ai/glm-5.3' }), 'openrouter@work');
});

test('a saved API source that cannot send keeps Settings\' own words, once the settings document names its keys', () => {
    const route = { model: 'deepseek::deepseek-v4' };
    const blind = buildModelIndex({ catalog: CATALOG });
    assert.ok(sourceUnlisted(blind, route), 'the catalog read did not list it: the settings read says why');
    assert.equal(describeRoute(blind, route).sourceLabel, 'DeepSeek', 'nothing is claimed before that read');
    const connections = [{ id: 'openrouter', enabled: true, configured: true }, { id: 'deepseek', enabled: true, configured: false },
        { id: 'openai@old', enabled: false, configured: true, label: 'OpenAI · Old' }];
    const index = buildModelIndex({ catalog: CATALOG, connections });
    assert.equal(describeRoute(index, route).sourceLabel, 'DeepSeek (no key)');
    assert.equal(describeRoute(index, { model: 'openai@old::gpt-6' }).sourceLabel, 'OpenAI · Old (off)');
    assert.equal(describeRoute(index, { model: 'zai@gone::glm-5.3' }).sourceLabel, 'Z.ai (GLM) · gone (removed)');
    assert.equal(describeRoute(index, { model: 'z-ai/glm-5.3' }).sourceLabel, 'OpenRouter', 'a key that serves says nothing more');
    assert.ok(!sourceUnlisted(index, { model: ASTRA }) && !sourceUnlisted(index, { model: 'qwen3.gguf', use_local: true }));
});

test('search spans the sources, each model once per provider; a source\'s own list keeps every row', () => {
    const index = buildModelIndex({ catalog: CATALOG });
    const keys = (rows) => rows.map((row) => row.route.model);
    assert.deepEqual(keys(searchModels(index, 'glm')), ['z-ai/glm-5.3'], 'the key added under Accounts repeats its provider\'s list');
    assert.deepEqual(keys(searchModels(index, 'glm', 'openrouter@work')), ['openrouter@work::z-ai/glm-5.3']);
    assert.deepEqual(keys(searchModels(index, 'kimi')), ['openrouter@work::moonshotai/kimi-k3'], 'a model only that key lists is still found');
    // Every word must be held, by the model or by its source; the best match of the last word leads.
    assert.deepEqual(keys(searchModels(index, 'codex 6')), [SOL, ASTRA]);
    assert.deepEqual(keys(searchModels(index, 'gpt')), [SOL, ASTRA, 'claudexor::codex=gpt-5-old', 'openai::gpt-6']);
    assert.deepEqual(keys(searchModels(index, 'opus')), ['anthropic/claude-opus-5.5'], 'a word inside a name');
    assert.deepEqual(keys(searchModels(index, 'a')).slice(0, 2), ['anthropic/claude-opus-5.5', ASTRA],
        'a name that starts with it leads, whatever its source; then a word inside a name');
    assert.deepEqual(searchModels(index, 'no-such-model'), []);
    assert.equal(searchModels(index, '').length, index.rows.filter((row) => !row.repeat).length, 'no words: every model once');
    assert.deepEqual(keys(searchModels(index, '', 'local')), ['qwen3.gguf']);
});

test('any id can still be typed: through the chosen source, the only source, or as the route grammar reads it', () => {
    const index = buildModelIndex({ catalog: CATALOG });
    assert.deepEqual(typedRoute(index, ' gpt-9 ', 'subscription:codex'), { model: 'claudexor::codex=gpt-9', use_local: false });
    assert.deepEqual(typedRoute(index, 'gpt-9', 'openai'), { model: 'openai::gpt-9', use_local: false });
    assert.deepEqual(typedRoute(index, 'other.gguf', 'local'), { model: 'other.gguf', use_local: true });
    assert.deepEqual(typedRoute(index, 'vendor/new-model'), { model: 'vendor/new-model', use_local: false }, 'a bare id is OpenRouter\'s');
    assert.deepEqual(typedRoute(index, 'openai::gpt-9'), { model: 'openai::gpt-9', use_local: false }, 'a written source is kept');
    const codexOnly = buildModelIndex({ catalog: { model_sources: CATALOG.model_sources, items: CATALOG.items.slice(0, 5) } });
    assert.deepEqual(typedRoute(codexOnly, 'gpt-9'), { model: 'claudexor::codex=gpt-9', use_local: false }, 'one source: no source step');
    const noOpenRouter = buildModelIndex({ catalog: { model_sources: CATALOG.model_sources,
        items: [...CATALOG.items.slice(0, 5), api('openai', 'gpt-6')] } });
    assert.equal(typedRoute(noOpenRouter, 'vendor/new-model'), null, 'a source that cannot lead here is never guessed');
    assert.equal(typedRoute(index, '  '), null);
});

test('models read newer first by the version their names state; a name that states none follows', () => {
    const order = (names) => [...names].sort(compareNewerFirst);
    assert.deepEqual(order(['gpt-5.6', 'gpt-6', 'gpt-6.1', 'gpt-6.10', 'gpt-6.9']), ['gpt-6.10', 'gpt-6.9', 'gpt-6.1', 'gpt-6', 'gpt-5.6']);
    assert.deepEqual(order(['sonnet', 'opus-5.5', 'opus-5']), ['opus-5.5', 'opus-5', 'sonnet']);
    assert.deepEqual(order(['b-2', 'a-1']), ['a-1', 'b-2'], 'families follow each other by name');
});
