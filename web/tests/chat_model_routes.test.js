// The chat model picker's data (chat_model_routes.js; DESIGN "Composer owner controls"): subscription sources are the
// catalog's own model sources, API sources the connections that answered or failed; one row per whole route however
// many accounts list it; a partial read keeps what answered and says what did not; a saved, starred or reference
// route no catalog lists stays a row of its own source; the references resolve honestly (a missing Fallback admits on
// Main); a search spans every source; a typed model ID is composed in its source's own spelling; routes read newer
// first by the version their names state (owner 30.10), the stars in their own order; a direct OpenAI key is one
// more API source when it is connected, and no source is offered that the catalog does not name.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
    buildRouteModel, compareNewerFirst, customRoute, describeRoute, homeOf, normalizeChoice, rememberedNames,
    resolveChoice, rowChoice, sameChoice, searchRoutes,
} from '../modules/chat_model_routes.js';

const SOL = 'claudexor::codex=gpt-6.1-sol';
const ASTRA = 'claudexor::codex=gpt-6-astra';
const FLASH = 'openai-compatible::giga-osa-ds-v41-flash/deepseek-v4.1-flash';
// The shape /api/model-catalog answers (gateway/models.py): a subscription source listed by two accounts, one of
// which could not be read, and API connections, a named one among them.
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [
        { value: SOL, name: 'GPT-6.1-Sol', provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'codex-default' },
        { value: ASTRA, name: 'GPT-6-Astra', provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'codex-default' },
        { value: SOL, name: 'GPT-6.1-Sol', provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'other' },
        { value: 'anthropic::claude-opus-5-5', name: 'Claude Opus 5.5', provider_id: 'anthropic', source: 'Anthropic' },
        { value: 'anthropic/claude-opus-5.5', name: 'Claude Opus 5.5', provider_id: 'openrouter', source: 'OpenRouter' },
        { value: 'openai-compatible@work::qwen', name: 'Qwen', provider_id: 'openai-compatible@work', provider_label: 'Work GPU' },
        { value: 'anthropic::claude-haiku-5-5', name: 'Claude Haiku 5.5', provider_id: 'anthropic', origin: 'hint' },
    ],
    errors: [
        { provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'work1', error: 'The account has no verified current catalog credential' },
        { provider_id: 'openai', error: 'HTTP 401 Unauthorized For more information check: https://example.invalid' },
    ],
};
const ROUTES = { main: { model: SOL, use_local: false }, fallback: { model: FLASH, use_local: false } };

test('subscription sources come from the catalog, not a fixed list, and a partial read keeps what answered', () => {
    const model = buildRouteModel({ catalog: CATALOG, routes: ROUTES });
    assert.deepEqual(model.laneSources('subscription').map((source) => [source.id, source.label, source.harness]),
        [['subscription:codex', 'Codex', 'codex']]);
    assert.deepEqual(model.sourceRows('subscription:codex').map((row) => row.route), [SOL, ASTRA], 'one row per route');
    assert.deepEqual(model.source('subscription:codex').problems, ['work1: The account has no verified current catalog credential']);
    assert.deepEqual(buildRouteModel({ catalog: { items: [], model_sources: [] } }).laneSources('subscription'), [],
        'no model source and no route, no subscription source: nothing is assumed');
    const undescribed = buildRouteModel({ catalog: { ...CATALOG, model_sources: [] } }).source('subscription:codex');
    assert.deepEqual([undescribed.label, undescribed.listed], ['codex', true], 'routes a catalog listed keep their source');
    const api = model.laneSources('api').map((source) => source.id);
    assert.deepEqual(api, ['openrouter', 'openai', 'anthropic', 'openai-compatible', 'openai-compatible@work']);
    assert.deepEqual(model.source('openai').problems, ['HTTP 401 Unauthorized'], 'a failed connection is listed with why');
    assert.equal(model.source('openai-compatible@work').label, 'Work GPU', 'a named key keeps its own name');
    assert.equal(model.row('anthropic::claude-haiku-5-5').shipped, true, 'a row known only from the shipped list says so');
});

test('one model through two sources is two routes; stars and references keep whole routes, listed or not', () => {
    const model = buildRouteModel({ catalog: CATALOG, routes: ROUTES, choice: { kind: 'model', model: 'claudexor::codex=gpt-7-unlisted' },
        favorites: ['anthropic/claude-opus-5.5', 'anthropic::claude-opus-5-5', 'vendor/unknown-model', 'anthropic/claude-opus-5.5'] });
    assert.deepEqual(model.favorites.map((row) => [row.route, row.source]), [
        ['anthropic/claude-opus-5.5', 'openrouter'], ['anthropic::claude-opus-5-5', 'anthropic'], ['vendor/unknown-model', 'openrouter']]);
    for (const route of ['claudexor::codex=gpt-7-unlisted', FLASH, 'vendor/unknown-model']) {
        assert.equal(model.row(route).listed, false, `${route} is a route of its own source, not dropped`);
    }
    assert.equal(model.row(FLASH).source, 'openai-compatible');
    const local = buildRouteModel({ catalog: CATALOG, routes: { main: { model: 'qwen3', use_local: true }, fallback: null } });
    assert.ok(local.source('local'), 'a local Main shows its local source');
    assert.equal(describeRoute(local, { route: 'qwen3', use_local: true }).sourceLabel, 'Local model');
});

test('the references resolve honestly: Main, the first Fallback, or Main when no Fallback is set', () => {
    const model = buildRouteModel({ catalog: CATALOG, routes: ROUTES });
    const main = resolveChoice(model, { kind: 'main' }, ROUTES);
    assert.deepEqual([main.current.name, main.current.sourceLabel, main.unconfigured], ['GPT-6.1-Sol', 'Codex', false]);
    const fallback = resolveChoice(model, { kind: 'fallback' }, ROUTES);
    assert.deepEqual([fallback.current.route, fallback.current.name], [FLASH, 'giga-osa-ds-v41-flash/deepseek-v4.1-flash']);
    const missing = resolveChoice(model, { kind: 'fallback' }, { main: ROUTES.main, fallback: null });
    assert.deepEqual([missing.current.route, missing.unconfigured], [SOL, true]);
    const chosen = resolveChoice(model, { kind: 'model', model: ASTRA }, ROUTES);
    assert.deepEqual([chosen.current.name, chosen.current.lane], ['GPT-6-Astra', 'subscription']);
    assert.deepEqual(homeOf(model, chosen), { lane: 'subscription', source: 'subscription:codex' });
    assert.deepEqual(rowChoice(model.row(SOL)), { kind: 'model', model: SOL, use_local: false },
        'a row is an explicit route even when it is Main\'s model; the reference is the foot\'s Use');
    assert.equal(sameChoice({ kind: 'model', model: SOL }, { kind: 'main' }), false);
    assert.deepEqual(normalizeChoice({ kind: 'bogus' }), { kind: 'main' });
});

test('a search spans every source by name, id and source; a typed id is spelled in its source\'s own route', () => {
    const model = buildRouteModel({ catalog: CATALOG, routes: ROUTES });
    assert.deepEqual(searchRoutes(model, 'opus').map((row) => row.source), ['openrouter', 'anthropic'],
        'results keep the rail\'s order of sources');
    assert.deepEqual(searchRoutes(model, 'opus openrouter').map((row) => row.route), ['anthropic/claude-opus-5.5']);
    assert.deepEqual(searchRoutes(model, 'zzz'), []);
    assert.equal(customRoute('subscription:codex', ' gpt-6-sol-mini '), 'claudexor::codex=gpt-6-sol-mini');
    assert.equal(customRoute('anthropic', 'claude-x'), 'anthropic::claude-x');
    assert.equal(customRoute('openrouter', 'vendor/model'), 'vendor/model');
    assert.equal(customRoute('openai-compatible@work', 'm'), 'openai-compatible@work::m');
    assert.equal(customRoute('local', 'x'), '', 'a local route is never typed here');
});

test('remembered display names label a route before the next read without listing it', () => {
    const names = rememberedNames({ old: 'Old' }, CATALOG, 3);
    assert.deepEqual(Object.keys(names), ['anthropic/claude-opus-5.5', 'openai-compatible@work::qwen',
        'anthropic::claude-haiku-5-5'], 'the newest listed names, bounded');
    assert.equal(rememberedNames({}, CATALOG)[SOL], 'GPT-6.1-Sol');
    const model = buildRouteModel({ routes: ROUTES, names: { [SOL]: 'GPT-6.1-Sol' } });
    assert.deepEqual([model.row(SOL).name, model.row(SOL).listed], ['GPT-6.1-Sol', false]);
});

test('a source-less subscription discovery failure remains unknown, with working API routes', () => {
    const model = buildRouteModel({ catalog: { model_sources: [], items: [CATALOG.items[3]],
        errors: [{ provider_id: 'claudexor', source_id: '', error: 'engine unavailable' }] } });
    assert.deepEqual(model.problems, [{ lane: 'subscription', text: 'engine unavailable' }]);
    assert.equal(model.laneRows('api')[0].route, 'anthropic::claude-opus-5-5');
});

test('names that state a version read newer first within their family; others follow; nothing is dropped', () => {
    const order = (names) => [...names].sort(compareNewerFirst);
    assert.deepEqual(order(['GPT-5.6 Sol', 'GPT-6', 'GPT-6.1 Terra', 'Auto Review', 'GPT-6.1']),
        ['GPT-6.1', 'GPT-6.1 Terra', 'GPT-6', 'GPT-5.6 Sol', 'Auto Review'], '6.1 > 6 > 5.6; a name without one follows');
    assert.deepEqual(order(['m 6.9', 'm 6.10', 'm 7', 'm 6.1.2', 'm 6.1']), ['m 7', 'm 6.10', 'm 6.9', 'm 6.1.2', 'm 6.1'],
        'dotted versions compare part by part as numbers, never as text');
    assert.deepEqual(order(['Claude Opus 4.7', 'Llama 3.3 70B', 'Claude Sonnet 5.5', 'Claude Opus 5.5']),
        ['Claude Opus 5.5', 'Claude Opus 4.7', 'Claude Sonnet 5.5', 'Llama 3.3 70B'],
        'unrelated families are not compared by version: they follow each other by name');
    assert.deepEqual(order(['gpt-4o-2024-05-13', 'chat-latest', 'gpt-4o', 'gpt-4o-2024-08-06']),
        ['gpt-4o', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'chat-latest'], 'an alias before its dated snapshots; an undated name last');
    assert.equal(compareNewerFirst('GPT-6 Sol', 'GPT-6 Sol'), 0);

    // The catalog's own order (alphabetical from the server) never decides; the room's routes do not change.
    const codex = (id, name) => ({ value: `claudexor::codex=${id}`, name, provider_id: 'claudexor', source_id: 'codex' });
    const catalog = { model_sources: CATALOG.model_sources, items: [
        codex('auto-review', 'Auto Review'), codex('gpt-5.6-luna', 'GPT-5.6 Luna'), codex('gpt-5.6-sol', 'GPT-5.6 Sol'),
        codex('gpt-6-astra', 'GPT-6 Astra'), codex('gpt-6-chat', 'GPT-6 Chat'), codex('gpt-6.1-sol', 'GPT-6.1 Sol'),
        codex('gpt-6.1-terra', 'GPT-6.1 Terra'),
        { value: 'openai::gpt-6', name: 'gpt-6', provider_id: 'openai' },
        { value: 'openai/gpt-6', name: 'OpenAI: GPT-6', provider_id: 'openrouter' },
        { value: 'openai/gpt-5.6', name: 'OpenAI: GPT-5.6', provider_id: 'openrouter' },
    ] };
    const favorites = ['claudexor::codex=gpt-5.6-luna', 'openai::gpt-6', 'claudexor::codex=gpt-6.1-sol'];
    const model = buildRouteModel({ catalog, favorites });
    assert.deepEqual(model.sourceRows('subscription:codex').map((row) => row.name),
        ['GPT-6.1 Sol', 'GPT-6.1 Terra', 'GPT-6 Astra', 'GPT-6 Chat', 'GPT-5.6 Luna', 'GPT-5.6 Sol', 'Auto Review']);
    assert.deepEqual(model.laneRows('api').map((row) => row.route), ['openai/gpt-6', 'openai/gpt-5.6', 'openai::gpt-6'],
        'across sources the rail\'s order holds; one model through two sources stays two rows');
    assert.deepEqual(searchRoutes(model, 'gpt 6').map((row) => row.route).slice(0, 3),
        ['claudexor::codex=gpt-6.1-sol', 'claudexor::codex=gpt-6.1-terra', 'claudexor::codex=gpt-6-astra']);
    assert.deepEqual(model.favorites.map((row) => row.route), favorites, 'the stars keep the order they were given');
    assert.equal(model.rows.length, catalog.items.length);
});

test('a connected direct OpenAI key is one more API source; a provider the catalog does not name is never offered', () => {
    const connected = buildRouteModel({ catalog: { model_sources: [], errors: [], items: [
        { value: 'openai::gpt-6.1', name: 'gpt-6.1', provider_id: 'openai', provider: 'OpenAI', source: 'OpenAI' },
        { value: 'openai::gpt-6', name: 'gpt-6', provider_id: 'openai', provider: 'OpenAI', source: 'OpenAI' },
        { value: 'anthropic/claude-opus-5.5', name: 'Anthropic: Claude Opus 5.5', provider_id: 'openrouter' },
    ] } });
    assert.deepEqual(connected.laneSources('api').map((source) => [source.id, source.label, source.kind, source.listed]),
        [['openrouter', 'OpenRouter', 'openrouter', true], ['openai', 'OpenAI', 'openai', true]]);
    assert.deepEqual(connected.sourceRows('openai').map((row) => [row.route, row.lane]),
        [['openai::gpt-6.1', 'api'], ['openai::gpt-6', 'api']]);
    assert.deepEqual(rowChoice(connected.row('openai::gpt-6.1')), { kind: 'model', model: 'openai::gpt-6.1', use_local: false });
    // Without its key the catalog has no OpenAI items and no OpenAI error: nothing stands for it, nor for any other
    // provider the owner has not connected (owner 30.10: only connected sources).
    const absent = buildRouteModel({ catalog: { model_sources: [], errors: [], items: [
        { value: 'anthropic/claude-opus-5.5', name: 'Anthropic: Claude Opus 5.5', provider_id: 'openrouter' }] } });
    assert.deepEqual(absent.laneSources('api').map((source) => source.id), ['openrouter']);
    assert.equal(absent.source('openai'), null);
});
