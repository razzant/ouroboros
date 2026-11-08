// The provider is CHOSEN from the grouped source select on every surface, and
// the editor composes the stored `provider::model` spelling. The owner never
// types a `::` prefix, and a provider without a stored key is not offered.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import {
    API_PROVIDER_CREDENTIAL_KEYS, API_PROVIDER_ORDER, API_ROUTE_CHOICE, apiProviderLabel,
    changeRouteChoice, compatibleEndpoint, configuredApiProviders, decodeRouteChoice, encodeRouteChoice,
    routeChoiceGroups, routeModelFields, routeModelInputHtml, routeModelSuggestions,
    routeTargetFromModel, sourceIdentityLabel,
} from '../modules/route_editor_primitives.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));
const api = (target, kind = 'api_model') => ({ kind, target_id: target });
const catalog = [
    { value: 'openai/gpt-5.6-terra', label: 'OpenRouter · Terra' },
    { value: 'openai::gpt-5.6-terra', label: 'OpenAI · Terra' },
    { value: 'anthropic::claude-opus-5', label: 'Anthropic · Opus' },
    { value: 'claudexor::codex=gpt-test', label: 'Codex · gpt-test' },
];

test('a provider is offered only while its credential is stored, in one owner-facing order', () => {
    assert.deepEqual(configuredApiProviders({}), []);
    const every = configuredApiProviders({
        // Settings answers a GET with a placeholder, never the stored bytes.
        OPENAI_COMPATIBLE_BASE_URL: 'http://localhost:11434/v1',
        GIGACHAT_USER: 'owner', GIGACHAT_PASSWORD: '***set***',
        CLOUDRU_FOUNDATION_MODELS_API_KEY: '***set***', MINIMAX_API_KEY: 'mm',
        DEEPSEEK_API_KEY: 'ds', ZAI_API_KEY: 'zai',
        ANTHROPIC_API_KEY: 'sk-ant', OPENAI_API_KEY: 'sk',
        OPENROUTER_API_KEY: 'sk-or',
    });
    assert.deepEqual(every.map((provider) => provider.id), API_PROVIDER_ORDER);
    // A blank or whitespace value is not a credential.
    assert.deepEqual(configuredApiProviders({ OPENAI_API_KEY: '   ', ANTHROPIC_API_KEY: '' }), []);
});

test('a provider with more than one credential shape accepts either of them', () => {
    const ids = (settings) => configuredApiProviders(settings).map((provider) => provider.id);
    assert.deepEqual(ids({ GIGACHAT_CREDENTIALS: '***set***' }), ['gigachat']);
    assert.deepEqual(ids({ GIGACHAT_USER: 'owner' }), [], 'half a basic-auth pair is not a login');
    assert.deepEqual(ids({ GIGACHAT_PASSWORD: '***set***' }), []);
    assert.deepEqual(ids({ GIGACHAT_USER: 'owner', GIGACHAT_PASSWORD: '***set***' }), ['gigachat']);
    assert.deepEqual(ids({ OPENAI_COMPATIBLE_BASE_URL: 'http://localhost:1234/v1' }), ['openai-compatible']);
    // The legacy address reaches an endpoint only with the OpenAI key it travels with, exactly as
    // provider_models.compatible_endpoint_configured reads it; that key still offers official OpenAI.
    assert.deepEqual(ids({ OPENAI_BASE_URL: 'http://localhost:1234/v1' }), []);
    assert.deepEqual(ids({ OPENAI_BASE_URL: 'http://localhost:1234/v1', OPENAI_API_KEY: '***set***' }),
        ['openai', 'openai-compatible']);
    // …and the key alone does not make an endpoint usable.
    assert.deepEqual(ids({ OPENAI_API_KEY: 'sk' }), ['openai']);
    for (const key of API_PROVIDER_CREDENTIAL_KEYS) assert.ok(configuredApiProviders({ [key]: 'x' }).length <= 1, key);
});

test('the built-in compatible source sends one pair, never borrowing the other pair\'s key', () => {
    const legacy = { OPENAI_BASE_URL: ' http://127.0.0.1:8080/v1 ', OPENAI_API_KEY: '***set***' };
    assert.deepEqual(compatibleEndpoint(legacy), { baseUrl: 'http://127.0.0.1:8080/v1', apiKey: '***set***', apiKeySetting: 'OPENAI_API_KEY', legacy: true });
    // A dedicated address wins and travels with its own key, blank meaning no auth.
    assert.deepEqual(compatibleEndpoint({ ...legacy, OPENAI_COMPATIBLE_BASE_URL: 'http://127.0.0.1:9090/v1' }),
        { baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '', apiKeySetting: 'OPENAI_COMPATIBLE_API_KEY', legacy: false });
    assert.deepEqual(compatibleEndpoint({ ...legacy, OPENAI_COMPATIBLE_BASE_URL: 'http://127.0.0.1:9090/v1',
        OPENAI_COMPATIBLE_API_KEY: 'dedicated' }), { baseUrl: 'http://127.0.0.1:9090/v1', apiKey: 'dedicated', apiKeySetting: 'OPENAI_COMPATIBLE_API_KEY', legacy: false });
    // A dedicated key without its address configures nothing and never reaches the legacy address.
    assert.deepEqual(compatibleEndpoint({ OPENAI_BASE_URL: 'http://127.0.0.1:8080/v1', OPENAI_COMPATIBLE_API_KEY: 'dedicated' }),
        { baseUrl: 'http://127.0.0.1:8080/v1', apiKey: '', apiKeySetting: 'OPENAI_API_KEY', legacy: true });
    assert.deepEqual(compatibleEndpoint({}), { baseUrl: '', apiKey: '', apiKeySetting: 'OPENAI_API_KEY', legacy: true });
});

test('provider names come from the setup contract, with a name for the ones it omits', () => {
    assert.equal(apiProviderLabel('openai', contract.providerProfiles), 'OpenAI');
    assert.equal(apiProviderLabel('cloudru', contract.providerProfiles), 'Cloud.ru Foundation Models');
    assert.equal(apiProviderLabel('gigachat', contract.providerProfiles), 'GigaChat');
    const profiles = structuredClone(contract.providerProfiles);
    profiles.gigachat.label = 'Configured GigaChat';
    assert.equal(apiProviderLabel('gigachat', profiles), 'Configured GigaChat');
    // Exercise an omitted profile explicitly, without pinning the setup contract's membership.
    delete profiles.gigachat;
    assert.equal(apiProviderLabel('gigachat', profiles), 'GigaChat');
    assert.equal(apiProviderLabel('openai-compatible'), 'OpenAI-compatible endpoint');
    assert.equal(apiProviderLabel('later-provider'), 'later-provider');
});

test('the API choice carries its provider and round-trips to the stored spelling', () => {
    for (const [target, choice] of [
        ['', 'api:openrouter'], ['openai/gpt-5.6-terra', 'api:openrouter'],
        ['openai::gpt-5.6-terra', 'api:openai'], ['anthropic::', 'api:anthropic'],
    ]) {
        assert.equal(encodeRouteChoice({ route: api(target) }), choice, target);
        assert.equal(decodeRouteChoice(choice).provider, choice.slice(4));
        assert.ok(!choice.includes('::'));
    }
    // The legacy bare choice is the OpenRouter lane it always meant.
    assert.deepEqual(decodeRouteChoice(API_ROUTE_CHOICE), { kind: 'api_model', provider: 'openrouter' });
    assert.deepEqual(decodeRouteChoice('subscription:codex'), { kind: 'api_model', source: 'codex' });
    assert.deepEqual(decodeRouteChoice('session:cursor'),
        { kind: 'agent_session', harness: 'cursor' });
});

test('choosing a provider keeps the delivery kind and leaves an empty, prefixed draft', () => {
    for (const kind of ['api_model', 'api_chat']) {
        assert.deepEqual(changeRouteChoice(api('openai::gpt-5.6-terra', kind), 'api:anthropic', { apiKind: kind }),
            { kind, target_id: 'anthropic::' });
        assert.deepEqual(changeRouteChoice(api('openai::gpt-5.6-terra', kind), 'api:openrouter', { apiKind: kind }),
            { kind, target_id: '' }, 'an unprefixed id IS the OpenRouter spelling');
        assert.deepEqual(changeRouteChoice(api('claudexor::codex=gpt-test', kind), 'api:openai', { apiKind: kind }),
            { kind, target_id: 'openai::' });
        // Re-picking the provider a row already uses preserves the whole row.
        const pinned = { ...api('openai::gpt-5.6-terra', kind), credential_profile_id: 'personal' };
        assert.deepEqual(changeRouteChoice(pinned, 'api:openai', { apiKind: kind }), pinned);
    }
});

test('the model field holds the model alone; the editor composes the provider prefix', () => {
    const fields = routeModelFields(api('openai::gpt-5.6-terra'), [], { providerProfiles: contract.providerProfiles });
    assert.equal(fields.model, 'gpt-5.6-terra');
    assert.equal(fields.provider, 'openai');
    assert.equal(fields.providerLabel, 'OpenAI');
    assert.equal(fields.subscription, false);
    const router = routeModelFields(api('openai/gpt-5.6-terra'));
    assert.equal(router.model, 'openai/gpt-5.6-terra', 'an OpenRouter id has no prefix to strip');
    assert.equal(router.provider, 'openrouter');
    assert.equal(routeModelFields(api('')).provider, 'openrouter');
    assert.equal(routeModelFields(api('anthropic::')).model, '');
    // A subscription route names a source, never a provider.
    const subscription = routeModelFields(api('claudexor::codex=gpt-test'), [{ id: 'codex', label: 'Codex' }]);
    assert.equal(subscription.provider, '');
    assert.equal(subscription.model, 'gpt-test');
    assert.equal(subscription.sourceLabel, 'Codex');
    // An agent session is a harness and a model, with no provider at all.
    assert.deepEqual(routeModelFields({ kind: 'agent_session', target_id: 'cursor=gpt-5.6-sol' }),
        { harness: 'cursor', model: 'gpt-5.6-sol', subscription: false, provider: '', providerLabel: '' });
});

test('composing the target restores every spelling the backend routes on', () => {
    assert.equal(routeTargetFromModel(api('openai::gpt-5.6-terra'), 'gpt-5.6-luna'), 'openai::gpt-5.6-luna');
    assert.equal(routeTargetFromModel(api(''), 'openai/gpt-5.6-terra'), 'openai/gpt-5.6-terra');
    assert.equal(routeTargetFromModel(api('anthropic::claude-opus-5'), ''), 'anthropic::',
        'an emptied model keeps the chosen provider as a transient draft');
    assert.equal(routeTargetFromModel(api('openai/gpt-5.6-terra'), ''), '');
    assert.equal(routeTargetFromModel(api('claudexor::codex=gpt-test'), 'gpt-next'), 'claudexor::codex=gpt-next');
    // A model id the owner types WITH a prefix still re-homes the row itself.
    assert.equal(routeTargetFromModel(api(''), 'openai::owner-model'), 'openai::owner-model');
    assert.equal(routeTargetFromModel({ kind: 'agent_session', target_id: 'cursor=old' }, 'new'), 'cursor=new');
});

test('catalog suggestions are scoped to the chosen provider and shown without its prefix', () => {
    assert.deepEqual(routeModelSuggestions(api('openai::gpt-5.6-terra'), catalog), ['gpt-5.6-terra']);
    assert.deepEqual(routeModelSuggestions(api('anthropic::'), catalog), ['claude-opus-5']);
    assert.deepEqual(routeModelSuggestions(api(''), catalog), ['openai/gpt-5.6-terra']);
    assert.deepEqual(routeModelSuggestions(api('claudexor::codex='), catalog), ['gpt-test']);
    const html = routeModelInputHtml('data-model', api('openai::gpt-5.6-terra'), catalog, 'models');
    assert.match(html, /value="gpt-5\.6-terra"/);
    assert.doesNotMatch(html, /openai::|claude-opus-5/, 'no prefix and no other provider in the list');
    // Suggestions assist; they never become an allowlist.
    assert.match(routeModelInputHtml('data-model', api('openai::owner-only'), catalog, 'models'),
        /value="owner-only"/);
});

test('every surface draws the same groups, in the same order, with the same words', () => {
    const args = { harnesses: [{ id: 'codex', display_name: 'Codex CLI' }],
        modelSources: [{ id: 'codex-models', label: 'Codex' }],
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k', OPENAI_API_KEY: 'k' }) };
    const groups = routeChoiceGroups(args);
    assert.deepEqual(groups.map((group) => group.label),
        ['Subscriptions · models', 'API keys', 'Agents · sessions']);
    assert.deepEqual(groups[1].options.map((option) => option.value),
        ['api:openrouter', 'api:openai', '']);
    assert.deepEqual(groups[1].options.map((option) => option.label),
        ['OpenRouter', 'OpenAI', 'Add a key in Accounts for more']);
    assert.equal(groups[1].options.at(-1).disabled, true);
    for (const group of groups) {
        for (const option of group.options) assert.ok(!option.value.includes('::'), option.value);
    }
    // A surface that cannot deliver a group omits it rather than renaming it.
    assert.deepEqual(routeChoiceGroups({ ...args, includeSessions: false })
        .map((group) => group.label), ['Subscriptions · models', 'API keys']);
    assert.deepEqual(routeChoiceGroups({ ...args, includeSubscriptions: false })
        .map((group) => group.label), ['API keys', 'Agents · sessions']);
});

test('a saved provider whose key is gone stays selectable and is labelled, never silently swapped', () => {
    const groups = routeChoiceGroups({ currentChoice: 'api:anthropic',
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }),
        providerProfiles: contract.providerProfiles });
    const options = groups.find((group) => group.label === 'API keys').options;
    assert.deepEqual(options.map((option) => option.value), ['api:openrouter', 'api:anthropic', '']);
    assert.equal(options[1].label, 'Anthropic (no key)');
    assert.ok(!options[1].disabled);
    // A configured provider is never duplicated by the saved-choice rescue.
    assert.deepEqual(routeChoiceGroups({ currentChoice: 'api:openrouter',
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }) })
        .find((group) => group.label === 'API keys').options.map((option) => option.value),
    ['api:openrouter', '']);
    // With no key at all the group still explains where keys are added.
    assert.deepEqual(routeChoiceGroups({}).find((group) => group.label === 'API keys').options,
        [{ value: '', disabled: true, label: 'Add a key in Accounts for more' }]);
});

test('the source chip names who serves the route, in the owner-facing words', () => {
    assert.equal(sourceIdentityLabel(api('openai::gpt-5.6-terra'),
        { providerProfiles: contract.providerProfiles }), 'API · OpenAI');
    assert.equal(sourceIdentityLabel(api('openai/gpt-5.6-terra')), 'API · OpenRouter');
    assert.equal(sourceIdentityLabel(api('claudexor::codex-models=gpt-test'),
        { modelSources: [{ id: 'codex-models', label: 'Codex' }] }), 'Codex · model');
    // An undiscovered source is still named by its own id, never by the aggregator.
    assert.equal(sourceIdentityLabel(api('claudexor::later=gpt-test')), 'later · model');
    assert.equal(sourceIdentityLabel({ kind: 'agent_session', target_id: 'codex=gpt-5.6-sol' },
        { harnesses: [{ id: 'codex', display_name: 'Codex CLI' }] }), 'Codex CLI · agent');
    assert.equal(sourceIdentityLabel({ kind: 'agent_session', target_id: 'codex' }), 'codex · agent');
});
