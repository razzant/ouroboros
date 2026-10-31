// Several API keys per provider (OUROBOROS_CONNECTIONS): every key the owner adds under Accounts is its own
// source in each picker, spelled `kind@name::model` by the editor, and a saved choice whose key is turned
// off or removed stays visible and is never swapped for another key. The Accounts rows render from the same
// document. Fixtures follow the settings API contract (`_meta.connections`, masked row secrets).
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import {
    apiConnections, apiProviderLabel, changeRouteChoice, composeModelSource, configuredApiProviders,
    decodeRouteChoice, encodeRouteChoice, parseModelSource, routeChoiceGroups, routeModelFields,
    routeModelInputHtml, routeModelSuggestions, routeSupportsAccount, routeTargetFromModel, sourceIdentityLabel,
} from '../modules/route_editor_primitives.js';
import { modelSourceGroups, sourceChoice, sourceFromChoice } from '../modules/model_roles.js';
import { adoptConnectionLabels, connectionLabel, routeConnectionId } from '../modules/connection_labels.js';
import { PROVIDER_CARDS, PROVIDER_SECRET_KEYS, connectionRowHtml } from '../modules/settings_provider_cards.js';
import { renderSettingsPage } from '../modules/settings_ui.js';
import {
    cacheHintsNote, connectionNameFor, connectionReferencesText, connectionRowFacts, connectionRows, connectionUsage,
} from '../modules/settings_connections.js';
import { renderLiveCardMeta } from '../modules/chat_activity.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));
const entry = (kind, name, { label = '', enabled = true, configured = true, source = 'settings' } = {}) => ({
    kind, name, label, builtin: !name, enabled, configured, source,
    id: name ? `${kind}@${name}` : kind,
    display: name ? `${apiProviderLabel(kind, contract.providerProfiles)} · ${label || name}` : apiProviderLabel(kind, contract.providerProfiles),
});
// The owner's install of the mockups: a second OpenRouter account, a keyless local server, a turned-off one.
const CONNECTIONS = [
    entry('openrouter', ''), entry('openrouter', 'claude-credits', { label: 'Claude credits' }),
    entry('openai', '', { configured: false, source: 'none' }),
    entry('anthropic', '', { source: 'environment' }),
    entry('openai-compatible', '', { configured: false, source: 'none' }),
    entry('openai-compatible', 'local-vllm', { label: 'Local vLLM' }),
    entry('openai-compatible', 'gigaagent', { label: 'GigaAgent', enabled: false }),
];
const SETTINGS = {
    OPENROUTER_API_KEY: 'sk-or-v1...',
    OUROBOROS_MODEL: 'openrouter@claude-credits::anthropic/claude-opus-5.5',
    OUROBOROS_MODEL_LIGHT: 'claudexor::codex=gpt-6.1-sol',
    OUROBOROS_MODEL_FALLBACKS: 'anthropic/claude-opus-5.5, openai-compatible@gigaagent::giga-max',
    OUROBOROS_CONNECTIONS: [
        { kind: 'openrouter', name: 'claude-credits', label: 'Claude credits', api_key: 'sk-or-v1...' },
        { kind: 'openai-compatible', name: 'local-vllm', label: 'Local vLLM', base_url: 'http://127.0.0.1:8000/v1' },
        { kind: 'openai-compatible', name: 'gigaagent', label: 'GigaAgent', enabled: false,
            base_url: 'https://gigaagent.example.com/v1', api_key: 'ga-live-...' },
    ],
    _meta: { connections: CONNECTIONS },
};
const api = (target, kind = 'api_model') => ({ kind, target_id: target });
const apiGroup = (groups) => groups.find((group) => group.label === 'API keys').options;

test('the API keys group lists one source per enabled connection that holds its key, in the server order', () => {
    const providers = configuredApiProviders(SETTINGS, contract.providerProfiles);
    assert.deepEqual(providers, [
        { id: 'openrouter', label: 'OpenRouter' },
        { id: 'openrouter@claude-credits', label: 'OpenRouter · Claude credits' },
        { id: 'anthropic', label: 'Anthropic' },
        { id: 'openai-compatible@local-vllm', label: 'OpenAI-compatible endpoint · Local vLLM' },
    ]);
    const options = apiGroup(routeChoiceGroups({ providers, connections: apiConnections(SETTINGS) }));
    assert.deepEqual(options.map((option) => option.value), [
        'api:openrouter', 'api:openrouter@claude-credits', 'api:anthropic', 'api:openai-compatible@local-vllm', '',
    ]);
    assert.equal(options.at(-1).label, 'Add a key in Accounts for more');
    for (const option of options) assert.ok(!option.value.includes('::'), option.value);
    // A document without the list (the first-run wizard's flat draft) still names its built-in keys.
    assert.deepEqual(configuredApiProviders({ OPENROUTER_API_KEY: 'k' }).map((provider) => provider.id), ['openrouter']);
    assert.deepEqual(apiConnections({}).filter((connection) => connection.configured), []);
});

test('a saved key that cannot send stays selectable, says why, and is never replaced by another key', () => {
    const connections = apiConnections(SETTINGS);
    const providers = configuredApiProviders(SETTINGS);
    const saved = (choice) => apiGroup(routeChoiceGroups({ providers, connections, currentChoice: choice }))
        .filter((option) => option.value === choice);
    assert.deepEqual(saved('api:openai-compatible@gigaagent'),
        [{ value: 'api:openai-compatible@gigaagent', label: 'OpenAI-compatible endpoint · GigaAgent (off)' }]);
    assert.deepEqual(saved('api:openrouter@old-team'), [{ value: 'api:openrouter@old-team', label: 'OpenRouter · old-team (removed)' }]);
    assert.deepEqual(saved('api:openai'), [{ value: 'api:openai', label: 'OpenAI (no key)' }]);
    // An offered key is never listed twice by the rescue.
    assert.equal(saved('api:openrouter@claude-credits').length, 1);
    // The Models editor keeps the saved source on a row that names a turned-off key.
    const fallback = modelSourceGroups({ providers, connections, current: 'openai-compatible@gigaagent', providerProfiles: contract.providerProfiles });
    assert.equal(sourceChoice('openai-compatible@gigaagent'), 'api:openai-compatible@gigaagent');
    assert.ok(apiGroup(fallback).some((option) => option.value === 'api:openai-compatible@gigaagent' && /\(off\)$/.test(option.label)));
    assert.equal(sourceFromChoice('api:openrouter@claude-credits'), 'openrouter@claude-credits');
});

test('an added key round-trips through the editor-authored spelling `kind@name::model`', () => {
    const value = 'openrouter@claude-credits::anthropic/claude-opus-5.5';
    assert.deepEqual(parseModelSource(value), { source: 'openrouter@claude-credits', model: 'anthropic/claude-opus-5.5' });
    assert.equal(composeModelSource('openrouter@claude-credits', 'anthropic/claude-opus-5.5'), value);
    assert.equal(composeModelSource('openai-compatible@local-vllm', 'qwen3-coder'), 'openai-compatible@local-vllm::qwen3-coder');
    // The built-in key keeps today's spelling; its long form is the same source.
    assert.equal(composeModelSource('openrouter', 'anthropic/claude-opus-5.5'), 'anthropic/claude-opus-5.5');
    assert.equal(composeModelSource('anthropic', 'claude-opus-5.5'), 'anthropic::claude-opus-5.5');
    assert.deepEqual(parseModelSource('anthropic@default::claude-opus-5.5'), { source: 'anthropic', model: 'claude-opus-5.5' });
    assert.equal(parseModelSource('openrouter@default::openai/gpt-6').source, 'openrouter');
    // The source choice carries the key and never a `::`; picking a key leaves an empty draft on it.
    assert.equal(encodeRouteChoice({ route: api(value) }), 'api:openrouter@claude-credits');
    assert.deepEqual(decodeRouteChoice('api:openrouter@claude-credits'), { kind: 'api_model', provider: 'openrouter@claude-credits' });
    assert.deepEqual(changeRouteChoice(api('anthropic/claude-opus-5.5'), 'api:openrouter@claude-credits'),
        { kind: 'api_model', target_id: 'openrouter@claude-credits::' });
    assert.equal(routeTargetFromModel(api('openrouter@claude-credits::'), 'openai/gpt-6'), 'openrouter@claude-credits::openai/gpt-6');
    assert.equal(routeTargetFromModel(api(value), ''), 'openrouter@claude-credits::');
    const fields = routeModelFields(api(value), [], { connections: apiConnections(SETTINGS) });
    assert.equal(fields.model, 'anthropic/claude-opus-5.5');
    assert.equal(fields.provider, 'openrouter@claude-credits');
    assert.equal(fields.providerLabel, 'OpenRouter · Claude credits');
    // An API key is a source, never a subscription account: the Account select stays off it.
    assert.equal(routeSupportsAccount(api(value)), false);
});

test("the model chooser lists the chosen key's own catalog; the same model on two keys stays two entries", () => {
    const catalog = [
        { value: 'anthropic/claude-opus-5.5', provider: 'openrouter' },
        { value: 'openrouter@claude-credits::anthropic/claude-opus-5.5', provider: 'openrouter@claude-credits' },
        { value: 'openrouter@claude-credits::openai/gpt-6', provider: 'openrouter@claude-credits' },
        { value: 'openai-compatible@local-vllm::qwen3-coder', provider: 'openai-compatible@local-vllm' },
    ];
    assert.deepEqual(routeModelSuggestions(api('openrouter@claude-credits::'), catalog), ['anthropic/claude-opus-5.5', 'openai/gpt-6']);
    assert.deepEqual(routeModelSuggestions(api(''), catalog), ['anthropic/claude-opus-5.5']);
    assert.deepEqual(routeModelSuggestions(api('openai-compatible@local-vllm::'), catalog), ['qwen3-coder']);
    const html = routeModelInputHtml('data-model', api('openrouter@claude-credits::'), catalog, 'models');
    assert.doesNotMatch(html, /openrouter@|::|qwen3/, 'the chooser shows models only, of this key only');
});

test("a route names its key in the owner's words: the chip, and the task card's model line", () => {
    const connections = apiConnections(SETTINGS);
    assert.equal(sourceIdentityLabel(api('openrouter@claude-credits::anthropic/claude-opus-5.5'), { connections }),
        'API · OpenRouter · Claude credits');
    assert.equal(sourceIdentityLabel(api('anthropic/claude-opus-5.5'), { connections }), 'API · OpenRouter');
    // Without the document the stored name still tells the keys apart.
    assert.equal(sourceIdentityLabel(api('openrouter@claude-credits::x')), 'API · OpenRouter · claude-credits');

    assert.equal(routeConnectionId('openrouter@claude-credits::anthropic/claude-opus-5.5'), 'openrouter@claude-credits');
    for (const builtin of ['anthropic/claude-opus-5.5', 'anthropic::claude-opus-5.5', 'openrouter@default::x',
        'claudexor::codex=gpt-6', 'model@tag', '']) assert.equal(routeConnectionId(builtin), '', builtin);
    assert.equal(connectionLabel('openrouter@claude-credits'), 'claude-credits', 'the stored name until a read names it');
    adoptConnectionLabels(CONNECTIONS);
    assert.equal(connectionLabel('openrouter@claude-credits'), 'Claude credits');
    assert.equal(connectionLabel(''), '');

    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }) };
    try {
        const card = (agentModel) => {
            const record = { finished: true, metaEl: { innerHTML: '', isConnected: true } };
            renderLiveCardMeta(record, { agentModel });
            return record.metaEl.innerHTML;
        };
        assert.match(card('openrouter@claude-credits::anthropic/claude-opus-5.5'), /claude-opus-5\.5 · Claude credits/);
        assert.doesNotMatch(card('anthropic/claude-opus-5.5'), /·\s*(Default|OpenRouter)/, 'the built-in key adds nothing');
        // An observed execution names the key it actually used, not the requested one.
        const record = { finished: true, metaEl: { innerHTML: '', isConnected: true }, modelExecution: {
            source: 'usable_solve_response', requested_model: 'anthropic/claude-opus-5.5',
            used_model: 'openai-compatible@local-vllm::qwen3-coder', reported_model: '' } };
        renderLiveCardMeta(record, { agentModel: 'anthropic/claude-opus-5.5' });
        assert.match(record.metaEl.innerHTML, /qwen3-coder[^<]*· Local vLLM/);
    } finally {
        globalThis.document = priorDocument;
        adoptConnectionLabels([]);
    }
});

test('Accounts renders one card per provider with its Default row, and an added row with its own fields', () => {
    const html = renderSettingsPage();
    assert.match(html, /Each role names the key it uses; keys never rotate\. For a second key of the same model, add it as a fallback\./);
    assert.equal([...html.matchAll(/data-connection-add="([^"]+)"/g)].length, PROVIDER_CARDS.length);
    assert.equal([...html.matchAll(/data-connection-row data-kind="[^"]+" data-builtin="1"/g)].length, PROVIDER_CARDS.length);
    // The built-in key keeps its flat setting, with one input per key on the whole page.
    for (const field of contract.providerFields.filter((field) => field.inputType === 'password')) {
        assert.equal(html.split(`data-secret-setting="${field.settingKey}"`).length - 1, 1, field.settingKey);
    }
    assert.match(html, /id="s-openrouter" name="OPENROUTER_API_KEY" data-secret-setting="OPENROUTER_API_KEY"/);
    // Provider keys have one input each, in Accounts: the Secrets tab keeps the other secrets only.
    const secrets = html.slice(html.indexOf('data-settings-panel="secrets"'), html.indexOf('data-settings-panel="models"'));
    for (const key of PROVIDER_SECRET_KEYS) assert.doesNotMatch(secrets, new RegExp(`data-secret-setting="${key}"`), key);
    assert.match(secrets, /data-secret-setting="GITHUB_TOKEN"/);
    assert.match(secrets, /data-secret-setting="OUROBOROS_NETWORK_PASSWORD"/);

    const compatible = PROVIDER_CARDS.find((card) => card.kind === 'openai-compatible');
    const row = connectionRowHtml(compatible, { uid: 'compatible-7', name: 'gigaagent', label: 'GigaAgent', enabled: false,
        values: { base_url: 'https://gigaagent.example.com/v1', api_key: 'ga-live-...' } });
    assert.match(row, /data-uid="compatible-7" data-name="gigaagent" data-enabled="0"/);
    assert.match(row, /<strong data-connection-title data-i18n-skip>GigaAgent<\/strong>/);
    assert.match(row, /data-connection-label value="GigaAgent"/);
    // A row secret arrives masked and is kept as shown; it is no flat setting.
    assert.match(row, /data-connection-field="api_key"[^>]*value="ga-live-\.\.\." data-applied-value="ga-live-\.\.\."[^>]*class="secret-input ui-control" type="password"/);
    assert.match(row, /data-connection-field="base_url"[^>]*value="https:\/\/gigaagent\.example\.com\/v1"/);
    assert.doesNotMatch(row, /data-secret-setting/);
    assert.match(row, />Enable<\/button>/);
    // Every field's accessible name says whose key it is.
    assert.match(row, /aria-label="OpenAI Compatible GigaAgent API key"/);
    const draft = connectionRowHtml(PROVIDER_CARDS[0], { uid: 'openrouter-8', draft: true });
    assert.match(draft, /data-draft="1"/);
    assert.match(draft, />New OpenRouter key</);
    assert.match(draft, /placeholder="e\.g\. Claude credits"/);
    // Values are escaped where they are written.
    assert.match(connectionRowHtml(PROVIDER_CARDS[0], { uid: 'x', label: '<b>"x"</b>' }), /value="&lt;b&gt;&quot;x&quot;&lt;\/b&gt;"/);
});

test('a row says what is true of its key now: status, use, and who provides it', () => {
    assert.deepEqual(connectionRowFacts({ enabled: true, configured: true, source: 'settings', usedBy: ['Main', 'Fallback 2'] }),
        { tone: 'ok', label: 'Ready', meta: 'Enabled · used by Main, Fallback 2' });
    assert.deepEqual(connectionRowFacts({ enabled: true, configured: true, source: 'environment' }),
        { tone: 'ok', label: 'Ready', meta: 'Provided by environment · not used by any role' });
    assert.deepEqual(connectionRowFacts({ enabled: false, configured: true, usedBy: ['Fallback 2'] }),
        { tone: 'neutral', label: 'Disabled', meta: 'Named by Fallback 2 · nothing is sent through it' });
    assert.deepEqual(connectionRowFacts({ enabled: false }),
        { tone: 'neutral', label: 'Disabled', meta: 'Not used by any role · nothing is sent through it' });
    assert.deepEqual(connectionRowFacts({ enabled: true, configured: false, usedBy: ['Light'] }),
        { tone: 'neutral', label: 'No key', meta: 'Used by Light' });
    assert.deepEqual(connectionRowFacts({ draft: true, usedBy: ['Main'] }), { tone: 'neutral', label: 'Not saved', meta: '' });

    const usage = connectionUsage({ ...SETTINGS, OUROBOROS_SUBAGENTS: { items: [
        { route: { kind: 'api_model', target_id: 'openrouter@claude-credits::openai/gpt-6' } },
        { route: { kind: 'agent_session', target_id: 'codex=gpt-6' } },
        { route: { kind: 'api_model', target_id: 'openai/gpt-6' } },
        { route: { kind: 'api_model', target_id: 'anthropic/claude-opus-5.5' } },
    ] } }, contract.modelSlots);
    // Roles by name, Agents rows as one count: a long roster never turns the line into a list.
    assert.deepEqual(usage.get('openrouter@claude-credits'), ['Main', '1 subagent']);
    assert.deepEqual(usage.get('openrouter'), ['Fallback 1', '2 subagents']);
    assert.deepEqual(usage.get('openai-compatible@gigaagent'), ['Fallback 2']);
    assert.equal(usage.has('subscription:codex'), false, 'subscriptions are not API keys');

    // A learned refusal of cache hints is a quiet fact, shown only when the server reports one.
    assert.equal(cacheHintsNote({ id: 'openai-compatible@local-vllm' }), '');
    assert.match(cacheHintsNote({ cache_hints: { refused: true, learned_at: '2026-10-20T10:00:00Z' } }),
        /^Cache hints: refused by this server since /);
    assert.equal(cacheHintsNote({ cache_hints: { refused: false } }), '');
});

test('a new key takes a stored name from its label once; the setting and the save receipt read safely', () => {
    assert.equal(connectionNameFor('Claude credits'), 'claude-credits');
    assert.equal(connectionNameFor('  Crédits — équipe 2 '), 'credits-equipe-2');
    assert.equal(connectionNameFor('Default'), 'default-2', '`default` is the built-in key');
    assert.equal(connectionNameFor('Claude credits', ['claude-credits', 'claude-credits-2']), 'claude-credits-3');
    assert.equal(connectionNameFor('Кредиты'), 'key', 'a label without latin letters still gets a valid name');
    assert.match(connectionNameFor('x'.repeat(60), ['x'.repeat(32)]), /^x{30}-2$/);
    for (const name of ['claude-credits', 'default-2', 'credits-equipe-2', 'key']) assert.match(name, /^[a-z0-9][a-z0-9_-]{0,31}$/);

    assert.deepEqual(connectionRows('[{"kind":"zai","name":"coding"}]'), [{ kind: 'zai', name: 'coding' }]);
    assert.deepEqual(connectionRows('not json'), []);
    assert.deepEqual(connectionRows([null, 'x', [], { kind: 'openai', name: 'b' }]), [{ kind: 'openai', name: 'b' }]);

    const connections = apiConnections(SETTINGS);
    assert.equal(connectionReferencesText([], connections), '');
    assert.equal(connectionReferencesText([
        { setting: 'OUROBOROS_MODEL', route: 'openrouter@claude-credits::anthropic/claude-opus-5.5', connection: 'claude-credits', reason: 'missing' },
        { setting: 'OUROBOROS_MODEL_FALLBACKS', entry: '2', route: 'openrouter@claude-credits::x', connection: 'claude-credits', reason: 'disabled' },
        { setting: 'OUROBOROS_SUBAGENTS', entry: 'reviewer-1', route: 'openrouter@claude-credits::y', connection: 'claude-credits', reason: 'missing' },
    ], connections).includes('Main ('), true);
    assert.match(connectionReferencesText([
        { setting: 'OUROBOROS_MODEL_FALLBACKS', entry: '2', route: 'openrouter@claude-credits::x', connection: 'claude-credits' },
    ], connections), /Fallback 2 \(/);
    assert.match(connectionReferencesText([
        { setting: 'OUROBOROS_SUBAGENTS', entry: 'reviewer-1', route: 'openrouter@claude-credits::y', connection: 'claude-credits' },
    ], connections), /Agents · reviewer-1 \(/);
    assert.doesNotMatch(connectionReferencesText([
        { setting: 'OUROBOROS_MODEL', route: 'openrouter@claude-credits::z', connection: 'claude-credits' },
    ], connections), /OUROBOROS_MODEL/);
    assert.equal(connectionReferencesText([{ label: 'Main', connection: 'openrouter@claude-credits' }, 'Subagent 2'], connections),
        'Still naming a removed or disabled key, so they cannot send: Main (OpenRouter · Claude credits), Subagent 2');
});

test('a wizard draft lists named endpoints without saved metadata, keeping identical model IDs distinct', () => {
    const settings = { OUROBOROS_CONNECTIONS: [
        { kind: 'openai-compatible', name: 'alpha', label: 'Local reasoning server', base_url: 'http://127.0.0.1:41001/v1' },
        { kind: 'openai-compatible', name: 'beta', label: 'Remote research server', base_url: 'http://127.0.0.1:41002/v1', api_key: 'b' },
        { kind: 'openai-compatible', name: 'off', base_url: 'http://127.0.0.1:41003/v1', enabled: false },
        { kind: 'openai-compatible', name: 'empty', api_key: 'no-url' },
    ] };
    const providers = configuredApiProviders(settings);
    assert.deepEqual(providers.map((entry) => entry.id), ['openai-compatible@alpha', 'openai-compatible@beta']);
    assert.equal(composeModelSource(providers[0].id, 'same-model'), 'openai-compatible@alpha::same-model');
    assert.equal(composeModelSource(providers[1].id, 'same-model'), 'openai-compatible@beta::same-model');
    assert.equal(apiConnections({ OPENAI_COMPATIBLE_BASE_URL: 'http://127.0.0.1:41001/v1',
        OUROBOROS_CONNECTIONS: [{ kind: 'openai-compatible', name: 'default', enabled: false }] })
        .find((entry) => entry.id === 'openai-compatible').enabled, false);
});
