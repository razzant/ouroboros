import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { composeModelSource, createModelRolesEditor, modelContextNote,
    modelSourceGroups, parseModelSource, sourceChoice, sourceFromChoice } from '../modules/model_roles.js';
import { configuredApiProviders, routeChoiceGroups } from '../modules/route_editor_primitives.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));

test('source/model spelling is reversible and never embeds the account', () => {
    for (const value of ['google/a', 'openai::b', 'claudexor::codex=gpt-test', 'custom::x=y']) {
        const { source, model } = parseModelSource(value);
        assert.equal(composeModelSource(source, model), value);
    }
    assert.equal(composeModelSource('subscription:codex', ''), '');
    assert.equal(composeModelSource('openrouter', 'openai::owner-model'), 'openai::owner-model');
});

test('Models shows only model-capable sources and only providers with a stored key', () => {
    const providers = configuredApiProviders({ OPENAI_API_KEY: '***set***' }, contract.providerProfiles);
    const groups = modelSourceGroups({ sources: [{ id: 'codex', label: 'Codex' }], providers,
        providerProfiles: contract.providerProfiles });
    assert.deepEqual(groups[0].options.map((row) => row.value), ['subscription:codex']);
    assert.deepEqual(groups[1].options.map((row) => row.value), ['api:openai', '']);
    assert.equal(groups[1].options[0].label, 'OpenAI');
    // Every other provider is a key the owner has not added; the tail says where.
    assert.equal(groups[1].options.at(-1).disabled, true);
    assert.match(groups[1].options.at(-1).label, /Accounts/);
    // A model role cannot be delivered by an agent session, so that group is absent.
    assert.deepEqual(groups.map((group) => group.label), ['Subscriptions · models', 'API keys']);
    const saved = modelSourceGroups({ current: 'subscription:future' });
    assert.equal(saved[0].options[0].value, 'subscription:future');
    assert.match(saved[0].options[0].label, /not checked/);
});

test('Models and the route editors draw the same groups, in the same order, from one builder', () => {
    const args = { sources: [{ id: 'codex', label: 'Codex' }],
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }), catalogKnown: true, accountsKnown: true };
    const shared = routeChoiceGroups({ modelSources: args.sources, providers: args.providers,
        catalogKnown: true, accountsKnown: true, includeSessions: false });
    assert.deepEqual(modelSourceGroups(args), shared);
    // "Uses Main" is the only entry a model role adds on top of the shared list.
    const inherited = modelSourceGroups({ ...args, current: 'inherit' });
    assert.deepEqual(inherited[0].options, [{ value: 'inherit', label: 'Uses Main' }]);
    assert.deepEqual(inherited.slice(1), shared);
});

test('a saved source whose key was removed stays selectable and says the key is gone', () => {
    const groups = modelSourceGroups({ current: 'anthropic',
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }) });
    const api = groups.find((group) => group.label === 'API keys').options;
    assert.deepEqual(api.map((row) => row.value), ['api:openrouter', 'api:anthropic', '']);
    assert.equal(api[1].label, 'Anthropic (no key)');
    assert.ok(!api[1].disabled, 'the owner can still keep their own assignment');
});

test('the select vocabulary is shared while the stored spelling is unchanged', () => {
    for (const [source, choice] of [['openrouter', 'api:openrouter'], ['openai', 'api:openai'],
        ['subscription:codex', 'subscription:codex'], ['inherit', 'inherit']]) {
        assert.equal(sourceChoice(source), choice);
        assert.equal(sourceFromChoice(choice), source);
        assert.ok(!choice.includes('::'), choice);
    }
});

test('Models distinguishes an unread Accounts facet from a confirmed empty account set', () => {
    for (const facts of [{}, { catalogKnown: true }, { accountsKnown: true }]) {
        const groups = modelSourceGroups(facts);
        assert.doesNotMatch(groups[0].options[0].label, /connect one/i);
    }
    const noAccounts = modelSourceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: false });
    assert.match(noAccounts[0].options[0].label, /connect one in Accounts/i);
    const empty = modelSourceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: true });
    assert.match(empty[0].options[0].label, /No model sources listed/);
    assert.match(empty[0].options[0].label, /refresh Model Catalog/i);
    assert.doesNotMatch(empty[0].options[0].label, /connect one in Accounts/i);
    const saved = modelSourceGroups({ current: 'subscription:owner-source' });
    assert.equal(saved[0].options[0].value, 'subscription:owner-source');
});

test('role pins and context survive a no-edit save, including identical model names', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    const settings = {
        OUROBOROS_MODEL: 'claudexor::codex=gpt-test',
        OUROBOROS_MODEL_LIGHT: 'claudexor::codex=gpt-test',
        OUROBOROS_MODEL_FALLBACKS: 'claudexor::codex=gpt-second, openai::model, openai/gpt-5.6-terra',
        OUROBOROS_MODEL_ACCOUNTS: { main: 'personal', light: 'work', vision: 'work', fallback: ['reserve', '', ''], websearch: 'saved' },
        OUROBOROS_MODEL_CONTEXT_WINDOWS: { main: 1000000, fallback: [872000, 0, 0], deep_review: 250000 },
    };
    editor.load(settings, contract);
    const after = editor.collect();
    for (const [key, value] of Object.entries(settings)) assert.deepEqual(after[key], value, key);
    editor.adoptCatalog({ model_sources: [{ id: 'codex', label: 'Codex' }], items: [] });
    assert.deepEqual(editor.collect(), after, 'new discovery never rewrites the assignment');
    assert.equal(editor.validate(), '');
    editor.destroy();
});

test('image-operation model saves as a bare engine id, without unsupported chat-role options', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    editor.load({ OUROBOROS_MODEL: 'openai::model', OUROBOROS_MODEL_IMAGE: 'gpt-image-2' }, contract);
    const saved = editor.collect();
    assert.equal(saved.OUROBOROS_MODEL_IMAGE, 'gpt-image-2');
    assert.ok(!('image' in (saved.OUROBOROS_MODEL_ACCOUNTS || {})));
    assert.ok(!('image' in (saved.OUROBOROS_MODEL_CONTEXT_WINDOWS || {})));
    assert.ok(!('image' in (saved.OUROBOROS_MODEL_PROCESSING_PREFERENCES || {})));
    assert.equal(editor.validate(), '');
    editor.load({ OUROBOROS_MODEL: 'openai::model', OUROBOROS_MODEL_IMAGE: 'claudexor::codex=gpt-image-2' }, contract);
    assert.match(editor.validate(), /bare engine model id/);
    editor.destroy();
});

test('an unloaded editor authors nothing and an API-only save does not create role maps', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    assert.deepEqual(editor.collect(), {});
    editor.load({ OUROBOROS_MODEL: 'openai::model' }, contract);
    const setting = editor.collect();
    assert.equal(setting.OUROBOROS_MODEL, 'openai::model');
    assert.ok(!('OUROBOROS_MODEL_ACCOUNTS' in setting));
    assert.ok(!('OUROBOROS_MODEL_CONTEXT_WINDOWS' in setting));
    editor.destroy();
});

test('context Auto uses the exact advertised maximum, while manual values stay assertions', () => {
    assert.match(modelContextNote({ context_window: 272000, max_context_window: 872000 }), /872,000.*advertised/);
    assert.match(modelContextNote({ max_context_window: 872000 }, 1000000), /1,000,000.*set by you/);
    assert.match(modelContextNote(null), /not known/);
});

test('the same role sheets are loaded by both actual UI hosts', () => {
    for (const file of ['index.html', 'onboarding_template.html']) {
        const html = readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');
        for (const sheet of ['model_roles.css', 'reviewer_slots.css']) {
            assert.ok(html.includes(`href="/static/${sheet}"`), `${file} loads ${sheet}`);
        }
    }
});

test('role disclosures draw a visible open/closed marker on their own line', () => {
    const css = readFileSync(new URL('../model_roles.css', import.meta.url), 'utf8');
    const block = (selector) => css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`))?.[1] ?? '';
    // `display` other than list-item drops the native ::marker, so the sheet
    // must hide it explicitly and draw the house glyph pair itself.
    const summary = block('\\.model-role-details\\s*>\\s*summary');
    assert.match(summary, /list-style:\s*none/);
    assert.match(summary, /color:\s*var\(--text-primary\)/);
    assert.match(block('\\.model-role-details\\s*>\\s*summary::-webkit-details-marker'), /display:\s*none/);
    assert.match(block('\\.model-role-details\\s*>\\s*summary::before'), /content:\s*"▸ "/);
    assert.match(block('\\.model-role-details\\[open\\]\\s*>\\s*summary::before'), /content:\s*"▾ "/);
    // The disclosure owns a full row under the status text instead of sharing its baseline.
    assert.match(block('\\.model-role-notes'), /flex-wrap:\s*wrap/);
    assert.match(block('\\.model-role-notes\\s*>\\s*\\.model-role-details'), /flex-basis:\s*100%/);
});
