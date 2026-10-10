import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { accountScopedModelCatalog, catalogModelOptions, harnessModelsKnown, modelsGapNote,
    routeModelSuggestions, routeModelInputHtml, sessionModelOptions } from '../modules/route_editor_primitives.js';
import { mergeModelCatalog, mergeHarnessModelCatalog } from '../modules/settings_catalog.js';
import { rowMeta, sessionRouteVerdict } from '../modules/subagent_status_primitives.js';

const entries = [
    { value: 'claudexor::codex=only-a', id: 'only-a', source_id: 'codex', credential_profile_id: 'personal',
        context_window: 272000, processing: { modes: ['standard'] }, availability: 'available', observed_at: '2026-09-12T00:00:00Z' },
    { value: 'claudexor::codex=only-b', id: 'only-b', source_id: 'codex', credential_profile_id: 'work',
        context_window: 1000000, processing: { modes: ['standard', 'fast'] }, availability: 'unavailable', observed_at: null },
];
const accounts = entries.map((item) => ({ credentialProfileId: item.credential_profile_id,
    availability: item.availability, problem: null, catalog: { models: [{ id: item.id }] } }));
const harness = { id: 'codex', enabled: true, status: 'ok',
    models: entries.map(({ value, ...item }) => item),
    model_catalog: { harnessId: 'codex', accounts, partial: false } };

test('API-model and reviewer pins filter per-account entries; Auto keeps every model', () => {
    for (const [kind, pinKey] of [['api_model', 'credential_profile_id'], ['api_chat', 'profile_id']]) {
        const route = { kind, target_id: 'claudexor::codex=owner-custom', [pinKey]: 'personal' };
        assert.deepEqual(routeModelSuggestions(route, entries), ['only-a']);
        assert.deepEqual(routeModelSuggestions({ ...route, [pinKey]: '' }, entries), ['only-a', 'only-b']);
        const html = routeModelInputHtml('data-model', route, entries, 'models');
        assert.match(html, /value="owner-custom"/);
        assert.match(html, /value="only-a"/);
        assert.doesNotMatch(html, /personal|work|available|date unknown/);
        assert.doesNotMatch(html, /only-b|work/);
        assert.equal(route[pinKey], 'personal');
    }
});

test('duplicate chooser values collapse to one model suggestion that makes no account claim', () => {
    const repeated = entries.map((item) => ({ ...item, value: 'same', id: 'same', name: 'Same model' }));
    const options = catalogModelOptions(repeated);
    assert.equal(options.length, 1);
    assert.equal(options[0].value, 'same');
    assert.equal(options[0].label, 'Same model');
    assert.doesNotMatch(options[0].label, /personal|work|available|unavailable|date unknown|272000|1000000|Fast|free/);
    assert.deepEqual(repeated.map((item) => item.context_window), [272000, 1000000]);
});

test('an option label is byte-identical with one account and with eighteen', () => {
    const eighteen = Array.from({ length: 18 }, (_, index) => ({
        value: 'same', id: 'same', name: 'Same model', credential_profile_id: `acct-${index + 1}`,
        availability: index % 2 ? 'unavailable' : 'available', observed_at: null,
    }));
    const many = catalogModelOptions(eighteen);
    assert.equal(many.length, 1);
    assert.equal(many[0].label, catalogModelOptions([eighteen[0]])[0].label);
    assert.equal(many[0].label.length, 'Same model'.length);
});

test('an alias names its resolution and a frozen-list row says so; the value stays the row id', () => {
    const alias = { id: 'default', label: 'Default (recommended)', origin: 'live', resolved_model: 'claude-opus-5-5[1m]' };
    const hint = { id: 'claude-sonnet-5', label: null, origin: 'hint', resolved_model: null };
    const [aliasOption, hintOption] = catalogModelOptions([alias, hint]);
    assert.deepEqual(aliasOption, { value: 'default', label: 'default → claude-opus-5-5[1m]' });
    assert.deepEqual(hintOption, { value: 'claude-sonnet-5', label: 'claude-sonnet-5 (shipped list)' });
    // The one projection feeds the session chooser too: the value is what travels.
    assert.deepEqual(sessionModelOptions({ models: [alias, hint] }, 'default').slice(1), [aliasOption, hintOption]);
    // Other direction: a row without the fields (a 3.13 engine), a live row, a resolution
    // equal to the id and an empty one all label exactly as before.
    for (const row of [{ id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol' }, { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol', origin: 'live' },
        { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol', resolved_model: 'gpt-5.6-sol' }, { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol', resolved_model: '' }]) {
        assert.deepEqual(catalogModelOptions([row]), [{ value: 'gpt-5.6-sol', label: 'GPT-5.6 Sol' }]);
    }
    assert.deepEqual(catalogModelOptions(['plain-id']), [{ value: 'plain-id', label: 'plain-id' }]);
    // Account-free: eighteen supplying accounts label byte-identically to one; the suffix
    // needs every supplier on the frozen list, and a disputed resolution is not shown.
    for (const row of [alias, hint]) {
        const eighteen = Array.from({ length: 18 }, (_, index) => ({ ...row, credential_profile_id: `acct-${index + 1}`,
            availability: index % 2 ? 'unavailable' : 'available', observed_at: '2026-09-24T00:00:00Z' }));
        assert.equal(catalogModelOptions(eighteen)[0].label, catalogModelOptions([eighteen[0]])[0].label);
        assert.doesNotMatch(catalogModelOptions(eighteen)[0].label, /acct|18|available|2026-09-24/);
    }
    assert.equal(catalogModelOptions([hint, { ...hint, origin: 'live' }])[0].label, 'claude-sonnet-5');
    assert.equal(catalogModelOptions([alias, { ...alias, resolved_model: 'claude-sonnet-5' }])[0].label, 'Default (recommended)');
});

test('a nameless first duplicate yields to a later name, and otherwise the value is the label', () => {
    const nameless = { value: 'same', id: 'same', name: undefined, label: undefined, credential_profile_id: 'personal' };
    assert.equal(catalogModelOptions([nameless, { ...nameless, name: 'Named' }])[0].label, 'Named');
    assert.equal(catalogModelOptions([nameless, { ...nameless, credential_profile_id: 'work' }])[0].label, 'same');
});

test('native pins select exact catalog and a missing account read does not prove model absence', () => {
    const failed = structuredClone(harness);
    failed.model_catalog.partial = true;
    failed.model_catalog.accounts[0] = { credentialProfileId: 'personal', availability: 'unknown', catalog: null,
        problem: { code: 'model_catalog_unavailable', message: 'Personal catalog unread' } };
    failed.models = failed.models.filter((item) => item.credential_profile_id === 'work');
    const personal = accountScopedModelCatalog(failed, 'personal');
    assert.equal(harnessModelsKnown(personal), false);
    assert.equal(modelsGapNote(personal), 'model list could not be read');
    assert.match(sessionModelOptions(personal, 'owner-custom').at(-1).label, /not checked/);
    assert.doesNotMatch(JSON.stringify(sessionModelOptions(personal, 'owner-custom')), /not in discovery|only-b/);
    const work = accountScopedModelCatalog(failed, 'work');
    assert.equal(harnessModelsKnown(work), true, 'the sibling failure does not withdraw this catalog read');
    assert.deepEqual(sessionModelOptions(work, '').map((item) => item.value), ['', 'only-b']);
    assert.deepEqual(accountScopedModelCatalog(harness).models.map((item) => item.id), ['only-a', 'only-b']);
    const empty = structuredClone(harness);
    empty.models = empty.models.filter((item) => item.credential_profile_id !== 'personal');
    empty.model_catalog.accounts[0].catalog.models = [];
    assert.match(sessionModelOptions(accountScopedModelCatalog(empty, 'personal'), 'owner-custom').at(-1).label, /not in discovery/);
});

test('session availability does not borrow a sibling model or treat a failed pin catalog as ready', () => {
    const snapshot = JSON.parse(readFileSync(new URL('./fixtures/subscription_setup.json', import.meta.url))).status;
    snapshot.harnesses = [structuredClone(harness)];
    snapshot.quota = [{ subject: { harness: 'codex', subject_id: 'personal' }, freshness: 'fresh', constraints: [] }];
    const state = { snapshot, catalogKnown: true, accountsKnown: true, quotaKnown: true };
    const row = { route: { kind: 'agent_session', target_id: 'codex=only-a', credential_profile_id: 'personal' } };
    assert.equal(sessionRouteVerdict(row, state).label, 'Available');
    row.route.target_id = 'codex=only-b';
    assert.equal(sessionRouteVerdict(row, state).label, 'Unavailable');
    snapshot.harnesses[0].model_catalog.accounts[0].catalog = null;
    snapshot.harnesses[0].model_catalog.partial = true;
    assert.equal(sessionRouteVerdict(row, state).label, 'Not checked');
    assert.equal(row.route.credential_profile_id, 'personal');
});

test('partial refresh replaces a successful empty account while retaining only the failed account', () => {
    const envelope = { source: 'codex', accounts: [
        { credentialProfileId: 'personal', availability: 'available', problem: null, catalog: { models: [] } },
        { credentialProfileId: 'work', availability: 'unknown', problem: { message: 'Work read failed' }, catalog: null },
    ], partial: true };
    const merged = mergeModelCatalog({ items: entries }, { items: [], account_catalogs: [envelope], partial: true,
        errors: [{ credential_profile_id: 'work', error: 'Work read failed' }] });
    assert.deepEqual(merged.items.map((item) => item.id), ['only-b']);
    assert.equal(merged.items[0].availability, 'unknown');
    assert.equal(merged.items[0].problem.message, 'Work read failed');
    assert.equal(merged.items[0].observed_at, null);
    const native = mergeHarnessModelCatalog(harness, { ...harness, models: [], model_catalog: { ...envelope, harnessId: 'codex' } });
    assert.deepEqual(native.models.map((item) => item.id), ['only-b']);
    assert.equal(harnessModelsKnown(accountScopedModelCatalog(native, 'work')), false);
    const recovered = mergeHarnessModelCatalog(native, harness);
    assert.equal(harnessModelsKnown(accountScopedModelCatalog(recovered, 'work')), true);
    assert.deepEqual(recovered.models.map((item) => item.id), ['only-a', 'only-b']);
});

test('older catalogs preserve their usable suggestions without inventing per-account proof', () => {
    const legacy = { id: 'codex', models: [{ id: 'legacy' }] };
    assert.equal(accountScopedModelCatalog(legacy, 'personal'), legacy);
    assert.deepEqual(routeModelSuggestions({ kind: 'api_model', target_id: 'claudexor::codex=legacy', credential_profile_id: 'personal' },
        [{ value: 'claudexor::codex=legacy' }]), ['legacy']);
});

test('Claude [1m] availability keeps account evidence and selected bytes across consumers', () => {
    const snapshot = JSON.parse(readFileSync(new URL('./fixtures/subscription_setup.json', import.meta.url))).status;
    snapshot.harnesses = [{ id: 'claude', enabled: true, status: 'ok', models: [
        { id: 'claude-future[1m]', credential_profile_id: 'failed' },
        { id: 'claude-future', credential_profile_id: 'personal' },
        { id: 'alias', resolved_model: 'claude-resolved[1m]', credential_profile_id: 'personal' },
    ] }];
    snapshot.profiles.profiles.forEach((row) => { row.profile.harness_id = 'claude'; });
    snapshot.quota = [{ subject: { harness: 'claude', subject_id: 'personal' }, freshness: 'fresh', constraints: [] }];
    const state = { snapshot, catalogKnown: true, accountsKnown: true, quotaKnown: true };
    const row = { route: { kind: 'agent_session', target_id: 'claude=claude-future[1M]', credential_profile_id: '' } };
    const verdict = sessionRouteVerdict(row, state);
    assert.equal(verdict.label, 'Available');
    assert.match(verdict.text, /Base model listed/);
    assert.equal(row.route.target_id, 'claude=claude-future[1M]');
    // The qualifier's meta line is flagged to wrap (readable without hover); other lines are not.
    const metaState = { ...state, setting: { items: [row] } };
    assert.equal(rowMeta(row, metaState, []).qualifier, true);
    assert.match(rowMeta(row, metaState, []).text, /checked by the engine at session start/);
    const exact = { route: { ...row.route, target_id: 'claude=claude-future' } };  // listed exactly: no qualifier
    assert.equal(rowMeta(exact, { ...state, setting: { items: [exact] } }, []).qualifier, undefined);
    const options = sessionModelOptions(snapshot.harnesses[0], 'claude-future[1m]', { snapshot });
    assert.match(options.find((option) => option.value === 'claude-future[1m]').label, /base model listed/);
    assert.match(sessionRouteVerdict({ route: { ...row.route, credential_profile_id: 'personal' } }, state).text, /Base model listed/);
    assert.equal(sessionRouteVerdict({ route: { ...row.route, credential_profile_id: 'work' } }, state).label, 'Unavailable');
    const onlyFailed = { ...state, snapshot: { ...snapshot, harnesses: [{ ...snapshot.harnesses[0], models: [snapshot.harnesses[0].models[0]] }] } };
    assert.equal(sessionRouteVerdict(row, onlyFailed).label, 'No account');
    const baseOnly = { ...snapshot.harnesses[0], models: snapshot.harnesses[0].models.slice(1) };
    assert.match(sessionModelOptions(baseOnly, 'claude-future[1M]').at(-1).label, /base model listed/);
    assert.doesNotMatch(sessionModelOptions(baseOnly, 'claude-resolved[1m]').at(-1).label, /not in discovery/);
    for (const selected of ['claude-future[other]', 'claude-absent']) {
        assert.equal(sessionRouteVerdict({ route: { ...row.route, target_id: `claude=${selected}` } }, state).label, 'Unavailable');
    }
    const unread = { ...state, snapshot: { ...snapshot, harnesses: [{ ...baseOnly, models_error: 'unread' }] } };
    assert.equal(sessionRouteVerdict(row, unread).label, 'Not checked');
    assert.match(sessionModelOptions({ id: 'codex', models: [{ id: 'future' }] }, 'future[1m]').at(-1).label, /not in discovery/);
});
