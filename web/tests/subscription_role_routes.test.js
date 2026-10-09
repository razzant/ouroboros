import assert from 'node:assert/strict';
import test from 'node:test';
import {
    changeRouteChoice, composeModelSource, decodeRouteChoice, encodeRouteChoice,
    parseModelSource, routeChoiceGroups, routeModelFields, routeModelSuggestions,
    routeSupportsAccount, routeTargetFromModel, serializeRouteSpec,
} from '../modules/route_editor_primitives.js';
import {
    availableSubagentRowMarkup, buildAvailableSubagentsSetting,
    parseAvailableSubagentsSetting, validateAvailableSubagentsSetting,
} from '../modules/subagents_settings.js';

const sources = [{ id: 'opaque-source', label: 'Subscription model', credentialHarness: 'codex' }];
const route = (kind = 'api_model', pin = 'personal') => ({
    kind, target_id: 'claudexor::opaque-source=gpt-test',
    [kind === 'api_model' ? 'credential_profile_id' : 'profile_id']: pin,
});
const actor = () => ({ subagent_id: 'native', recommended_use: 'Inspect the repository.', route: route() });
const roster = () => ({ enabled: true, items: [actor()] });

test('source parsing is shared with Models and never assumes source id equals harness', () => {
    assert.deepEqual(parseModelSource(route().target_id), { source: 'subscription:opaque-source', model: 'gpt-test' });
    assert.equal(composeModelSource('subscription:opaque-source', 'gpt-test'), route().target_id);
    assert.equal(routeModelFields(route(), sources).harness, 'codex');
    assert.equal(routeModelFields(route(), []).harness, '');
    assert.equal(routeSupportsAccount(route()), true);
    assert.equal(routeSupportsAccount({ kind: 'api_model', target_id: 'openai::gpt-test' }), false);
});

test('each delivery owner retains its own kind and account field', () => {
    for (const kind of ['api_model', 'api_chat']) {
        const field = kind === 'api_model' ? 'credential_profile_id' : 'profile_id';
        assert.deepEqual(serializeRouteSpec(route(kind), { apiKind: kind, credentialField: field }), route(kind));
        assert.deepEqual(serializeRouteSpec({ ...route(kind), target_id: 'openai::gpt-test' },
            { apiKind: kind, credentialField: field }), { kind, target_id: 'openai::gpt-test' });
    }
    const parsed = parseAvailableSubagentsSetting(JSON.stringify(roster()));
    assert.equal(parsed.error, '');
    assert.deepEqual(buildAvailableSubagentsSetting(parsed.setting), roster());
    assert.deepEqual(validateAvailableSubagentsSetting(roster()), []);
});

test('subscription source choices round-trip and do not hide a saved undiscovered source', () => {
    assert.equal(encodeRouteChoice({ route: route() }), 'subscription:opaque-source');
    assert.deepEqual(decodeRouteChoice('subscription:opaque-source'), { kind: 'api_model', source: 'opaque-source' });
    const groups = routeChoiceGroups({ modelSources: sources, harnesses: [{ id: 'cursor' }] });
    assert.deepEqual(groups.map((group) => group.label), ['Subscriptions · models', 'API keys', 'Agents · sessions']);
    const missing = routeChoiceGroups({ currentChoice: 'subscription:removed', catalogKnown: true });
    assert.match(missing[0].options[0].label, /not checked/);
});

test('model and account edits retain native versus packed delivery', () => {
    for (const kind of ['api_model', 'api_chat']) {
        const original = route(kind);
        assert.deepEqual(changeRouteChoice(original, 'subscription:opaque-source', { apiKind: kind }), original);
        assert.equal(routeTargetFromModel(original, 'gpt-next'), 'claudexor::opaque-source=gpt-next');
        assert.equal(routeTargetFromModel(original, ''), 'claudexor::opaque-source=');
        assert.deepEqual(changeRouteChoice(original, 'subscription:other', { apiKind: kind }),
            { kind, target_id: 'claudexor::other=' });
        assert.deepEqual(changeRouteChoice(original, 'api', { apiKind: kind }), { kind, target_id: '' });
    }
    assert.ok(validateAvailableSubagentsSetting({ ...roster(), items: [
        { ...actor(), route: { ...route(), target_id: 'claudexor::opaque-source=' } },
    ] }).length);
});

test('catalog suggestions are source-scoped without borrowing session inventory', () => {
    assert.deepEqual(routeModelSuggestions(route(), [
        'claudexor::opaque-source=gpt-test', 'openai::api-only', 'claudexor::other=not-this-source',
    ]), ['gpt-test']);
});

test('a reviewer row on a subscription keeps its pin and is priced as a seat, not in money', () => {
    const reviewer = { ...actor(), review_eligible: true };
    const parsed = parseAvailableSubagentsSetting({ enabled: true, items: [reviewer] });
    assert.equal(parsed.error, '');
    assert.equal(buildAvailableSubagentsSetting(parsed.setting).items[0].route.credential_profile_id, 'personal');
    const html = availableSubagentRowMarkup(reviewer, { catalogKnown: false, accountsKnown: false, modelSources: sources });
    assert.match(html, /data-subagent-review-facts>In the review pool · uses a session seat and time · reviews at high effort</);
    // Delivery belongs to the route kind: a subscription model call is an API-model row.
    assert.match(html, /data-subagent-field="delivery"/);
});

test('actor subscription controls use the mapped account family and preserve unlisted pins', () => {
    const state = { catalogKnown: true, accountsKnown: true, modelSources: sources,
        apiModels: [route().target_id], snapshot: { harnesses: [], profiles: { profiles: [
            { profile: { harness_id: 'codex', profile_id: 'work' } },
        ] } } };
    const html = availableSubagentRowMarkup(actor(), state);
    assert.match(html, /data-subagent-field="account"/);
    assert.match(html, /value="work"/);
    assert.match(html, /value="personal" selected/);
    assert.match(html, /value="gpt-test"/);
    assert.match(html, /value="subscription:opaque-source" selected/);
    // The chip names the SOURCE and wears its CREDENTIAL HARNESS's mark — the
    // opaque source id is never split as if it were a session target.
    assert.match(html, /data-harness-identity="codex"/);
    assert.match(html, />Subscription model · model<\/span>/);
    const unknown = availableSubagentRowMarkup(actor(), { ...state, modelSources: [] });
    assert.match(unknown, /personal \(not checked\)/);
    // With no catalog to map the id, the chip falls back to the source id
    // itself and still never claims a harness it could not resolve.
    assert.match(unknown, /data-harness-identity="opaque-source"/);
});
