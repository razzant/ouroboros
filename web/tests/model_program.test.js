import test from 'node:test';
import assert from 'node:assert/strict';
import { catalogPrograms, modelProgramMarkup } from '../modules/model_program.js';

const producer = {path: '/vendor/program', version: '2.0.0'};
for (const count of [1, 18]) test(`${count} accounts carry one catalog producer, never the inventory version`, () => {
    const catalog = {accounts: Array.from({length: count}, (_, index) => ({catalog: {credentialProfileId: `p${index}`, sourceProgram: producer, models: []}}))};
    const programs = catalogPrograms(catalog, {harness: 'future', label: 'Future'});
    assert.equal(programs.length, 1); assert.equal(programs[0].programs.length, 1);
    const html = modelProgramMarkup(programs[0], {entry: {selection: {path: '/vendor/program', version: '9.0.0'}, updateAvailable: 'no'}});
    assert.match(html, /List from Future 2.0.0/);
    assert.doesNotMatch(html, /9.0.0|Details|button|Up to date/);
    assert.deepEqual(catalogPrograms({accounts: [{catalog: {sourceProgram: null}}]}, {harness: 'future'}), []);
});
test('raw and native account catalogs filter pins, keep empty-list provenance and group distinct versions', () => {
    const lists = {account_catalogs: [{source: 'transport', accounts: ['one', 'two'].map((id, index) => ({catalog: {
        source: 'transport', credentialProfileId: id, sourceProgram: {...producer, version: String(index)}, models: [],
    }}))}]};
    const options = {sources: [{id: 'transport', credentialHarness: 'vendor'}], source: 'transport'};
    assert.equal(catalogPrograms(lists, options)[0].programs.length, 2);
    assert.equal(catalogPrograms(lists, {...options, profile: 'two'})[0].programs[0].version, '1');
    assert.deepEqual(catalogPrograms(lists, {...options, source: 'other'}), []);
});
test('newest verdict, refusal and operation outcome are engine facts, and text is escaped', () => {
    const program = {harness: 'opaque', label: 'Vendor <', programs: [producer]};
    for (const verdict of ['yes', 'cannot_tell', 'no']) {
        const html = modelProgramMarkup(program, {entry: {selection: {path: '/p'}, updateAvailable: verdict,
            newest: {status: 'available', version: '1'}, actions: {update: {available: true}}}});
        assert.equal(/>Update<\/button>/.test(html), verdict !== 'no');
        assert.doesNotMatch(html, />· /, 'wrapped clauses have no leading separator');
        assert.match(html, /Vendor &lt;/);
    }
    const html = modelProgramMarkup(program, {entry: {selection: {path: '/p'}, updateAvailable: 'yes', actions: {update: {available: false, reason: 'Clear the override'}}},
        operation: {state: 'succeeded', outcome: 'not_changed', limitations: ['Nothing was removed.'], progress: ['Vendor <says>']}});
    assert.match(html, /Clear the override/); assert.match(html, /Nothing was removed/); assert.match(html, /Nothing changed/);
    assert.match(html, /Vendor &lt;says&gt;/); assert.doesNotMatch(html, /<button/);
});

for (const count of [1, 18]) test(`mixed catalogs name the supplied subset; ${count} accounts do not multiply it`, () => {
    const items = Array.from({length: count}, (_, index) => ({id: 'vendor-model', value: 'claudexor::transport=vendor-model',
        source_id: 'transport', credential_profile_id: String(index), sourceProgram: producer}));
    const sources = [{id: 'transport', credentialHarness: 'codex'}];
    const api = {id: 'api-model', value: 'openai::api-model'};
    const pure = catalogPrograms({items, model_sources: sources}, {sources});
    assert.equal(pure.length, 1);
    assert.match(modelProgramMarkup(pure[0]), /List from Codex 2.0.0/);
    for (const extra of [api, {...api, source_id: 'transport', sourceProgram: null}]) {
        const mixed = catalogPrograms({items: [...items, extra]}, {sources});
        assert.equal(mixed.length, 1);
        assert.match(modelProgramMarkup(mixed[0]), /Models supplied by Codex 2.0.0/);
        assert.doesNotMatch(modelProgramMarkup(mixed[0]), /List from/);
    }
    const scoped = catalogPrograms({items: [...items, api]}, {source: 'transport', profile: '0', sources});
    assert.match(modelProgramMarkup(scoped[0]), /List from Codex 2.0.0/);
    const two = catalogPrograms({items: [...items, {id: 'second', credentialHarness: 'claude', sourceProgram: producer}]}, {sources});
    assert.equal(two.length, 2);
    assert.ok(two.every((group) => group.mixed));
});
test('native flat suggestions inherit only the models their supplying catalogs actually list', () => {
    const catalog = {model_catalog: {sourceProgram: producer, models: [{id: 'known'}]}, models: [{id: 'known'}]};
    const options = {harness: 'future', label: 'Future'};
    assert.match(modelProgramMarkup(catalogPrograms(catalog, options)[0]), /List from Future/);
    catalog.models.push({id: 'hint', sourceProgram: null});
    assert.match(modelProgramMarkup(catalogPrograms(catalog, options)[0]), /Models supplied by Future/);
    catalog.model_catalog.models.push({id: 'hint'});
    assert.match(modelProgramMarkup(catalogPrograms(catalog, options)[0]), /List from Future/);
});
