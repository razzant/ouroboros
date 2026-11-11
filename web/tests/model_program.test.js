import test from 'node:test';
import assert from 'node:assert/strict';
import { catalogPrograms, modelProgramMarkup } from '../modules/model_program.js';

const producer = {path: '/vendor/program', version: '2.0.0'};
for (const count of [1, 18, 44]) test(`${count} accounts carry one catalog producer, never the inventory version`, () => {
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

for (const count of [1, 18, 44]) test(`mixed catalogs name the supplied subset; ${count} accounts do not multiply it`, () => {
    const items = Array.from({length: count}, (_, index) => ({id: 'vendor-model', value: 'claudexor::transport=vendor-model',
        source_id: 'transport', source: 'Claudexor', credential_profile_id: String(index), sourceProgram: producer}));
    const sources = [{id: 'transport', credentialHarness: 'codex'}];
    const account_catalogs = [{source: 'transport', accounts: items.map((item) => ({catalog: {source: 'transport', credentialProfileId: item.credential_profile_id, sourceProgram: producer, models: [{id: item.id}]}}))}];
    for (const item of items) delete item.sourceProgram;
    const api = {id: 'api-model', value: 'openai::api-model'};
    const pure = catalogPrograms({items, account_catalogs, model_sources: sources}, {sources});
    assert.equal(pure.length, 1);
    assert.match(modelProgramMarkup(pure[0]), /List from Codex 2.0.0/);
    for (const extra of [api, {...api, source_id: 'transport', sourceProgram: null}]) {
        const mixed = catalogPrograms({items: [...items, extra], account_catalogs}, {sources});
        assert.equal(mixed.length, 1);
        assert.match(modelProgramMarkup(mixed[0]), /Claudexor entries supplied by Codex 2.0.0/);
        assert.doesNotMatch(modelProgramMarkup(mixed[0]), /List from/);
    }
    const scoped = catalogPrograms({items: [...items, api], account_catalogs}, {source: 'transport', profile: '0', sources});
    assert.match(modelProgramMarkup(scoped[0]), /List from Codex 2.0.0/);
    const two = catalogPrograms({items: [...items, {id: 'second', source_id: 'other'}], account_catalogs: [...account_catalogs, {source: 'other', accounts: [{catalog: {source: 'other', credentialHarness: 'claude', sourceProgram: producer, models: [{id: 'second'}]}}]}]}, {sources});
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

test('mixed provenance uses the actual entry source label and escapes it', () => {
    const [group] = catalogPrograms({items: [
        {id: 'vendor', source_id: 'transport', source: 'Reader <source>', credentialHarness: 'future', sourceProgram: producer},
        {id: 'api', source: 'Another source'},
    ]});
    assert.match(modelProgramMarkup(group), /Reader &lt;source&gt; entries supplied by/);
    assert.doesNotMatch(modelProgramMarkup(group), /Claudexor|List from|Reader <source>/);
});

test('opening renews a changed producer once; status subscription exists only for pending work', async (t) => {
    const {bindModelProgramFooter} = await import('../modules/model_program.js');
    const {claudexorStatus} = await import('../modules/claudexor_status_store.js');
    const doc = new EventTarget(); doc.getElementById = () => null;
    t.mock.method(globalThis, 'fetch', async (url, options = {}) => {
        let body = {};
        if (String(url).includes('/maintenance/harnesses')) body = {harnesses: [{harness: 'test-open', selection: {...producer, version}, actions: {update: {available: true}}, updateAvailable: 'yes', operation: receipt}]};
        else if (String(url).includes('/maintenance/operations')) {
            if (options.method === 'POST') receipt = {id: 'open-op', harness: 'test-open', state: 'running', termination: 'not_applicable'};
            body = receipt;
        } else { raw++; body = {items: []}; }
        return {ok: true, json: async () => body};
    });
    const oldDoc = globalThis.document; globalThis.document = doc;
    t.after(() => { globalThis.document = oldDoc; });
    let version = '2.0.0', raw = 0, native = 0, subscriptions = 0, releases = 0, poll, receipt = null;
    t.mock.method(claudexorStatus, 'refresh', async () => { native++; });
    t.mock.method(claudexorStatus, 'subscribe', (callback) => { subscriptions++; poll = callback; return () => { releases++; }; });
    const input = {dataset: {}, ownerDocument: {activeElement: null}};
    let click;
    const footer = {contains: () => false, addEventListener: (_, fn) => { click = fn; }, removeEventListener() {}};
    const binding = bindModelProgramFooter(input, footer, () => {});
    const group = {harness: 'test-open', label: 'Future', programs: [producer]};
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    binding.update([group]); binding.open(); await flush();
    assert.equal(native, 0); assert.equal(raw, 0); assert.equal(subscriptions, 0);
    binding.close(); version = '3.0.0'; binding.open(); await flush();
    assert.equal(native, 1); assert.equal(raw, 1);
    binding.update([{...group, programs: [{...producer, version}]}]);
    binding.close(); binding.open(); await flush();
    assert.equal(raw, 1); assert.equal(subscriptions, 0);
    click({target: {closest: () => ({dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'test-open'}})})}});
    await flush(); assert.equal(subscriptions, 1);
    receipt = {...receipt, state: 'succeeded', termination: 'confirmed', after: {...producer, version}};
    poll({loading: false}); await flush();
    assert.equal(releases, 1); assert.equal(raw, 2);
    binding.dispose();
});
