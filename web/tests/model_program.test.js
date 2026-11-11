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
        if (String(url).includes('/maintenance/harnesses')) {
            inventories++;
            body = {harnesses: [{harness: 'test-open', selection: {path, version}, actions: {update: {available: true}}, updateAvailable: 'yes', operation: receipt}]};
        }
        else if (String(url).includes('/maintenance/operations')) {
            if (options.method === 'POST') receipt = {id: 'open-op', harness: 'test-open', state: 'running', termination: 'not_applicable'};
            body = receipt;
        } else { raw++; body = {items: []}; }
        return {ok: true, json: async () => body};
    });
    const oldDoc = globalThis.document; globalThis.document = doc;
    t.after(() => { globalThis.document = oldDoc; });
    let path = producer.path, version = '2.0.0', raw = 0, native = 0, inventories = 0, subscriptions = 0, releases = 0, poll, receipt = null;
    t.mock.method(claudexorStatus, 'refresh', async () => { native++; return {harnesses: [{id: 'test-open', model_catalog: {partial: false}}]}; });
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
    assert.equal(inventories, 2);
    binding.update([{...group, programs: [{...producer, version}]}]); await flush();
    assert.equal(inventories, 2, 'renewed lists already match the inventory; newest is read only once');
    binding.update([{...group, label: 'Renamed', programs: [{...producer, version}]}]); await flush();
    assert.equal(inventories, 2, 'presentation changes need no inventory');
    version = '4.0.0';
    binding.update([{...group, programs: [{...producer, version}]}]); await flush();
    assert.equal(inventories, 3, 'a new producer different from the held selection is observed');
    assert.equal(raw, 1, 'a list that already names the selected program needs no renewal');
    binding.close(); binding.open(); await flush();
    assert.equal(raw, 1); assert.equal(subscriptions, 0);
    path += '-replacement'; binding.close(); binding.open(); await flush();
    assert.equal(raw, 2, 'a path change also renews the list');
    binding.update([{...group, programs: [{path, version}]}]);
    click({target: {closest: () => ({dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'test-open'}})})}});
    await flush(); assert.equal(subscriptions, 1);
    receipt = {...receipt, state: 'succeeded', termination: 'confirmed', after: {path, version}};
    poll({loading: false}); await flush();
    assert.equal(releases, 1); assert.equal(raw, 3);
    binding.dispose();
});

for (const failure of ['raw', 'native']) test(`failed ${failure} refresh retries at the next opening; concurrent choosers share it`, async (t) => {
    const {bindModelProgramFooter} = await import('../modules/model_program.js');
    const {claudexorStatus} = await import('../modules/claudexor_status_store.js');
    const {refreshMaintenanceCatalogs} = await import('../modules/settings_catalog.js');
    const harness = `retry-${failure}`, current = {...producer, version: '3.0.0'};
    const doc = new EventTarget(); doc.getElementById = () => null;
    const oldDoc = globalThis.document; globalThis.document = doc;
    t.after(() => { globalThis.document = oldDoc; });
    let fail = true, raw = 0, native = 0;
    t.mock.method(globalThis, 'fetch', async (url) => {
        if (String(url).includes('/maintenance/harnesses')) return {ok: true, json: async () => ({harnesses: [{harness, selection: current, updateAvailable: 'no'}]})};
        raw++;
        if (fail && failure === 'raw') throw new Error('Catalog offline');
        return {ok: true, json: async () => ({items: []})};
    });
    t.mock.method(claudexorStatus, 'refresh', async () => {
        native++;
        return {harnesses: [{id: harness, model_catalog: {partial: fail && failure === 'native'}}]};
    });
    const make = () => bindModelProgramFooter({dataset: {}, ownerDocument: {activeElement: null}},
        {contains: () => false, addEventListener() {}, removeEventListener() {}}, () => {});
    const bindings = [make(), make()], group = {harness, label: 'Future', programs: [producer]};
    const flush = async () => { for (let n = 0; n < 100; n++) await Promise.resolve(); };
    t.after(() => bindings.forEach((binding) => binding.dispose()));
    bindings.forEach((binding) => { binding.update([group]); binding.open(); }); await flush();
    assert.equal(raw, 1); assert.equal(native, 1, 'two open choosers share the read');
    await flush(); assert.equal(raw, 1, 'failure causes no retry loop while open');
    bindings.forEach((binding) => binding.close()); fail = false;
    bindings.forEach((binding) => binding.open()); await flush();
    assert.equal(raw, 2); assert.equal(native, 2);
    bindings.forEach((binding) => binding.update([{...group, programs: [current]}])); await flush();
    bindings.forEach((binding) => { binding.close(); binding.open(); }); await flush();
    assert.equal(raw, 2, 'successful lists carry the current producer, needing no memory');
    const receipt = {id: `receipt-${failure}`, after: current, finishedAt: 'today'};
    fail = true;
    assert.ok(!await refreshMaintenanceCatalogs(harness, receipt));
    fail = false;
    assert.ok(await refreshMaintenanceCatalogs(harness, current), 'a failed settlement cannot silence chooser recovery');
    await refreshMaintenanceCatalogs(harness, receipt);
    const successful = raw;
    await refreshMaintenanceCatalogs(harness, receipt);
    assert.equal(raw, successful, 'only successful settlement reads are remembered');
});

test('an already-open footer sees Accounts observations immediately, before its own read answers', async (t) => {
    const {bindModelProgramFooter} = await import('../modules/model_program.js');
    const {createHarnessMaintenanceController} = await import('../modules/harness_maintenance.js');
    const harness = 'shared-observation';
    let version = '3.0.0', release;
    const row = () => ({harness, selection: {...producer, version}, updateAvailable: version === '3.0.0' ? 'yes' : 'no',
        newest: {status: 'available', version: '4.0.0'}, actions: {update: {available: true}}});
    const accounts = createHarnessMaintenanceController({api: {harnessMaintenanceInventory: async () => ({harnesses: [row()]})}});
    const input = {dataset: {}, ownerDocument: {activeElement: null}};
    const footer = {contains: () => false, addEventListener() {}, removeEventListener() {}};
    const binding = bindModelProgramFooter(input, footer, () => {});
    t.mock.method(globalThis, 'fetch', () => new Promise(resolve => { release = resolve; }));
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    await accounts.refresh();
    binding.update([{harness, label: 'Future', programs: [producer]}]); binding.open(); await flush();
    assert.match(footer.innerHTML, /4.0.0 available/);
    version = '4.0.0'; await accounts.refresh({fresh: true});
    assert.doesNotMatch(footer.innerHTML, /4.0.0 available|>Update</);
    assert.match(footer.innerHTML, /List from Future 2.0.0/, 'inventory never relabels the old list');
    binding.dispose(); accounts.dispose();
    release({ok: false, json: async () => ({error: 'Fixture read closed'})}); await flush();
});

test('an unanswered native catalog releases the shared read at its bound; reopening can recover', async (t) => {
    const {bindModelProgramFooter} = await import('../modules/model_program.js');
    const {claudexorStatus} = await import('../modules/claudexor_status_store.js');
    const {MODEL_CATALOG_TIMEOUT_MS} = await import('../modules/settings_catalog.js');
    const harness = 'hung-native', current = {...producer, version: '3.0.0'};
    const doc = new EventTarget(); doc.getElementById = () => null;
    const oldDoc = globalThis.document; globalThis.document = doc;
    t.after(() => { globalThis.document = oldDoc; });
    t.mock.timers.enable({apis: ['setTimeout']});
    let raw = 0;
    const native = [];
    t.mock.method(globalThis, 'fetch', async (url) => {
        if (String(url).includes('/maintenance/harnesses')) return {ok: true, json: async () => ({harnesses: [{harness, selection: current}]})};
        raw++;
        return {ok: true, json: async () => ({items: []})};
    });
    t.mock.method(claudexorStatus, 'refresh', () => new Promise(resolve => native.push(resolve)));
    const binding = bindModelProgramFooter({dataset: {}, ownerDocument: {activeElement: null}},
        {contains: () => false, addEventListener() {}, removeEventListener() {}}, () => {});
    t.after(() => binding.dispose());
    const group = {harness, label: 'Future', programs: [producer]};
    const flush = async () => { for (let n = 0; n < 100; n++) await Promise.resolve(); };
    const reopen = async () => { binding.close(); binding.open(); await flush(); };
    binding.update([group]); binding.open(); await flush();
    assert.equal(raw, 1); assert.equal(native.length, 1);
    t.mock.timers.tick(MODEL_CATALOG_TIMEOUT_MS - 1); await flush(); await reopen();
    assert.equal(raw, 1, 'openings share the pending read until its deadline');
    t.mock.timers.tick(1); await flush();
    assert.equal(raw, 1, 'expiry itself never retries');
    await reopen();
    assert.equal(raw, 2); assert.equal(native.length, 2, 'an opening starts a fresh pair of reads');
    const answer = {harnesses: [{id: harness, model_catalog: {partial: false}}]};
    native[0](answer); await flush(); await reopen();
    assert.equal(raw, 2, 'a late answer cannot retire the newer shared read');
    native[1](answer); await flush();
    binding.update([{...group, programs: [current]}]); await flush(); await reopen();
    assert.equal(raw, 2, 'a recovered list needs no further catalog read');
});
