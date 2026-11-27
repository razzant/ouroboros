import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
        assert.equal(html.includes('class="harness-model-latest"'), verdict !== 'no');
        assert.equal(html.match(/<span class="harness-model-latest">([^<]+)<\/span>/)?.[1] || '',
            { yes: '1 available', cannot_tell: 'Latest unknown', no: '' }[verdict],
            'the latest-version fact stays one clause beside Update when the footer wraps');
        assert.match(html, /Vendor &lt;/);
    }
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /\.harness-model-latest\s*\{\s*white-space: nowrap;\s*\}/);
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
    poll({loading: true}); await flush();
    assert.equal(releases, 1, 'maintenance settles even while independent full discovery is loading'); assert.equal(raw, 3);
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

 test('the producer and the available version read as one fact, its separator never starting a line', () => {
    const html = modelProgramMarkup({ harness: 'codex', label: 'Codex', programs: [{version: '0.162.1'}] },
        { entry: { selection: { path: '/codex' }, updateAvailable: 'yes', newest: { status: 'available', version: '0.169.0' } } });
    assert.match(html, /<span>List from Codex 0\.162\.1\u00a0· <span class="harness-model-latest">0\.169\.0 available<\/span><\/span>/);
 });

// The setup wizard's family line (docs/DESIGN.md §7): the same builder, led by the program's own version, shown only
// with something to act on once the newest-release read has answered, and read only while the engine runs.
const familyEntry = (harness, extra = {}) => ({harness, selection: {path: `/vendor/${harness}`, version: '0.50.0'},
    newest: {status: 'available', version: '0.60.0', observedAt: 'now', source: 'registry'}, updateAvailable: 'yes',
    actions: {update: {available: true, reason: null}, install: {available: true, reason: null}}, operation: null, ...extra});

test('a family line shows what there is to act on and stays silent otherwise', async () => {
    const {familyProgramShown} = await import('../modules/model_program.js');
    const shown = (entry, view = {}) => familyProgramShown({entry, ...view});
    for (const verdict of ['yes', 'cannot_tell']) {
        const entry = familyEntry('f', {updateAvailable: verdict});
        assert.equal(shown(entry), true, verdict);
        const html = modelProgramMarkup({harness: 'f'}, {entry}, 'Program 0.50.0');
        assert.match(html, /<span>Program 0\.50\.0\u00a0· /);
        assert.match(html, />Update<\/button>/);
        assert.doesNotMatch(html, /List from|Details/);
    }
    assert.equal(shown(familyEntry('f', {updateAvailable: 'no'})), false, 'up to date says nothing here');
    assert.equal(shown(familyEntry('f', {selection: {path: null, version: null}})), false, 'Connect installs a missing program');
    assert.equal(shown(familyEntry('f', {newest: {status: 'not_read', source: null}})), false, 'the line waits for the newest read');
    // W5 (rl-20261127T050408): a refused read of this family is its own applicable fact — reason + Check again;
    // only the whole list's refusal stays out of the row (its one notice lives in `others()`).
    assert.equal(shown(familyEntry('f'), {inspectionError: 'Program inspection unavailable: refused'}), true);
    assert.equal(shown(familyEntry('f'), {inspectionError: 'x', inventoryError: 'y'}), false,
        'the engine-wide notice is not repeated on the row');
    assert.equal(shown(familyEntry('f'), {inventoryError: 'Program inspection unavailable: refused'}), false);
    const refusedLine = modelProgramMarkup({harness: 'f'}, {entry: familyEntry('f'), inspectionError: 'Program inspection unavailable: refused'}, 'Program 0.50.0');
    assert.match(refusedLine, /data-tone="warn">Program inspection unavailable: refused/);
    assert.match(refusedLine, /data-program-action="inspect">Check again</);
    assert.doesNotMatch(refusedLine, /data-program-action="(install|latest|rejoin)"/, 'no press stands on the refused read');
    const refused = familyEntry('f', {actions: {update: {available: false, reason: 'Installed by another tool; update it there'}}});
    assert.equal(shown(refused), true);
    assert.match(modelProgramMarkup({harness: 'f'}, {entry: refused}, 'Program 0.50.0'), /Installed by another tool/);
    assert.doesNotMatch(modelProgramMarkup({harness: 'f'}, {entry: refused}, 'Program 0.50.0'), /<button/);
    const running = familyEntry('f', {updateAvailable: 'no', operation: {id: 'op', state: 'running'}});
    assert.equal(shown(running), true, 'an operation under way stays in view');
    assert.equal(shown(familyEntry('f'), {operationError: 'Update not started: refused'}), true,
        'the error of an operation the owner started is the one failure the line keeps');
    // Review r2 (Fable, Carbon R2-MINOR-2): a press error does not outlive the update it was about.
    assert.equal(shown(familyEntry('f', {updateAvailable: 'no'}), {operationError: 'Update not started: refused'}), false,
        'a program up to date by another road, with nothing pending, is silent again');
    assert.equal(shown(familyEntry('f', {updateAvailable: 'no'}), {operationError: 'Update not started: refused', request: {key: 'k'}}), true,
        'an unresolved press still keeps its line');
    // A failed passive read hides the line, except for what the owner left unresolved: its reason and recovery stay.
    const failed = {inventoryError: 'Program inspection unavailable: refused'};
    assert.equal(shown(familyEntry('f'), {...failed, request: {key: 'k'}}), true, 'an unconfirmed press keeps Retry same request');
    assert.equal(shown(familyEntry('f'), {...failed, operation: {id: 'op', state: 'running'}}), true, 'an operation under way');
    assert.equal(shown(familyEntry('f'), {...failed, operation: {id: 'op', state: 'succeeded'}}), false, 'a settled one is not the owner\'s pending');
    // Review r2 (Astra R2-3): a successful read that finds no selected program (being replaced) keeps the owner's
    // unresolved press and its recovery; a family with no program and nothing of the owner's pending stays silent.
    const replaced = familyEntry('f', {selection: {path: null, version: null}});
    assert.equal(shown(replaced, {request: {key: 'k'}}), true, 'Retry same request survives a read that finds no program');
    assert.equal(shown(replaced, {operation: {id: 'op', state: 'running'}}), true, 'an operation under way, program absent');
    assert.equal(shown(replaced), false, 'no program and nothing pending: silent');
    // Review r3 (Astra R3-2): a refused press alone does not put Install on the row of a program that is gone.
    assert.equal(shown(replaced, {operationError: 'Update not started: refused'}), false, 'a press error alone, program absent: silent');
    const retry = modelProgramMarkup({harness: 'f'}, {entry: replaced, request: {key: 'k'}}, 'Not installed');
    assert.match(retry, /<span>Not installed\u00a0· <span class="harness-model-latest">0\.60\.0 available<\/span><\/span>/);
    assert.match(retry, /data-program-action="rejoin">Retry same request</);
    // An absent program the engine cannot install here, or whose install failed, says why and offers Check again; the
    // family row's own Install & connect is the one press that installs it.
    const unable = familyEntry('f', {selection: {path: null, version: null},
        actions: {install: {available: false, reason: 'f has no installer for linux.', commands: []}}});
    assert.equal(shown(unable), true);
    const unableHtml = modelProgramMarkup({harness: 'f'}, {entry: unable}, 'Not installed');
    assert.match(unableHtml, /f has no installer for linux\./);
    assert.match(unableHtml, /data-program-action="inspect">Check again</);
    assert.doesNotMatch(unableHtml, /data-program-action="(latest|install)"|Install it yourself/);
    assert.match(unableHtml, /If this system cannot run it, choose another program or run Ouroboros on a system that supports it\./,
        'a refusal never calls to install what this system may not run');
    const lost = familyEntry('f', {selection: {path: null, version: null}, releaseNotesUrl: 'https://vendor.invalid/f/releases',
        operation: {id: 'op', state: 'failed', termination: 'confirmed', commands: [{binary: 'npm', args: ['install', '-g', 'vendor-f'], env: {X: '1'}, script: null}]}});
    assert.equal(shown(lost), true);
    // On a family row its own press installs; the line says the failure and reads it again, nothing to run by hand.
    const lostHtml = modelProgramMarkup({harness: 'f'}, {entry: lost}, 'Not installed', {install: false});
    assert.match(lostHtml, /Install failed/);
    assert.doesNotMatch(lostHtml, /npm install|X=1|vendor\.invalid\/f\/releases/, 'neither what the engine ran nor a release-notes page');
    assert.match(lostHtml, /data-program-action="inspect">Check again</);
    assert.doesNotMatch(lostHtml, /data-program-action="(latest|install)"/);
    // A model chooser has no family row: its line keeps the Install it always offered an absent producer.
    const chooser = modelProgramMarkup({harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'}, {entry: lost});
    assert.match(chooser, /data-program-action="install">Install<\/button>/);
    assert.match(modelProgramMarkup({harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'},
        {entry: familyEntry('f', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'})}), /data-program-action="install">Install</);
});

test('an absent program offers Install only on a read that stands; a refused or stale one keeps its reason and Check again', () => {
    const absent = familyEntry('f', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const program = {harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'};
    const refusal = 'Program inspection unavailable: The engine is not running.';
    const line = (view, options) => modelProgramMarkup(program, {entry: absent, ...view}, 'Not installed', options);
    assert.match(line({}), /data-program-action="install">Install<\/button>/, 'a standing read offers it');
    assert.doesNotMatch(line({}, {install: false}), /data-program-action="install"/, 'a family row\'s own press installs it instead');
    for (const [why, view] of [['a refused read of the program', {inspectionError: refusal}], ['a refused read of the whole list', {inventoryError: refusal}],
        ['a newer list that left it out', {listed: false}]]) {
        for (const options of [undefined, {install: false}]) {
            const html = line(view, options);
            assert.doesNotMatch(html, /data-program-action="(install|latest)"/, `${why}: no Install on an observation that no longer stands`);
            assert.match(html, /data-program-action="inspect">Check again</, `${why}: the way back is to read it again`);
            assert.match(html, /Not installed/, `${why}: the program as last read`);
        }
        if (view.listed === undefined) assert.match(line(view), /data-tone="warn">Program inspection unavailable: The engine is not running\./, `${why}: its reason`);
    }
    // An unresolved press keeps its recovery through a refused read as before; an operation under way offers no second press.
    assert.match(line({inspectionError: refusal, request: {key: 'k'}}), /data-program-action="rejoin"[^>]*>Retry same request</);
    const running = line({operation: {id: 'op', state: 'running'}});
    assert.match(running, /data-program-action="install" disabled>Install</, 'an operation under way: the press stands but disabled');
    assert.doesNotMatch(running, /data-program-action="inspect"/);
    // W5 (rl-20261127T050408): a cold read can be the whole story — `familyProgramShown` shows a family's refused
    // targeted read with no earlier row (`entry` null, e.g. a login reply then a refused program read), and that row
    // keeps its reason and its way back. A cold row with nothing to act on stays button-free, and a shorter list that
    // merely left the program out earns no Check again.
    const cold = (view) => modelProgramMarkup({harness: 'f'}, view, 'Program version not checked');
    assert.doesNotMatch(cold({}), /Check again/, 'no earlier row and no refusal: no way back to offer');
    assert.doesNotMatch(cold({listed: false}), /Check again/, 'a shorter list that left it out: a cold row says nothing');
    const coldRefusal = cold({inspectionError: refusal});
    assert.match(coldRefusal, /data-tone="warn">Program inspection unavailable: The engine is not running\./, 'the cold refusal keeps its reason');
    assert.match(coldRefusal, /data-program-action="inspect">Check again</, 'and the row the read again');
    assert.doesNotMatch(coldRefusal, /data-program-action="(install|latest|rejoin)"/, 'with no press on the refused read');
    assert.match(cold({inspectionError: refusal, request: {key: 'k'}}), /data-program-action="rejoin"[^>]*>Retry same request</,
        'an unresolved press keeps its own action even with no earlier row');
});

test('a wizard row whose Install press met a refused read keeps its line, sends nothing and offers Check again', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const calls = [], starts = [];
    let refuse = false;
    const absent = (harness) => familyEntry(harness, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const api = {
        async harnessMaintenanceInventory(query) {
            calls.push(query);
            if (refuse && query.harness) throw Object.assign(new Error('The engine stopped answering.'), {status: 503});
            return {harnesses: [absent('wiz-stale-family'), absent('wiz-stale-alone')]};
        },
        async startHarnessMaintenance(body, key) { starts.push([body, key]); return {id: 'op', harness: body.harness, state: 'running', phase: 'installing'}; },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-stale-family'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.match(programs.others(), /data-program="wiz-stale-alone"[\s\S]*data-program-action="install">Install</, 'read absent: its own Install');
    let press;
    const bind = () => programs.bind({querySelectorAll: () => [{dataset: {programAction: 'install'}, closest: () => ({dataset: {program: 'wiz-stale-alone'}}),
        addEventListener: (_type, fn) => { press = fn; }}]});
    bind();
    refuse = true;
    press(); await flush();
    assert.deepEqual(starts, [], 'the read before the Install was refused: nothing was sent');
    const others = programs.others();
    assert.match(others, /data-program="wiz-stale-alone"/, 'the pressed program keeps its line');
    assert.match(others, /Program inspection unavailable: The engine stopped answering\./, 'with its reason');
    assert.match(others, /data-program-action="inspect">Check again</, 'and its recovery');
    assert.doesNotMatch(others, /data-program-action="install"/, 'but no Install on the stale observation');
    // Read again, it stands again: one Install, one request.
    refuse = false;
    bind();
    await programs.read(); await flush();
    assert.match(programs.others(), /data-program-action="install">Install</);
    bind();
    press(); await flush();
    assert.deepEqual(starts.map(([body]) => body), [{harness: 'wiz-stale-alone', target: {kind: 'latest'}}]);
    programs.dispose();
});

test('the wizard with Claudexor stopped says once that Connect installs, and its one press starts the daemon to list programs', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const calls = [];
    let woke = {ok: false, error: 'claudexor_unavailable: The engine could not start.'};
    const store = {snapshot: null, async wake() {
        calls.push('wake');
        if (woke.ok) store.snapshot = {daemon: {state: 'running', runtime: {state: 'ready'}}};
        return woke;
    }};
    const absent = (harness) => familyEntry(harness, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const api = {async harnessMaintenanceInventory(query) { calls.push(query); return {harnesses: [absent('wiz-cold'), absent('wiz-alone')]}; }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-cold'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.equal(programs.others(), '', 'nothing heard of the engine yet: nothing said');
    store.snapshot = {daemon: {state: 'not_provisioned', runtime: {state: 'not_installed'}}};
    programs.observe(); await flush();
    const others = programs.others();
    assert.match(others, /^<div class="agent-other-programs" data-programs-unknown><span>If a program is missing, Connect tries to install it first\.<\/span>/);
    assert.match(others, /data-program-action="find">Show programs to install \(prepares and starts the agent daemon\)<\/button>/);
    assert.equal(programs.missing('wiz-cold'), null, 'unread, no program is said absent: the row keeps Connect');
    assert.deepEqual(calls, [], 'drawing and observing start nothing');

    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'find'}, addEventListener: (_type, fn) => { press = fn; }}]});
    press();
    assert.match(programs.others(), /data-program-action="find" disabled>Starting the agent daemon…</);
    await flush();
    assert.deepEqual(calls, ['wake']);
    assert.match(programs.others(), /Could not start the agent daemon: claudexor_unavailable: The engine could not start\./);
    assert.match(programs.others(), /data-program-action="find">Show programs to install/, 'a refused start leaves the press');

    woke = {ok: true, error: ''};
    press(); await flush();
    assert.deepEqual(calls.slice(1), ['wake', {harness: '', fresh: true, checkLatest: false}], 'one start, then the programs as it reads them');
    assert.deepEqual(programs.missing('wiz-cold'), {installable: true, failed: false, byHand: false}, 'the family row now says Install & connect');
    assert.doesNotMatch(programs.others(), /data-programs-unknown/);
    assert.match(programs.others(), /<span>More programs:<\/span>[\s\S]*data-program="wiz-alone"[\s\S]*data-program-action="install">Install</);
    programs.dispose();
});

test('a sign-in answer reads its program again unless a current read shows it installed; a row that cannot install gives its line the Install', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const calls = [];
    let rows = [];
    const store = {snapshot: {daemon: {state: 'running'}}};
    const api = {async harnessMaintenanceInventory(query) { calls.push(query); return {harnesses: rows.filter((row) => !query.harness || row.harness === query.harness)}; }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-ans'], store, api, onSettled: () => {}});
    const answer = async () => { calls.length = 0; programs.loginHooks().onAnswer('wiz-ans'); await flush(); return calls; };
    assert.deepEqual(await answer(), [{harness: 'wiz-ans', fresh: true, checkLatest: false}], 'never read: the sign-in may have installed it');
    rows = [familyEntry('wiz-ans', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'})];
    await programs.read(); await flush();
    assert.equal((await answer()).length, 1, 'read absent: read again');
    assert.equal(programs.line('wiz-ans'), '', 'the row\'s own press installs it');
    assert.match(programs.line('wiz-ans', {install: true}), /data-program-action="install">Install</, 'a row that cannot leaves it to the line');
    rows = [familyEntry('wiz-ans')];
    await programs.read(); await flush();
    assert.deepEqual(await answer(), [], 'read installed now: nothing to read again');
    programs.dispose();
});

// The engine could not be acquired for the press: the host answers that this attempt was not submitted.
const notSubmitted = () => Object.assign(new Error('Engine temporarily busy'), {status: 503,
    body: {error: {code: 'engine_unavailable', context: {scope: 'this_attempt', acceptance: 'not_submitted'}}}});

test('an Update the engine never received keeps its reason and offers Update again, also after a healthy read', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const store = {snapshot: {daemon: {state: 'running'}}};
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const starts = [];
    const api = {
        harnessMaintenanceInventory: async () => ({harnesses: [familyEntry('wiz-refused')]}),
        async startHarnessMaintenance(body, key) { starts.push(key); if (starts.length === 1) throw notSubmitted(); return {id: 'op', harness: body.harness, state: 'queued'}; },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-refused'], store, api, onSettled: () => {}});
    programs.observe(); await flush();
    const press = async () => {
        let click;
        programs.bind({querySelectorAll: () => [{dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'wiz-refused'}}),
            addEventListener: (_type, fn) => { click = fn; }}]});
        click(); await flush();
    };
    await press();
    for (const when of ['after the refusal', 'after a healthy read']) {
        const line = programs.line('wiz-refused');
        assert.match(line, /Update not started: Engine temporarily busy/, when);
        assert.match(line, /data-program-action="latest">Update</, `${when}: a new press is offered`);
        assert.doesNotMatch(line, /Retry same request/, when);
        await programs.read(); await flush();
    }
    await press();
    assert.equal(starts.length, 2, 'the second press is a new request');
    assert.notEqual(starts[0], starts[1]);
    assert.match(programs.line('wiz-refused'), /Update queued/);
    programs.dispose();
});

test('an Update whose acceptance is unknown keeps Retry same request through a failed read', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const store = {snapshot: {daemon: {state: 'running'}}};
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    let fail = false;
    const api = {
        async harnessMaintenanceInventory() { if (fail) throw Object.assign(new Error('Service unavailable'), {status: 503}); return {harnesses: [familyEntry('wiz-unknown')]}; },
        async startHarnessMaintenance() { throw Object.assign(new Error('Bad gateway'), {status: 503}); },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-unknown'], store, api, onSettled: () => {}});
    programs.observe(); await flush();
    let click;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'wiz-unknown'}}),
        addEventListener: (_type, fn) => { click = fn; }}]});
    click(); await flush();
    assert.match(programs.line('wiz-unknown'), /data-program-action="rejoin"[^>]*>Retry same request</);
    fail = true;
    await programs.read(); await flush();
    const line = programs.line('wiz-unknown');
    assert.match(line, /data-program-action="rejoin"[^>]*>Retry same request</, 'the recovery stays through the failed read');
    assert.match(line, /Update acceptance unconfirmed: Bad gateway/, 'and so does its reason');
    programs.dispose();
});

test('the wizard asks the engine only while it runs, the newest release only for a selected program', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const store = {snapshot: {daemon: {state: 'stale'}}};
    const calls = [], settled = [];
    let updated = false, latest;
    const rows = () => [familyEntry('wiz-one', updated ? {updateAvailable: 'no', selection: {path: '/vendor/wiz-one', version: '0.60.0'}} : {}),
        familyEntry('wiz-none', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'})];
    const api = {
        async harnessMaintenanceInventory(query) {
            calls.push(query);
            const list = rows().filter((row) => !query.harness || row.harness === query.harness);
            if (query.checkLatest) await latest?.promise;
            return {harnesses: query.checkLatest ? list : list.map((row) => ({...row, newest: {status: 'not_read', source: null}, updateAvailable: 'cannot_tell'}))};
        },
        startHarnessMaintenance: async (body) => ({id: 'wiz-op', harness: body.harness, state: 'queued'}),
        harnessMaintenanceOperation: async (id) => ({id, harness: 'wiz-one', state: 'succeeded', outcome: 'changed', mutation: 'changed',
            before: {version: '0.50.0'}, after: {version: '0.60.0'}, finishedAt: '2026-11-20T00:00:00Z'}),
    };
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    let changes = 0;
    const programs = bindFamilyPrograms({harnesses: ['wiz-one', 'wiz-none'], store, api, onChange: () => { changes++; },
        onSettled: (harness) => settled.push(harness)});
    programs.observe({remount: true}); await programs.read(); await flush();
    assert.deepEqual(calls, [], 'an engine that is not running is never asked, not even after a sign-in settles');
    assert.equal(programs.line('wiz-one'), '');

    store.snapshot = {daemon: {state: 'running'}};
    latest = Promise.withResolvers();
    programs.observe(); await flush();
    assert.deepEqual(calls, [{harness: '', fresh: true, checkLatest: false}, {harness: 'wiz-one', fresh: true, checkLatest: true}],
        'it comes up: one inventory, then the newest release of the one family with a program');
    assert.equal(programs.line('wiz-one'), '', 'nothing flashes before the newest read answers');
    latest.resolve(); await flush();
    const line = programs.line('wiz-one');
    assert.match(line, /<span>Program 0\.50\.0\u00a0· <span class="harness-model-latest">0\.60\.0 available<\/span><\/span>/);
    assert.match(line, /data-program-action="latest"[^>]*>Update</);
    assert.equal(programs.line('wiz-none'), '');
    programs.observe(); await flush();
    assert.equal(calls.length, 2, 'a tick of a running engine asks nothing new');

    await programs.read();
    assert.equal(calls.length, 4, 'a settled sign-in asks again');

    let click;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'wiz-one'}}),
        addEventListener: (_type, fn) => { click = fn; }}]});
    click(); await flush();
    assert.match(programs.line('wiz-one'), /Update queued/);
    updated = true;
    programs.observe(); await flush();
    assert.deepEqual(settled, ['wiz-one'], 'the status tick polls the operation until it settles');
    assert.deepEqual(calls.slice(-2), [{harness: '', fresh: true, checkLatest: false}, {harness: 'wiz-one', fresh: true, checkLatest: true}],
        'a settled operation asks again');
    assert.equal(programs.line('wiz-one'), '', 'an up-to-date program leaves the row quiet');

    const [before, asked] = [changes, calls.length];
    programs.dispose();
    await programs.read(); await flush();
    assert.deepEqual([changes, calls.length], [before, asked], 'a disposed line asks and paints nothing more');
});

test('a refused or failed inventory read paints nothing', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const store = {snapshot: {daemon: {state: 'running'}}};
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    for (const [harness, failure] of [['wiz-old-engine', Object.assign(new Error('Update the app or connect an engine at version 6.0.0 or newer to install or update harness programs.'), {status: 503})],
        ['wiz-attached', new TypeError('Failed to fetch')]]) {
        let fail = false;
        const api = {async harnessMaintenanceInventory() {
            if (fail) throw failure;
            return {harnesses: [familyEntry(harness)]};
        }};
        const programs = bindFamilyPrograms({harnesses: [harness], store, api, onSettled: () => {}});
        programs.observe(); await flush();
        assert.match(programs.line(harness), /Update/, 'the control: an answered read shows the line');
        fail = true;
        await programs.read(); await flush();
        assert.equal(programs.line(harness), '', `${harness}: a refused read clears the line`);
        // A running engine that refused its inventory keeps its words where the other programs stand, and no start.
        assert.match(programs.others(), /data-programs-unknown>[\s\S]*Connect tries to install it first\.[\s\S]*data-tone="warn"/);
        assert.doesNotMatch(programs.others(), /data-program-action="find"/);
        programs.dispose();
    }
});

// W2 residual: the program row and the maintenance controller's completion boundary.

test('a pending press outranks an old receipt: the row offers Retry same request, never Check operation', () => {
    const program = {harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'};
    const entry = familyEntry('f', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const html = modelProgramMarkup(program, {entry, request: {key: 'k'},
        operation: {id: 'op', state: 'failed', termination: 'unconfirmed'}}, 'Not installed');
    assert.match(html, /data-program-action="rejoin"[^>]*>Retry same request</, 'the unresolved press is the recovery');
    assert.doesNotMatch(html, /Check operation/, 'an older receipt never labels the row');
    assert.doesNotMatch(html, /data-program-action="(install|latest)"/, 'and it offers no second press');
});

test('facts that no longer stand offer no press and no advice by hand, only the way back', () => {
    const program = {harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'};
    // A newer list left this program out: its retained row is no current basis for an Update.
    const present = familyEntry('f');
    const stale = modelProgramMarkup(program, {entry: present, listed: false}, 'Program 0.50.0');
    assert.doesNotMatch(stale, /data-program-action="(install|latest)"/, 'no new press on a fact that no longer stands');
    assert.match(stale, /data-program-action="inspect">Check again</, 'the way back is the row\'s own read');
    assert.match(stale, /Program 0\.50\.0/, 'the program as last read stays');
    // An absent program the owner may install by hand: a stale list is no basis for that advice either.
    const absent = familyEntry('f', {selection: {path: null, version: null},
        actions: {install: {available: false, reason: 'f has no installer for linux.'}}});
    const advice = modelProgramMarkup(program, {entry: absent, listed: false}, 'Not installed');
    assert.doesNotMatch(advice, /data-program-action="(install|latest)"/);
    assert.doesNotMatch(advice, /If this system cannot run it|To install it by hand/, 'no advice by hand from a retained fact');
    assert.match(advice, /data-program-action="inspect">Check again</);
});

test('a failed operation keeps its own reason on the row', () => {
    const program = {harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'};
    const entry = familyEntry('f', {selection: {path: null, version: null},
        operation: {id: 'op', state: 'failed', termination: 'confirmed', problem: {message: 'npm registry refused the package'}}});
    const html = modelProgramMarkup(program, {entry}, 'Not installed');
    assert.match(html, /Install failed/, 'the outcome stays');
    assert.match(html, /npm registry refused the package/, 'and so does the engine\'s own reason');
});

test('a cold login answer reads its program even when the store has not polled the daemon again', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const calls = [];
    // The store still says the daemon is not running: the answer itself may have started it.
    const store = {snapshot: {daemon: {state: 'not_provisioned', runtime: {state: 'not_installed'}}}};
    const api = {async harnessMaintenanceInventory(query) {
        calls.push(query);
        return {harnesses: [familyEntry('wiz-cold-answer')]};
    }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-cold-answer'], store, api, onSettled: () => {}});
    programs.loginHooks().onAnswer('wiz-cold-answer');
    await flush();
    assert.deepEqual(calls, [{harness: 'wiz-cold-answer', fresh: true, checkLatest: false}],
        'the program\'s own targeted read is not skipped because the store has not polled');
    programs.dispose();
});

test('a pressed program keeps its row after the wizard is remounted', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const absent = (harness) => familyEntry(harness, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    let refuse = false;
    const api = {async harnessMaintenanceInventory(query) {
        if (refuse && query.harness) throw Object.assign(new Error('The engine stopped answering.'), {status: 503});
        return {harnesses: [absent('wiz-fam'), absent('wiz-alone')]};
    }};
    const first = bindFamilyPrograms({harnesses: ['wiz-fam'], store, api, onSettled: () => {}});
    first.observe({remount: true}); await flush();
    let press;
    first.bind({querySelectorAll: () => [{dataset: {programAction: 'install'}, closest: () => ({dataset: {program: 'wiz-alone'}}),
        addEventListener: (_type, fn) => { press = fn; }}]});
    refuse = true;
    press(); await flush();
    assert.match(first.others(), /data-program="wiz-alone"/, 'the pressed program keeps its line');
    // The wizard is remounted: the press belongs with the shared record, so the row and its recovery stay.
    const second = bindFamilyPrograms({harnesses: ['wiz-fam'], store, api, onSettled: () => {}});
    assert.match(second.others(), /data-program="wiz-alone"/, 'the pressed program keeps its row after the remount');
    assert.match(second.others(), /data-program-action="inspect">Check again</, 'with its recovery');
    first.dispose(); second.dispose();
});

// ---- W3 (rl-20261126T203549-c634366eecb0): a refused or omitting read is no fact to show a family line by,
// and the owner's pressed Install keeps its line, reason and recovery through it. ----

test('familyProgramShown shows a targeted refusal and hides whole-list or omitting reads before any absent-program speech', async () => {
    const {familyProgramShown} = await import('../modules/model_program.js');
    const unable = familyEntry('f', {selection: {path: null, version: null},
        actions: {install: {available: false, reason: 'f has no installer for linux.', commands: []}}});
    assert.equal(familyProgramShown({entry: unable}), true, 'an unpressed by-hand row is unchanged');
    // W5 (rl-20261127T050408): the refused targeted read is this family's own fact even with no program
    // selected (a login answer read it and was refused); its read again carries no Install speech.
    assert.equal(familyProgramShown({entry: unable, inspectionError: 'x'}), true,
        'a targeted refusal speaks with its reason and Check again');
    for (const view of [{inventoryError: 'x'}, {listed: false}])
        assert.equal(familyProgramShown({entry: unable, ...view}), false,
            `${JSON.stringify(view)}: no duplicate refusal beside the whole list's notice`);
});

test('a pressed family row keeps its line, reason and Check again when the read before its Install is refused', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const starts = [];
    let refuse = false;
    const absent = (h) => familyEntry(h, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const api = {
        async harnessMaintenanceInventory(query) {
            if (refuse && query.harness) throw Object.assign(new Error('The engine stopped answering.'), {status: 503});
            return {harnesses: [absent('wiz-fam')]};
        },
        async startHarnessMaintenance(body, key) { starts.push([body, key]); return {id: 'op', harness: body.harness, state: 'running', phase: 'installing'}; },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-fam'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'install'}, closest: () => ({dataset: {program: 'wiz-fam'}}),
        addEventListener: (_t, fn) => { press = fn; }}]});
    refuse = true;
    press(); await flush();
    assert.deepEqual(starts, [], 'the refused pre-install read sent nothing');
    const line = programs.line('wiz-fam');
    assert.match(line, /data-program="wiz-fam"/, 'the pressed family row keeps its line');
    assert.match(line, /Program inspection unavailable: The engine stopped answering\./, 'with its reason');
    assert.match(line, /data-program-action="inspect">Check again</, 'and its recovery');
    assert.doesNotMatch(line, /data-program-action="(install|latest)"/, 'no new press on the refused observation');
    programs.dispose();
});

test('others() renders a pressed program beside the whole-inventory refusal', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const starts = [];
    let failWhole = false;
    const absent = (h) => familyEntry(h, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const api = {
        async harnessMaintenanceInventory(query) {
            if (failWhole && !query.harness) throw Object.assign(new Error('Inventory refused'), {status: 503});
            return {harnesses: [absent('wiz-fam'), absent('wiz-alone')]};
        },
        async startHarnessMaintenance(body) { starts.push(body); return {id: 'op', harness: body.harness, state: 'running', phase: 'installing'}; },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-fam'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'install'}, closest: () => ({dataset: {program: 'wiz-alone'}}),
        addEventListener: (_t, fn) => { press = fn; }}]});
    press(); await flush();
    assert.equal(starts.length, 1, 'the pre-install read answered absent: one Install sent');
    failWhole = true;
    await programs.read(); await flush();
    const html = programs.others();
    assert.match(html, /data-programs-unknown[\s\S]*Inventory refused/, 'the whole refusal keeps its one notice');
    assert.match(html, /data-program="wiz-alone"/, 'the pressed program renders beside it');
    programs.dispose();
});

// ---- W5: the recovery row of the setup wizard. A refused inventory offers its way back, a failed status read
// retires what the old running said (and the press says the start it will make), and a stopped engine hides
// nothing of the owner's own. ----

test('a refused inventory on a running engine offers Check again: one fresh inventory, no wake, no press', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    const calls = [];
    let refuse = false;
    const store = {snapshot: {daemon: {state: 'running'}}, error: '', async wake() { calls.push('wake'); return {ok: true}; }};
    const api = {async harnessMaintenanceInventory(query) {
        calls.push(query);
        if (refuse) throw Object.assign(new Error('Inventory refused'), {status: 503});
        return {harnesses: [familyEntry('wiz-inv')]};
    }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-inv'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.deepEqual(calls[0], {harness: '', fresh: true, checkLatest: false}, 'the engine is asked for its inventory once');
    refuse = true;
    await programs.read(); await flush();
    const others = programs.others();
    assert.match(others, /data-programs-unknown[\s\S]*Inventory refused/, 'the refusal keeps its explanation');
    assert.match(others, /data-program-action="check-inventory">Check again</, 'and the way back beside it');
    assert.doesNotMatch(others, /data-program-action="find"/, 'a running engine is never started by this row');
    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'check-inventory'},
        addEventListener: (_t, fn) => { press = fn; }}]});
    const before = calls.length;
    press(); await flush();
    assert.deepEqual(calls.slice(before), [{harness: '', fresh: true, checkLatest: false}],
        'exactly one fresh inventory: no wake, no press, no second read');
    programs.dispose();
});

test('a failed status read retires the old running: no program rows, and the press is the labelled start', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    const calls = [];
    const store = {snapshot: {daemon: {state: 'running'}}, error: 'unreachable',
        // The real store clears its read error only when the engine answers this, and the row follows that answer.
        async wake() { calls.push('wake'); this.error = ''; return {ok: true}; }};
    const api = {async harnessMaintenanceInventory(query) {
        calls.push(query);
        return {harnesses: [familyEntry('wiz-stale')]};
    }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-stale'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.deepEqual(calls, [], 'a status read that failed asks nothing by itself');
    const others = programs.others();
    assert.doesNotMatch(others, /data-program="wiz-stale"/, 'no program row from the old running read');
    assert.doesNotMatch(others, /data-program-action="install"/, 'and no new install from it');
    assert.match(others, /data-program-action="find">Show programs to install \(prepares and starts the agent daemon\)/,
        'the retained running row behind the failed read signs no mere read: the press says the start it will make');
    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'find'},
        addEventListener: (_t, fn) => { press = fn; }}]});
    press(); await flush();
    assert.equal(calls[0], 'wake', 'one start, as the press promised');
    assert.deepEqual(calls.slice(1, 2), [{harness: '', fresh: true, checkLatest: false}], 'and one fresh inventory behind it');
    assert.equal(calls.filter((call) => call?.harness === '').length, 1, 'no second whole read');
    assert.equal(programs.line('wiz-stale') !== '', true, 'the engine that answered again speaks its row');
    programs.dispose();
});

test('a partial answer is an answering engine: its refused programs row gives the fresh Check, never a start', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    const calls = [];
    // The daemon calls itself unreachable while one facet landed: that IS an answer (`daemonAnswered`), and a start
    // would be a second act the owner never asked for.
    const store = {snapshot: {daemon: {state: 'unreachable'}, reads: {accounts: 'ok', catalog: 'not_read', quota: 'not_read'}},
        error: '', async wake() { calls.push('wake'); return {ok: true}; }};
    const api = {async harnessMaintenanceInventory(query) {
        calls.push(query);
        throw Object.assign(new Error('Inventory refused'), {status: 503});
    }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-part'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.deepEqual(calls[0], {harness: '', fresh: true, checkLatest: false}, 'the engine that answered is asked for its programs');
    const others = programs.others();
    assert.match(others, /data-program-action="check-inventory">Check again</, 'the way back is a fresh read');
    assert.doesNotMatch(others, /data-program-action="find"/, 'and no start is offered to an engine that answers');
    assert.equal(calls.includes('wake'), false, 'nothing started itself');
    programs.dispose();
});

test('a stopped engine keeps the owner\'s unresolved press and its receipt, and offers no new install', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    const starts = [];
    const store = {snapshot: {daemon: {state: 'running'}}, error: ''};
    const absent = (harness) => familyEntry(harness, {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const api = {
        async harnessMaintenanceInventory() { return {harnesses: [absent('wiz-fam'), absent('wiz-stray')]}; },
        async startHarnessMaintenance(body) { starts.push(body); throw Object.assign(new Error('Bad gateway'), {status: 503}); },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-fam'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    let press;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'install'}, closest: () => ({dataset: {program: 'wiz-stray'}}),
        addEventListener: (_t, fn) => { press = fn; }}]});
    press(); await flush();
    assert.equal(starts.length, 1, 'the press was sent once');
    assert.match(programs.others(), /data-program="wiz-stray"[\s\S]*Retry same request/, 'the unconfirmed press keeps its receipt');
    // The engine stops.
    store.snapshot = {daemon: {state: 'stale'}};
    const others = programs.others();
    assert.match(others, /data-program="wiz-stray"[\s\S]*Retry same request/, 'a stopped engine hides nothing of the owner\'s own');
    assert.match(others, /acceptance unconfirmed: Bad gateway/, 'and keeps the reason of the uncertain receipt');
    assert.doesNotMatch(others, /data-program-action="install"/, 'no new install is offered');
    assert.match(others, /data-program-action="find">Show programs to install/, 'the deliberate start stays');
    assert.equal(programs.line('wiz-fam'), '', 'a family with nothing of the owner\'s own stays quiet');
    programs.dispose();
});

// ---- W5 final (rl-20261127T050408): the two UI-facing recovery cases of the setup wizard — a refused targeted
// read after a login reply keeps its family line, and a durable receipt keeps its other-program row after a
// reload — plus the controller's unsafe-resend projection and the wizard's one pending-intent predicate. ----

test('a refused targeted read after a login reply keeps the family line with its reason, never silence', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const calls = [];
    let refuse = false;
    const api = {async harnessMaintenanceInventory(query) {
        calls.push(query);
        if (refuse && query.harness) throw Object.assign(new Error('The engine stopped answering.'), {status: 503});
        return {harnesses: [familyEntry('wiz-antigravity', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'})]};
    }};
    const programs = bindFamilyPrograms({harnesses: ['wiz-antigravity'], store, api, onSettled: () => {}});
    programs.observe({remount: true}); await flush();
    assert.equal(programs.line('wiz-antigravity'), '', 'absent with nothing pending: the row\'s own press installs it');
    refuse = true;
    programs.loginHooks().onAnswer('wiz-antigravity'); await flush();
    assert.deepEqual(calls.at(-1), {harness: 'wiz-antigravity', fresh: true, checkLatest: false}, 'the answer read that program itself');
    const line = programs.line('wiz-antigravity');
    assert.match(line, /data-program="wiz-antigravity"/, 'the refused targeted read is no longer silence');
    assert.match(line, /data-tone="warn">Program inspection unavailable: The engine stopped answering\./);
    assert.match(line, /data-program-action="inspect">Check again</);
    assert.doesNotMatch(line, /data-program-action="(install|latest)"/, 'no installation speech from the refused read');
    assert.doesNotMatch(programs.others(), /data-programs-unknown/, 'the whole list answered: no engine-wide banner is invented');
    programs.dispose();
});

test('after a reload a durable receipt keeps its other-program row though pressed is false', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 80; n++) await Promise.resolve(); };
    for (const receipt of [{id: 'wiz-open-op', harness: 'wiz-open', state: 'failed', termination: 'unconfirmed', phase: 'installing'},
        {id: 'wiz-open-run', harness: 'wiz-open', state: 'running', termination: 'not_applicable', phase: 'updating'}]) {
        const store = {snapshot: {daemon: {state: 'running'}}};
        const api = {async harnessMaintenanceInventory() {
            return {harnesses: [familyEntry('wiz-settled', {updateAvailable: 'no'}),
                {...familyEntry('wiz-open', {updateAvailable: 'no'}), operation: receipt}]};
        }};
        const programs = bindFamilyPrograms({harnesses: [], store, api, onSettled: () => {}});
        programs.observe({remount: true}); await flush();
        const others = programs.others();
        assert.match(others, /data-program="wiz-open"/, `${receipt.state}: the durable receipt keeps its row`);
        assert.doesNotMatch(others, /data-program="wiz-settled"/, 'a settled program with nothing of the owner\'s own stays out');
        assert.doesNotMatch(others, /data-program-action="install"/, `${receipt.state}: no new install from a receipt`);
        assert.match(others, receipt.termination === 'unconfirmed' ? /data-program-action="rejoin">Check operation</ : /data-program-action="latest"[^>]*disabled>Update</);
        programs.dispose();
    }
});

test('an unsafe resend keeps the owner\'s press, shows the reason and offers the read-only Check again', () => {
    const program = {harness: 'f', programs: [{path: '/vendor/f', version: '0.50.0'}], label: 'F'};
    const entry = familyEntry('f', {selection: {path: null, version: null}, updateAvailable: 'cannot_tell'});
    const blocked = 'Install outcome unknown; resending could duplicate it.';
    for (const value of [blocked, {reason: blocked}, true]) {
        const html = modelProgramMarkup(program, {entry, request: {key: 'k'}, retryBlocked: value}, 'Not installed');
        assert.match(html, /data-program-action="inspect">Check again</, 'the way on is a read');
        assert.doesNotMatch(html, /Retry same request|Sending…/, 'nothing offers the resend');
        assert.doesNotMatch(html, /data-program-action="(install|latest)"/);
    }
    assert.match(modelProgramMarkup(program, {entry, request: {key: 'k'}, retryBlocked: blocked}, 'Not installed'),
        /data-tone="warn">Install outcome unknown; resending could duplicate it\./);
    const plain = modelProgramMarkup(program, {entry, request: {key: 'k'}}, 'Not installed');
    assert.match(plain, /data-program-action="rejoin">Retry same request</, 'without the controller\'s block the replay stands');
});

test('the wizard exposes one pending-intent predicate for a family, from the standing view', async () => {
    const {bindFamilyPrograms} = await import('../modules/model_program.js');
    const flush = async () => { for (let n = 0; n < 60; n++) await Promise.resolve(); };
    const store = {snapshot: {daemon: {state: 'running'}}};
    const api = {
        async harnessMaintenanceInventory() { return {harnesses: [familyEntry('wiz-pend')]}; },
        async startHarnessMaintenance(body) { return {id: 'wiz-pend-op', harness: body.harness, state: 'queued'}; },
    };
    const programs = bindFamilyPrograms({harnesses: ['wiz-pend'], store, api, onSettled: () => {}});
    assert.equal(programs.pending('wiz-pend'), false, 'nothing of the owner\'s own, nothing pending');
    programs.observe({remount: true}); await flush();
    assert.equal(programs.pending('wiz-pend'), false, 'an answered read with no operation is not pending intent');
    let click;
    programs.bind({querySelectorAll: () => [{dataset: {programAction: 'latest'}, closest: () => ({dataset: {program: 'wiz-pend'}}),
        addEventListener: (_t, fn) => { click = fn; }}]});
    click(); await flush();
    assert.equal(programs.pending('wiz-pend'), true, 'an operation under way is the family\'s true pending intent');
    assert.match(programs.line('wiz-pend'), /data-program="wiz-pend"/, 'the row keeps the family the predicate names');
    programs.dispose();
});
