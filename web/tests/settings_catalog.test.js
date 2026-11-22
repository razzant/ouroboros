import assert from 'node:assert/strict';
import test from 'node:test';
import { createScopedModelCatalog, modelSlotCatalogQuery, catalogReadNote, catalogReadState, mergeModelCatalog, refreshMaintenanceCatalogs, refreshModelCatalog, summarizeReadErrors, withConnectionCatalogs } from '../modules/settings_catalog.js';
import { claudexorStatus } from '../modules/claudexor_status_store.js';

const first = {
    items: [{ value: 'claudexor::opaque=owner-model', source_id: 'opaque', credential_profile_id: 'personal', observed_at: 'earlier' }],
    model_sources: [{ id: 'opaque', label: 'Model service', credentialHarness: 'codex' }],
    observed_at: 'earlier', errors: [],
};

test('an editor coalesces identical reads and publishes only current evidence for each scope', async () => {
    const replies = [];
    const catalog = createScopedModelCatalog({ fetchCatalog: (url, init) => new Promise(resolve => replies.push({ url, resolve, signal: init.signal })) });
    const api = modelSlotCatalogQuery('openrouter@team', 'ignored');
    const subscription = modelSlotCatalogQuery('subscription:opaque/source', 'my account');
    assert.equal(api, 'provider_id=openrouter%40team');
    assert.equal(subscription, 'source_id=opaque%2Fsource&credential_profile_id=my+account');
    const previous = catalog.read('view=sources');
    assert.equal(catalog.read('view=sources'), previous);
    const newer = catalog.read('view=sources', { force: true });
    const apiRead = catalog.read(api);
    await Promise.resolve();
    replies[1].resolve(first);
    await newer;
    replies[0].resolve({ items: [], model_sources: [] });
    await previous;
    replies[2].resolve({ items: [{ value: 'openrouter@team::model' }], model_sources: [] });
    await apiRead;
    assert.deepEqual(catalog.get('view=sources').model_sources, first.model_sources);
    assert.equal(catalog.get(api).items.length, 1);
    const pending = catalog.read(subscription);
    await Promise.resolve();
    catalog.dispose();
    replies[3].resolve(first);
    await pending;
    assert.equal(catalog.get(subscription), undefined);
    assert.equal(catalog.get(api), undefined);
});

test('a client on an older core uses only previously requested evidence for unsupported scopes', async () => {
    const calls = [];
    let previous;
    const catalog = createScopedModelCatalog({ supportsScopes: () => false, legacyCatalog: () => previous,
        fetchCatalog: async url => { calls.push(url); return first; } });
    assert.equal((await catalog.read('view=sources')).errors[0].code, 'scoped_catalog_unsupported');
    previous = { ...first, items: [...first.items, { value: 'vendor/default' }, { value: 'openrouter@work::owner/model' }] };
    assert.deepEqual((await catalog.read('view=sources', { force: true })).model_sources, first.model_sources);
    assert.deepEqual((await catalog.read('provider_id=openrouter%40work')).items, [{ value: 'openrouter@work::owner/model' }]);
    assert.deepEqual(calls, []);
    await catalog.read('source_id=opaque&credential_profile_id=personal');
    assert.deepEqual(calls, ['/api/model-catalog?source_id=opaque&credential_profile_id=personal']);
    catalog.dispose();
});

test('failed settlement reads stay retryable; successful receipts refresh once', async (t) => {
    const previousDocument = globalThis.document, previousFetch = globalThis.fetch;
    t.after(() => { globalThis.document = previousDocument; globalThis.fetch = previousFetch; });
    const doc = new EventTarget(); doc.getElementById = () => null; globalThis.document = doc;
    const events = [], native = []; let raw = 0;
    doc.addEventListener('harness-maintenance:settled', event => events.push(event.detail));
    t.mock.method(claudexorStatus, 'refresh', async options => { native.push(options); return {harnesses: [{id: 'future', model_catalog: {partial: false}}]}; });
    globalThis.fetch = async () => { raw++; throw new Error('general catalog unavailable'); };
    const receipt = {id: 'one', finishedAt: '2026-11-10'};
    await refreshMaintenanceCatalogs('future', receipt);
    await refreshMaintenanceCatalogs('future', {...receipt});
    assert.deepEqual(events, [{harness: 'future'}, {harness: 'future'}]);
    assert.ok(native.every(options => options.includeModels && options.afterCurrent));
    assert.equal(raw, 2);
    globalThis.fetch = async () => { raw++; return {ok: true, json: async () => ({items: []})}; };
    await refreshMaintenanceCatalogs('future', receipt);
    await refreshMaintenanceCatalogs('future', receipt);
    assert.equal(events.length, 3); assert.equal(raw, 3);
    await refreshMaintenanceCatalogs('future', {...receipt, id: 'two'});
    assert.equal(events.length, 4); assert.equal(raw, 4);
});

test('connection discovery replaces only the exact source and preserves general refresh provenance', () => {
    const general = mergeModelCatalog({}, { ...first, coverage: { complete: true },
        account_catalogs: [{ source: 'opaque', accounts: [] }], items: [...first.items,
            { value: 'openai::same-model' }, { value: 'openai-compatible::same-model' },
            { value: 'openai-compatible@alpha::old' }, { value: 'openai-compatible@beta::same-model' }] });
    const connections = { 'openai-compatible@alpha': [{ value: 'openai-compatible@alpha::same-model' }] };
    const composed = withConnectionCatalogs(general, connections);
    assert.deepEqual(composed.items.map((item) => item.value), [first.items[0].value,
        'openai::same-model', 'openai-compatible::same-model', 'openai-compatible@beta::same-model',
        'openai-compatible@alpha::same-model']);
    const { items: _items, ...provenance } = composed;
    const { items: _generalItems, ...originalProvenance } = general;
    assert.deepEqual(provenance, originalProvenance);
    // A general catalog nobody has read stays unread under a connection's own list: its rows are not a whole read,
    // so a role line says the catalog's note and never "Price unknown".
    assert.equal(withConnectionCatalogs({}, connections).read_state, 'not_read');
    assert.equal(catalogReadState({ items: withConnectionCatalogs({}, connections).items }), 'ok', 'the rows alone would read as a whole catalog');
    for (const incoming of [
        { read_state: 'transport', errors: [{ error: 'offline' }] },
        { items: [], model_sources: [], errors: [{ error: 'source failed' }] },
        { items: [{ value: 'openai::new' }], model_sources: [], partial: true },
        { items: [{ value: 'openai::new' }], model_sources: [], errors: [] },
        { items: [], model_sources: [], errors: [] },
    ]) {
        const refreshed = mergeModelCatalog(general, incoming);
        const after = withConnectionCatalogs(refreshed, connections);
        assert.deepEqual(after, { ...refreshed, items: [...refreshed.items.filter((item) =>
            item.value !== 'openai-compatible@alpha::old'), ...connections['openai-compatible@alpha']] });
        assert.deepEqual(withConnectionCatalogs(refreshed, { 'openai-compatible@alpha': [] }).items,
            refreshed.items.filter((item) => item.value !== 'openai-compatible@alpha::old'));
    }
    assert.ok(general.items.some((item) => item.value === 'openai-compatible@alpha::old'), 'composition does not mutate the general store');
    assert.deepEqual(withConnectionCatalogs({ items: [
        { value: 'openai-compatible@default::old' }, { value: 'openai-compatible@beta::old' },
    ] }, { 'openai-compatible': [] }).items, [{ value: 'openai-compatible@beta::old' }],
    'source identity uses the existing parser, including the built-in default alias');
});

test('unread, failed, and stale discovery preserve last-known entries with their provenance', () => {
    const before = mergeModelCatalog({}, first);
    for (const gap of [
        { read_state: 'not_read', items: [], model_sources: [] },
        { errors: [{ provider_id: 'claudexor', error: 'connection refused' }], items: [], model_sources: [] },
        { freshness: 'stale', items: [], model_sources: [], coverage: { complete: false } },
        { read_state: 'transport', errors: [{ error: 'network offline' }] },
    ]) {
        const after = mergeModelCatalog(before, gap);
        assert.notEqual(catalogReadState(after), 'ok');
        assert.deepEqual(after.items, first.items);
        assert.deepEqual(after.model_sources, first.model_sources);
        assert.equal(after.observed_at, 'earlier');
        assert.match(catalogReadNote(after), /selection.*kept/);
        assert.doesNotMatch(catalogReadNote(after), /connect.*account|\blog[ -]?in\b/i);
        if (gap.coverage) assert.deepEqual(after.coverage, gap.coverage);
    }
});

test('partial discovery enriches healthy choices and a later successful empty read clears only discovery', () => {
    const partial = mergeModelCatalog(first, {
        items: [{ value: 'openai::new-model', label: 'New model' }], model_sources: [],
        errors: [{ provider_id: 'claudexor', error: 'catalog timeout' }],
    });
    assert.deepEqual(partial.items.map((item) => item.value), ['claudexor::opaque=owner-model', 'openai::new-model']);
    assert.deepEqual(partial.model_sources, first.model_sources);
    assert.match(catalogReadNote(partial), /catalog timeout/);
    const empty = mergeModelCatalog(partial, { items: [], model_sources: [], errors: [] });
    assert.equal(empty.read_state, 'ok');
    assert.equal(empty.sources_read_state, 'ok');
    assert.deepEqual(empty.items, []);
    assert.deepEqual(empty.model_sources, []);
    assert.equal(catalogReadNote(empty), '');
    assert.equal(empty.observed_at, undefined, 'an earlier observation is not the new read timestamp');
});

test('one unreachable source is partial, not a failed catalog: the API models that loaded are named', () => {
    const partial = mergeModelCatalog({}, {
        items: [{ value: 'openai/gpt-5.6-terra' }, { value: 'openai::gpt-5.6-terra' },
            { value: 'claudexor::opaque=owner-model' }],
        model_sources: [], errors: [{ provider_id: 'claudexor', error: 'daemon unreachable' }],
    });
    assert.equal(catalogReadState(partial), 'partial');
    const note = catalogReadNote(partial);
    assert.match(note, /^Some model sources could not be read: daemon unreachable\./);
    assert.match(note, /2 API models loaded/, 'the subscription entry is not an API model');
    assert.doesNotMatch(note, /^Model catalog could not be read/);
    assert.match(note, /selection are kept/);
    assert.match(note, /press Read again in Models/, 'the sentence names the button that exists');
    // One API model reads as one.
    assert.match(catalogReadNote(mergeModelCatalog({}, { items: [{ value: 'openai::only' }],
        errors: [{ error: 'daemon unreachable' }] })), /1 API model loaded/);
    // Nothing loaded is still an outright failure, with the same words as before.
    const failed = mergeModelCatalog({}, { items: [], errors: [{ error: 'daemon unreachable' }] });
    assert.equal(catalogReadState(failed), 'failed');
    assert.match(catalogReadNote(failed), /^Model catalog could not be read: daemon unreachable\./);
    // A partial read still retains last-known entries (read_state is not 'ok').
    const retained = mergeModelCatalog(first, { items: [{ value: 'openai::new' }], model_sources: [],
        errors: [{ error: 'daemon unreachable' }] });
    assert.deepEqual(retained.items.map((item) => item.value),
        ['claudexor::opaque=owner-model', 'openai::new']);
    assert.deepEqual(retained.model_sources, first.model_sources);
});

test('reading models does not prove an omitted model-source list was read', () => {
    const legacy = mergeModelCatalog(first, { items: [], errors: [] });
    assert.equal(legacy.read_state, 'ok');
    assert.equal(legacy.sources_read_state, 'not_read');
    assert.deepEqual(legacy.model_sources, first.model_sources);
});

test('the real Refresh event keeps read failures distinct and carries complete catalog evidence', async (t) => {
    const previousDocument = globalThis.document;
    const previousFetch = globalThis.fetch;
    t.after(() => { globalThis.document = previousDocument; globalThis.fetch = previousFetch; });
    const status = { textContent: '', dataset: {} };
    const document = new EventTarget();
    document.getElementById = (id) => id === 'settings-model-catalog-status' ? status : null;
    globalThis.document = document;
    const events = [];
    document.addEventListener('settings-model-catalog:updated', (event) => events.push(event.detail));
    let response = first;
    globalThis.fetch = async () => {
        if (response instanceof Error) throw response;
        return { ok: true, json: async () => response };
    };
    await refreshModelCatalog();
    assert.deepEqual(events.at(-1).model_sources, first.model_sources);
    assert.equal(events.at(-1).observed_at, 'earlier');
    assert.equal(events.at(-1).read_state, 'ok');
    response = new Error('offline');
    await refreshModelCatalog();
    assert.equal(events.at(-1).read_state, 'transport');
    assert.equal(events.at(-1).items, undefined, 'a failed request must not broadcast an empty catalog');
    assert.equal(events.at(-1).model_sources, undefined);
    assert.match(status.textContent, /offline/);
    response = { items: [], model_sources: [], errors: [], coverage: { complete: true } };
    await refreshModelCatalog();
    assert.equal(events.at(-1).read_state, 'ok');
    assert.deepEqual(events.at(-1).items, []);
    assert.deepEqual(events.at(-1).coverage, { complete: true });
    assert.doesNotMatch(status.textContent, /offline/);
    response = {};
    await refreshModelCatalog();
    assert.equal(events.at(-1).read_state, 'transport', 'a malformed 2xx does not certify an empty read');
    assert.match(status.textContent, /no model list/);
});

test('an older Refresh completion cannot replace newer success or its read facts', async (t) => {
    const previousDocument = globalThis.document;
    const previousFetch = globalThis.fetch;
    t.after(() => { globalThis.document = previousDocument; globalThis.fetch = previousFetch; });
    const document = new EventTarget();
    document.getElementById = () => null;
    globalThis.document = document;
    const events = [];
    document.addEventListener('settings-model-catalog:updated', (event) => events.push(event.detail));
    let finishFirst;
    globalThis.fetch = () => new Promise((resolve) => { finishFirst = resolve; });
    const old = refreshModelCatalog();
    globalThis.fetch = async () => ({ ok: true, json: async () => first });
    await refreshModelCatalog();
    finishFirst({ ok: true, json: async () => ({ items: [], model_sources: [] }) });
    assert.equal((await old).stale, true);
    assert.equal(events.length, 1);
    assert.deepEqual(events[0].items, first.items);
});

test('background refresh cannot strand a superseded manual button busy', async (t) => {
    const previousDocument = globalThis.document;
    const previousFetch = globalThis.fetch;
    t.after(() => { globalThis.document = previousDocument; globalThis.fetch = previousFetch; });
    const document = new EventTarget();
    document.getElementById = () => null;
    globalThis.document = document;
    const button = { disabled: false, setAttribute() {}, removeAttribute() {} };
    const pending = [];
    globalThis.fetch = () => new Promise((resolve) => pending.push(resolve));
    const manual = refreshModelCatalog({ button });
    const background = refreshModelCatalog();
    pending[1]({ ok: true, json: async () => first });
    await background;
    assert.equal(button.disabled, true, 'manual request still owns its busy state');
    pending[0]({ ok: true, json: async () => first });
    assert.equal((await manual).stale, true);
    assert.equal(button.disabled, false);
    const old = refreshModelCatalog({ button });
    const latest = refreshModelCatalog({ button });
    pending[2]({ ok: true, json: async () => first });
    await old;
    assert.equal(button.disabled, true, 'older request cannot release a newer button owner');
    pending[3]({ ok: true, json: async () => first });
    await latest;
    assert.equal(button.disabled, false);
});

test('read errors drop the httpx documentation pointer and rows get the compact form', () => {
    const data = {
        items: [{ value: 'openai/gpt-x' }],
        errors: [{ provider_id: 'openai', error: "Client error '401 Unauthorized' for url 'https://api.openai.com/v1/models'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401" }],
    };
    const note = catalogReadNote(data);
    assert.match(note, /Some model sources could not be read: Client error '401 Unauthorized' for url 'https:\/\/api\.openai\.com\/v1\/models'\. 1 API model loaded\./);
    assert.doesNotMatch(note, /For more information/);
    assert.equal(catalogReadNote(data, { compact: true }),
        'Some model sources could not be read. Existing suggestions and your selection are kept.');
    assert.equal(catalogReadNote({ errors: [{ error: 'boom' }] }, { compact: true }),
        'Model catalog could not be read. Existing suggestions and your selection are kept.');
    assert.equal(catalogReadNote({ items: [] }, { compact: true }), '');
});

const QUOTA_LIMIT = 'The account has an active quota limit';
const quotaProfiles = ['polina', 'chatgptpro1_anton', 'proton3', 'proton2', 'gptpro1',
    'gptpro2', 'gptpro3', 'gptpro4', 'gptpro6'];
const liveAccountErrors = [
    ...quotaProfiles.map((credential_profile_id) => ({ provider_id: 'claudexor', source_id: 'codex',
        credential_profile_id, code: 'quota_exhausted', error: QUOTA_LIMIT })),
    { provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'proton4',
        code: 'sign_in_required', error: 'The selected managed Codex account requires sign-in.' },
    { provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'gptopro6',
        code: 'no_credential', error: 'The account has no verified current catalog credential' },
];

test('accounts sharing one cause are one clause with the profiles named, not one line each', () => {
    const note = catalogReadNote({
        items: [{ value: 'openai::gpt-5.6-terra' }, { value: 'anthropic::claude-fable-5' },
            { value: 'claudexor::codex=owner-model' }],
        errors: liveAccountErrors,
    });
    assert.equal(note, 'Some model sources could not be read: polina, chatgptpro1_anton, proton3 and 6 more:'
        + ' The account has an active quota limit; proton4: The selected managed Codex account requires'
        + ' sign-in; gptopro6: The account has no verified current catalog credential.'
        + ' 2 API models loaded. Existing suggestions and your selection are kept.'
        + ' To retry, press Read again in Models.');
    // Every cause is still named: only the repeated subjects are counted.
    for (const cause of [QUOTA_LIMIT, 'requires sign-in', 'no verified current catalog credential']) {
        assert.ok(note.includes(cause), cause);
    }
    assert.equal(note.split('; ').length, 3, 'one clause per cause, not one per error');
});

test('an account cause and an API-provider cause each get their own clause', () => {
    const note = catalogReadNote({
        items: [{ value: 'openai::gpt-x' }],
        errors: [
            { provider_id: 'claudexor', source_id: 'codex', credential_profile_id: 'polina',
                code: 'quota_exhausted', error: QUOTA_LIMIT },
            { provider_id: 'openai', error: "Client error '401 Unauthorized' for url 'https://api.openai.com/v1/models'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401" },
        ],
    });
    assert.match(note, /^Some model sources could not be read: polina: The account has an active quota limit; Client error '401 Unauthorized' for url 'https:\/\/api\.openai\.com\/v1\/models'\. 1 API model loaded\./);
    assert.doesNotMatch(note, /For more information/);
    assert.doesNotMatch(note, /openai: Client error/, 'a source without a profile is not named');
});

test('repeated causes collapse and shared causes name every distinct profile', () => {
    assert.equal(summarizeReadErrors([]), '');
    assert.equal(summarizeReadErrors([
        { credential_profile_id: 'polina', error: QUOTA_LIMIT },
        { credential_profile_id: 'polina', error: QUOTA_LIMIT },
    ]), `polina: ${QUOTA_LIMIT}`);
    assert.equal(summarizeReadErrors([
        { credential_profile_id: 'polina', error: QUOTA_LIMIT },
        { credential_profile_id: 'proton3', error: QUOTA_LIMIT },
    ]), `polina, proton3: ${QUOTA_LIMIT}`);
    // A source without a profile is folded into the named clause for the same cause; it adds no name.
    assert.equal(summarizeReadErrors([
        { provider_id: 'claudexor', error: QUOTA_LIMIT },
        { credential_profile_id: 'polina', error: QUOTA_LIMIT },
    ]), `polina: ${QUOTA_LIMIT}`);
    assert.equal(summarizeReadErrors([{ provider_id: 'claudexor', code: 'daemon_unreachable' }]), 'daemon_unreachable');
    // A profile with no stated cause is still named, never rendered as "x: ".
    assert.equal(summarizeReadErrors([{ credential_profile_id: 'x' }, {}]), 'x');
});

test('the vendor does not own the banner length: one cause is clamped', () => {
    const long = 'The vendor returned a very long diagnostic. '.padEnd(220, 'x');
    assert.equal(long.length, 220);
    const clause = summarizeReadErrors([{ error: long }]);
    assert.equal(clause, `${long.slice(0, 159)}…`);
    assert.equal(clause.length, 160);
    assert.doesNotMatch(clause, /…./, 'exactly one ellipsis, at the end');
    assert.match(catalogReadNote({ items: [], errors: [{ error: long }] }),
        /^Model catalog could not be read: The vendor returned .*…\. Existing suggestions/);
});

test('a vendor message that ends in a period does not produce a double stop', () => {
    const note = catalogReadNote({
        items: [{ value: 'openai::gpt-x' }],
        errors: [{ credential_profile_id: 'proton4', error: 'The selected managed Codex account requires sign-in.' }],
    });
    assert.match(note, /proton4: The selected managed Codex account requires sign-in\. 1 API model loaded\./);
    assert.doesNotMatch(note, /sign-in(\.\.|\.;|;)/);
});

test('a merge marks what this read itself said: what a gap keeps for the list is not that read\'s word', () => {
    const merged = mergeModelCatalog({}, first);
    assert.equal(merged.current, first, 'a whole read is its own word');
    const partial = { items: [{ value: 'openai::new' }], model_sources: [], errors: [{ error: 'daemon unreachable' }] };
    const kept = mergeModelCatalog(merged, partial);
    assert.deepEqual([kept.items.length, kept.current], [first.items.length + 1, partial], 'the earlier rows stay listed; only the new answer is this read\'s');
    const offline = mergeModelCatalog(kept, { read_state: 'transport', errors: [{ error: 'network offline' }] });
    assert.deepEqual([offline.items.length, offline.current], [kept.items.length, null], 'a read that listed nothing said nothing');
    assert.equal(mergeModelCatalog({}, kept).current, partial, 'a catalog merged before keeps the mark it has');
});

test('a scoped legacy read and a connection overlay use the entry\'s own identity, never a reparsed value', async () => {
    const legacy = { items: [
        { name: 'legacy-openrouter-nested', value: 'openrouter::openai::vendor/model' },
        { name: 'legacy-unknown-head', value: 'vendor::family/model' },
        { name: 'modern-openrouter-nested', provider_id: 'openrouter', id: 'openai::vendor/model', value: 'openai::vendor/model' },
        { name: 'local', provider_id: 'local', id: 'qwen.gguf', value: 'local::qwen.gguf', use_local: true },
        { name: 'modern-openai', provider_id: 'openai', id: 'vendor/model', value: 'openai::vendor/model' },
    ] };
    const names = (answer) => answer.items.map((item) => item.name);
    const catalog = createScopedModelCatalog({ supportsScopes: () => false, legacyCatalog: () => legacy });
    assert.deepEqual(names(await catalog.read('provider_id=openrouter')),
        ['legacy-openrouter-nested', 'legacy-unknown-head', 'modern-openrouter-nested'],
        'OpenRouter\'s own ids stay in its scope, including one whose head names no provider');
    assert.deepEqual(names(await catalog.read('provider_id=openai')), ['modern-openai'],
        'the OpenAI key keeps its own model, never the OpenRouter row that spells the same id');
    assert.deepEqual(names(await catalog.read('provider_id=local')), ['local'],
        'the local row is the local lane\'s, never a provider\'s');
    // The connection overlay replaces exactly the rows of its own source.
    const composed = withConnectionCatalogs({ read_state: 'ok', items: legacy.items }, { 'openai': [{ name: 'openai-connection' }] });
    assert.deepEqual(composed.items.map((item) => item.name),
        ['legacy-openrouter-nested', 'legacy-unknown-head', 'modern-openrouter-nested', 'local', 'openai-connection'],
        'only the OpenAI connection\'s own rows are replaced');
});
