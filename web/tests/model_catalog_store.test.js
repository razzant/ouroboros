// The composer's one client store over /api/model-catalog (model_catalog_store.js): a read when asked and then only
// when stale or failed, a failed read keeping what an earlier one listed, the owner's recent and pinned models from the
// preferences (none while the server does not carry them), a pin as one addressed write, the processing tiers Main and
// a lead take, and the settings document's connections read once.
import assert from 'node:assert/strict';
import test from 'node:test';
import { describeRoute } from '../modules/chat_model_routes.js';
import { apiConnectionOf, createModelCatalogStore, dropTransport, revisionIn } from '../modules/model_catalog_store.js';

const item = (id) => ({ provider_id: 'openrouter', provider: 'OpenRouter', id, name: id, value: id });
const route = (model, local = false) => ({ model, use_local: local });
const flush = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

test('one read serves every reader until it is stale; a failed read keeps the earlier list and says so', async () => {
    let clock = 1000;
    const answers = [{ items: [item('a/one')] }, new Error('offline'), { items: [item('a/two')] }];
    let reads = 0;
    const store = createModelCatalogStore({ now: () => clock, readCatalog: async () => {
        const answer = answers[reads++];
        if (answer instanceof Error) throw answer;
        return answer;
    } });
    const seen = [];
    const stop = store.subscribe(() => seen.push({ ...store.status() }));
    assert.deepEqual(store.status(), { reading: false, read: false, failed: false });
    assert.equal(store.index().rows.length, 0);
    const first = store.read();
    assert.equal(store.read(), first, 'a read in flight is the one read');
    await first;
    assert.deepEqual([reads, store.index().rows.map((row) => row.id)], [1, ['a/one']]);
    assert.deepEqual(seen, [{ reading: true, read: false, failed: false }, { reading: false, read: true, failed: false }]);
    await store.read();
    assert.equal(reads, 1, 'fresh: no second request');
    clock += 6 * 60 * 1000;
    await store.read();
    assert.deepEqual([reads, store.status().failed, store.index().rows.map((row) => row.id)], [2, true, ['a/one']],
        'stale: read again; the failure keeps what was listed');
    await store.read();
    assert.deepEqual([reads, store.status().failed, store.index().rows.map((row) => row.id)], [3, false, ['a/two']], 'a failed read is asked again');
    stop();
    const heard = seen.length;
    await store.read({ force: true });
    assert.equal(seen.length, heard, 'an unsubscribed reader hears nothing');
});

test('recent models come from the preferences, newest first; a body without them is none, never an error', async () => {
    const bodies = [{ model_recents: [{ model: 'a/one', use_local: false }, { model: ' ' }, null, { model: 'q.gguf', use_local: true }] },
        {}, null];
    let at = 0;
    const store = createModelCatalogStore({ readPreferences: async () => { const body = bodies[at++]; if (body === null) throw new Error('down'); return body; } });
    assert.deepEqual(store.recents(), []);
    await store.readRecents();
    assert.deepEqual(store.recents(), [{ model: 'a/one', use_local: false }, { model: 'q.gguf', use_local: true }]);
    await store.readRecents();
    assert.deepEqual(store.recents(), [], 'the server does not carry the field: no recents');
    await store.readRecents();
    assert.deepEqual(store.recents(), [], 'a failed read is no recents either');
});

test('Settings\' own refresh feeds the store; the settings document\'s connections are read once', async () => {
    let settingsReads = 0;
    const store = createModelCatalogStore({ readCatalog: async () => { throw new Error('never asked'); },
        readSettings: async () => { settingsReads += 1; return { _meta: { connections: [{ id: 'openai@old', kind: 'openai', name: 'old',
            display: 'OpenAI · Old', enabled: false, configured: true }] } }; } });
    let heard = 0;
    store.subscribe(() => { heard += 1; });
    store.adopt({ nothing: true });
    assert.equal(heard, 0, 'only a catalog answer is adopted');
    store.adopt({ items: [item('a/one')] });
    assert.deepEqual([heard, store.status().read, store.index().rows.length], [1, true, 1]);
    await store.read();
    assert.equal(store.index().connections, null);
    await Promise.all([store.readConnections(), store.readConnections()]);
    await store.readConnections();
    assert.equal(settingsReads, 1);
    assert.deepEqual(store.index().connections.map((entry) => [entry.id, entry.label, entry.enabled]), [['openai@old', 'OpenAI · Old', false]]);
});

test('the owner\'s pins come with the recents, in pin order; a body without them is none', async () => {
    const bodies = [{ model_recents: [{ model: 'a/one' }], model_pins: [{ model: ' a/two ' }, { model: '' }, null, { model: 'q.gguf', use_local: true }] },
        { model_recents: [] }];
    let at = 0;
    const store = createModelCatalogStore({ readPreferences: async () => bodies[at++] });
    assert.deepEqual(store.pins(), []);
    await store.readRecents();
    assert.deepEqual([store.recents(), store.pins()], [[route('a/one')], [route('a/two'), route('q.gguf', true)]]);
    assert.deepEqual([store.isPinned(route('a/two')), store.isPinned(route('q.gguf')), store.isPinned(route('q.gguf', true))], [true, false, true],
        'the local lane is another route');
    await store.readRecents();
    assert.deepEqual(store.pins(), [], 'the server does not carry the field: no pins');
});

test('a pin is one addressed write, shown at once; its answer is the list; a refusal restores the list as the server has it', async () => {
    const saved = [];
    const server = { pins: [route('x/other')] };  // pinned in another window
    let answer = null;
    const store = createModelCatalogStore({ readPreferences: async () => ({ model_recents: [], model_pins: server.pins }),
        savePreferences: (body) => { saved.push(body); return new Promise((resolve, reject) => { answer = { resolve, reject }; }); } });
    await store.readRecents();
    let heard = 0;
    store.subscribe(() => { heard += 1; });
    const pinning = store.setPinned({ model: ' a/one ', use_local: false }, true);
    assert.deepEqual(saved, [{ model_pins: { pin: [route('a/one')] } }], 'only this pin: a pin made elsewhere meanwhile stays');
    assert.deepEqual([store.pins(), heard], [[route('x/other'), route('a/one')], 1], 'shown pinned before the answer');
    await store.readRecents();
    assert.equal(store.isPinned(route('a/one')), true, 'a read answered meanwhile does not take the press back');
    server.pins = [route('x/other'), route('a/one'), route('y/third')];
    answer.resolve({ model_pins: server.pins });
    assert.equal(await pinning, true);
    assert.deepEqual(store.pins(), server.pins, 'the answer\'s own list');
    const unpinning = store.setPinned(route('x/other'), false);
    assert.deepEqual(saved.at(-1), { model_pins: { unpin: [route('x/other')] } });
    assert.equal(store.isPinned(route('x/other')), false, 'unpinned at once');
    answer.reject(new Error('preferences are read-only'));
    await assert.rejects(unpinning, /read-only/, 'the caller says the refusal');
    await flush();
    assert.deepEqual(store.pins(), server.pins, 'refused: the list is read again, as the server has it');
    assert.equal(await store.setPinned({ model: ' ' }, true), false, 'no route, no write');
    assert.equal(saved.length, 2);
});

test('a refused press stops showing even when the read of the server\'s list fails too; a saved one stays', async () => {
    let answer = null;
    let offline = false;
    const store = createModelCatalogStore({
        readPreferences: async () => { if (offline) throw new Error('offline'); return { model_recents: [], model_pins: [route('x/other')] }; },
        savePreferences: () => new Promise((resolve, reject) => { answer = { resolve, reject }; }) });
    await store.readRecents();
    offline = true;
    const pinning = store.setPinned(route('a/one'), true);
    assert.equal(store.isPinned(route('a/one')), true, 'shown at once');
    answer.reject(new Error('preferences are read-only'));
    await assert.rejects(pinning, /read-only/);
    await flush();
    assert.deepEqual(store.pins(), [route('x/other')], 'the server\'s list as last read, without the refused pin');
    const unpinning = store.setPinned(route('x/other'), false);
    assert.equal(store.isPinned(route('x/other')), false);
    answer.reject(new Error('preferences are read-only'));
    await assert.rejects(unpinning, /read-only/);
    await flush();
    assert.deepEqual(store.pins(), [route('x/other')], 'a refused Unpin puts the pin back');
    // Saved with no list in its answer: the server has the press, so it stays while the read after it fails.
    const saved = store.setPinned(route('a/one'), true);
    answer.resolve({ ok: true });
    assert.equal(await saved, true);
    await flush();
    assert.deepEqual(store.pins(), [route('x/other'), route('a/one')]);
});

test('a Pin and an Unpin pressed before the first answered go out in order; the older answer never brings the pin back', async () => {
    const saved = [];
    const answers = [];
    const server = { pins: [] };
    const store = createModelCatalogStore({ readPreferences: async () => ({ model_recents: [], model_pins: server.pins }),
        savePreferences: (body) => { saved.push(body); return new Promise((resolve, reject) => { answers.push({ resolve, reject }); }); } });
    const one = route('a/one');
    const pinning = store.setPinned(one, true);
    const unpinning = store.setPinned(one, false);
    assert.equal(store.isPinned(one), false, 'the latest press shows at once');
    assert.deepEqual(saved, [{ model_pins: { pin: [one] } }], 'the Unpin waits for the Pin\'s answer');
    server.pins = [one];
    answers[0].resolve({ model_pins: [one] });
    assert.equal(await pinning, true);
    assert.equal(store.isPinned(one), false, 'the Pin\'s answer does not undo the Unpin still on its way');
    await flush();
    assert.deepEqual(saved, [{ model_pins: { pin: [one] } }, { model_pins: { unpin: [one] } }]);
    server.pins = [];
    answers[1].resolve({ model_pins: [] });
    assert.equal(await unpinning, true);
    assert.deepEqual(store.pins(), [], 'the server\'s list after the last press');
});

test('a read sent before a write answered never stands over that answer: one more read settles the list', async () => {
    const server = { pins: [route('a/one')] };
    const reads = [];
    const store = createModelCatalogStore({
        readPreferences: () => new Promise((resolve) => { const seen = [...server.pins]; reads.push(() => resolve({ model_pins: seen })); }),
        savePreferences: async (body) => { server.pins = server.pins.filter((pinned) => pinned.model !== body.model_pins.unpin?.[0]?.model);
            return { model_pins: server.pins }; } });
    const first = store.readRecents();  // a notice's read, out while the owner unpins
    await null;
    assert.equal(reads.length, 1);
    assert.equal(await store.setPinned(route('a/one'), false), true);
    assert.deepEqual(store.pins(), []);
    reads[0]();  // the read answers with the list as it was before the write
    await first;
    assert.deepEqual(store.pins(), [], 'the older read does not bring the pin back');
    await flush();
    assert.equal(reads.length, 2, 'it is read once more');
    reads[1]();
    await flush();
    assert.deepEqual(store.pins(), []);
    // A notice while a read is out is not lost: the read after it runs once the one out answers.
    const again = store.readRecents();
    await null;
    store.readRecents();
    store.readRecents();
    assert.equal(reads.length, 3);
    reads[2]();
    await again;
    await flush();
    assert.equal(reads.length, 4, 'one more read for the notices that came while it was out');
    reads[3]();
    await flush();
});

test('another window\'s pin reaches this one: a preferences change that names the pins, or names nothing, is read again', async () => {
    let reads = 0;
    const store = createModelCatalogStore({ readPreferences: async () => { reads += 1; return {}; } });
    const handlers = new Map();
    let offs = 0;
    const detach = store.attach({ ws: { on: (type, handler) => { handlers.set(type, handler); return () => { offs += 1; }; } } });
    const changed = handlers.get('ui_preferences_changed');
    changed({ type: 'ui_preferences_changed', keys: ['contributions'] });
    await flush();
    assert.equal(reads, 0, 'another group\'s change reads nothing');
    changed({ type: 'ui_preferences_changed', keys: ['contributions', 'model_pins'] });
    await flush();
    assert.equal(reads, 1);
    changed({ type: 'ui_preferences_changed' });
    await flush();
    assert.equal(reads, 2, 'a change that does not say which group may be the pins');
    detach();
    assert.equal(offs, 1);
    assert.doesNotThrow(() => store.attach()(), 'no socket: nothing to attach or detach');
});

test('the tiers come with the state body or a settings document; whether a role\'s catalog price is another tier\'s follows them', () => {
    const store = createModelCatalogStore();
    let heard = 0;
    store.subscribe(() => { heard += 1; });
    assert.equal(store.tiers(), null);
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead')], [true, true], 'until the tiers are read a price could be another tier\'s');
    store.adoptTiers({ main: '', default: 'standard' });
    assert.deepEqual([store.tiers(), heard], [{ main: '', default: 'standard' }, 1]);
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead')], [false, false]);
    store.adoptTiers({ main: '', default: 'standard' });
    assert.equal(heard, 1, 'the same tiers again change nothing');
    store.adoptTiers({ main: 'fast', default: 'standard' });
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead'), heard], [true, false, 2], 'Main pays its own tier, a lead the default');
    store.adoptTiers({ main: '', default: 'economy' });
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead')], [true, true], 'Main without its own tier takes the default');
    store.adoptTiers({ main: 'ultrafast', default: 'standard' });
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead')], [true, false]);
    store.adoptTiers({ main: 'standard', default: 'fast' });
    assert.deepEqual([store.tierPriced('main'), store.tierPriced('lead')], [false, true]);
    store.adoptTiers(null);
    store.adoptTiers('fast');
    assert.deepEqual([store.tiers(), heard], [{ main: 'standard', default: 'fast' }, 5], 'anything but an object is no tiers');
    store.adoptSettings({ OUROBOROS_PROCESSING_PREFERENCE: 'economy', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}' });
    assert.deepEqual(store.tiers(), { main: 'fast', default: 'economy' }, 'a settings document says the same in its own keys');
    store.adoptSettings({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'standard' } });
    assert.deepEqual(store.tiers(), { main: 'standard', default: '' });
    store.adoptSettings({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{not json' });
    assert.deepEqual(store.tiers(), { main: '', default: '' });
});

test('the index keeps the catalog\'s own price whatever the tiers; whoever says a row decides whether its context pays it', async () => {
    const opus = { ...item('anthropic/claude-opus-5.5'), price: { input: 4, output: 20 } };
    const store = createModelCatalogStore({ readCatalog: async () => ({ items: [opus] }) });
    await store.read();
    store.adoptTiers({ main: 'fast', default: 'economy' });
    assert.deepEqual(store.index().row(route(opus.value)).price, { input: 4, output: 20 }, 'never masked by a tier');
    assert.equal(describeRoute(store.index(), route(opus.value), { tier: 'economy' }).price, '');
    assert.equal(describeRoute(store.index(), route(opus.value), { tier: 'standard' }).price, '$4 in / $20 out');
});

test('a cache-write figure is said only while the settings name the cache tier it is of', async () => {
    // The server stamps a figure with the tier it was read under (`price.cache_write_ttl`).
    const priced = (figure, tier) => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: figure, cache_write_ttl: tier } }] });
    const said = (store) => describeRoute(store.index(), { model: 'a/one' }).cacheWrite;
    let reads = 0;
    let answer = () => priced(5, '5m');
    const store = createModelCatalogStore({ readCatalog: async () => { reads += 1; return answer(); } });
    await store.read();
    assert.equal(said(store), '', 'no settings document yet: no cache tier is named, so no figure is said');
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m', OUROBOROS_MODEL: 'a/one' });
    await store.read();
    assert.deepEqual([reads, said(store)], [1, 'Cache write: $5 per 1M tokens.'], 'the same tier leaves a fresh catalog alone');
    // Another tier: the held figure is the old tier's, so it is unsaid at once and the catalog is stale.
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    assert.equal(said(store), '');
    // A refresh that fails, or that answers with that provider's error (a 200 whose rows the merge keeps), never
    // brings the old figure back.
    answer = () => { throw new Error('offline'); };
    await store.read();
    assert.deepEqual([reads, said(store), store.status().failed], [2, '', true]);
    answer = () => ({ items: [], errors: [{ provider_id: 'openrouter', error: 'timed out' }] });
    await store.read();
    assert.deepEqual([reads, said(store), store.catalog().items.length], [3, '', 1], 'the row is kept, its figure unsaid');
    // That read was begun under the tier the settings name, so the catalog keeps its usual life: no read per capsule.
    await store.read();
    assert.equal(reads, 3);
    answer = () => priced(8, '1h');
    await store.read({ force: true });  // its next read: five minutes on, or Settings' own Refresh
    assert.deepEqual([reads, said(store)], [4, 'Cache write: $8 per 1M tokens.']);
    // A read begun under one tier that answers after the setting moved is the old tier's: unsaid and still stale.
    let release;
    answer = () => new Promise((resolve) => { release = () => resolve(priced(8, '1h')); });
    const late = store.read({ force: true });
    await Promise.resolve();
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    answer = () => priced(5, '5m');
    release();
    await late;
    assert.equal(said(store), '', 'begun under 1h, landed under 5m');
    await store.read();
    assert.deepEqual([reads, said(store)], [6, 'Cache write: $5 per 1M tokens.']);
    // Settings' own refresh hands its catalog over (`adopt`): one begun before a tier was saved carries the old tier.
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    store.adopt(priced(5, '5m'));
    assert.equal(said(store), '', 'a handed-over figure of the old tier');
    answer = () => priced(8, '1h');
    await store.read();
    assert.deepEqual([reads, said(store)], [7, 'Cache write: $8 per 1M tokens.']);
    // A handed-over figure of the tier the settings name is said: Settings' refresh feeds the store without a read.
    store.adopt(priced(9, '1h'));
    assert.equal(said(store), 'Cache write: $9 per 1M tokens.');
    await store.read();
    assert.equal(reads, 7, 'and it is fresh');
});

test('another cache tier makes a held catalog stale even where it states no figure: the new tier may state one', async () => {
    let reads = 0;
    const store = createModelCatalogStore({ readCatalog: async () => { reads += 1; return { items: [item('a/one')] }; } });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    await store.read();
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m', OUROBOROS_MODEL: 'a/one' });
    await store.read();
    assert.equal(reads, 1, 'the same tier: fresh');
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    await store.read();
    assert.equal(reads, 2);
});

test('a figure of a tier the settings do not name asks for one more read under that name, then the catalog keeps its life', async () => {
    let reads = 0;
    const store = createModelCatalogStore({ readCatalog: async () => {
        reads += 1;
        return { items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: 8, cache_write_ttl: '1h' } }] };
    } });
    await store.read();                                        // read before any settings document
    // A hand-edited value the server resolves to its default: no answer will carry it.
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: 'weekly' });
    await store.read();
    assert.equal(reads, 2, 'the first document names another tier than the held figure: one more read');
    await store.read();
    await store.read();
    assert.equal(reads, 2, 'and then none per opened capsule');
    assert.equal(describeRoute(store.index(), { model: 'a/one' }).cacheWrite, '', 'the figure is never said under another name');
});

test('a first catalog read that began before any settings document says the figure of the tier that document names', async () => {
    let release;
    let reads = 0;
    const answer = { items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: 8, cache_write_ttl: '1h' } }] };
    const store = createModelCatalogStore({ readCatalog: () => {
        reads += 1;
        return reads > 1 ? Promise.resolve(answer) : new Promise((resolve) => { release = () => resolve(answer); });
    } });
    const first = store.read();
    await Promise.resolve();
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1H' });   // the settings answer first: a local read; spelled as hand-edited
    release();
    await first;
    assert.equal(describeRoute(store.index(), { model: 'a/one' }).cacheWrite, 'Cache write: $8 per 1M tokens.');
    await store.read();
    assert.equal(reads, 1, 'and the catalog is fresh: no second read');
});

/* A key model's own endpoints (`?processing_model=<exact routed value>`): what its tiers are told by. */
const endpoints = (value, modes, extra = {}) => ({ value, processing: { modes, complete: true, endpoints: [
    { tag: 'a/standard', service_tier: null, mode: 'standard', price: { input: 4, output: 20 } },
    { tag: 'a/fast', service_tier: 'priority', mode: 'fast', price: { input: 8, output: 40, tiers: [{ min_prompt_tokens: 200000, input: 16, output: 60 }] } },
    { tag: 'b/unclassified', service_tier: 'turbo' }], ...extra } });

test('a key model\'s endpoints are read once for the route asked, kept under its own value, and never asked for another kind of route', async () => {
    let clock = 1000;
    const asked = [];
    const store = createModelCatalogStore({ now: () => clock, readProcessing: async (value) => { asked.push(value); return endpoints(value, ['standard', 'fast']); } });
    let heard = 0;
    store.subscribe(() => { heard += 1; });
    assert.equal(store.detail(route('z-ai/glm-5.3')), null, 'never asked: nothing is told');
    const reading = store.readDetail(route('z-ai/glm-5.3'));
    assert.deepEqual([store.detail(route('z-ai/glm-5.3')).state, heard], ['reading', 1]);
    assert.equal(store.readDetail(route('z-ai/glm-5.3')), reading, 'one read at a time for a model');
    await reading;
    const said = store.detail(route('z-ai/glm-5.3'));
    assert.deepEqual([said.state, [...said.modes], said.complete, heard], ['ok', ['standard', 'fast'], true, 2]);
    assert.deepEqual(said.endpoints, [{ mode: 'standard', tag: 'a/standard', price: { input: 4, output: 20 } },
        { mode: 'fast', tag: 'a/fast', price: { input: 8, output: 40, tiers: [{ min_prompt_tokens: 200000, input: 16, output: 60 }] } }],
    'an endpoint no tier was read for is none of a tier\'s');
    await store.readDetail(route('z-ai/glm-5.3'));
    assert.deepEqual(asked, ['z-ai/glm-5.3'], 'a fresh answer is not asked again');
    // Another model, a key added under Accounts: its own exact value, its own answer.
    await store.readDetail(route('openrouter@work::z-ai/glm-5.3'));
    assert.deepEqual(asked, ['z-ai/glm-5.3', 'openrouter@work::z-ai/glm-5.3']);
    assert.equal(store.detail(route('moonshotai/kimi-k3')), null, 'a model nobody opened was never asked');
    // A subscription's tiers are its accounts' lists, another provider's key has no such read, a local model none.
    for (const other of [route('claudexor::codex=gpt-6-astra'), route('openai::gpt-6'), route('qwen3.gguf', true), null, { model: '' }]) {
        await store.readDetail(other);
        assert.equal(store.detail(other), null);
    }
    assert.equal(asked.length, 2);
    clock += 5 * 60 * 1000 + 1;
    const again = store.readDetail(route('z-ai/glm-5.3'));
    assert.equal(store.detail(route('z-ai/glm-5.3')).state, 'reading', 'a stale answer is not this moment\'s: nothing is told while it is read again');
    await again;
    assert.deepEqual([asked.length, store.detail(route('z-ai/glm-5.3')).state], [3, 'ok']);
});

test('a late answer never stands under another model; a failed, foreign or tierless one tells nothing, never Standard only', async () => {
    const waiting = new Map();
    const store = createModelCatalogStore({ readProcessing: (value) => new Promise((resolve, reject) => { waiting.set(value, { resolve, reject }); }) });
    const state = (model) => store.detail(route(model))?.state || null;
    const firstRead = store.readDetail(route('a/selector-witness'));
    await Promise.resolve();
    waiting.get('a/selector-witness').resolve(endpoints('a/selector-witness', ['standard']));
    await firstRead;
    const [first, second] = [store.readDetail(route('a/one')), store.readDetail(route('a/two'))];
    await Promise.resolve();
    waiting.get('a/two').resolve(endpoints('a/two', ['standard', 'fast', 'ultrafast']));
    await second;
    assert.deepEqual([state('a/one'), state('a/two')], ['reading', 'ok'], 'the route chosen later answered first');
    waiting.get('a/one').resolve(endpoints('a/one', ['standard']));   // the earlier model's answer arrives late
    await first;
    assert.deepEqual([[...store.detail(route('a/one')).modes], [...store.detail(route('a/two')).modes]], [['standard'], ['standard', 'fast', 'ultrafast']],
        'each under its own value');
    // An answer for another value than the one asked, one that names no tier, and a refused read: none is a model's tiers.
    for (const [model, answer] of [['a/foreign', endpoints('a/other', ['standard', 'fast'])], ['a/empty', endpoints('a/empty', [])],
        ['a/shape', { value: 'a/shape', processing: { complete: true } }], ['a/refused', new Error('HTTP 502')]]) {
        const reading = store.readDetail(route(model));
        await Promise.resolve();
        if (answer instanceof Error) waiting.get(model).reject(answer); else waiting.get(model).resolve(answer);
        await reading;
        const said = store.detail(route(model));
        assert.deepEqual([said.state, [...said.modes], said.complete, said.endpoints], ['failed', [], false, []], model);
    }
    // A failed read is asked again by the next call (an opened sheet), not kept for the catalog's lifetime.
    const retry = store.readDetail(route('a/refused'));
    await Promise.resolve();
    waiting.get('a/refused').resolve(endpoints('a/refused', ['standard', 'economy']));
    await retry;
    assert.deepEqual([...store.detail(route('a/refused')).modes], ['standard', 'economy']);
});

test('a core that ignores the detail selector leaves modes unknown without repeatedly fetching its catalog', async () => {
    let reads = 0;
    const store = createModelCatalogStore({ readProcessing: async () => { reads += 1; return { items: [] }; } });
    const target = route('a/one');
    await store.readDetail(target);
    await store.readDetail(target);
    await store.readDetail(route('a/two'));
    const said = store.detail(target);
    assert.deepEqual([reads, said.complete, [...said.modes]], [1, false, []]);
});

test('concurrent cold detail reads share the legacy selector discovery', async () => {
    const reads = [];
    let respond;
    const catalog = new Promise((resolve) => { respond = resolve; });
    const store = createModelCatalogStore({ readProcessing: (value) => { reads.push(value); return catalog; } });
    const first = store.readDetail(route('a/one'));
    const second = store.readDetail(route('a/two'));
    respond({ items: [] });
    await Promise.all([first, second]);
    assert.deepEqual(reads, ['a/one']);
    for (const model of ['a/one', 'a/two']) assert.notEqual(store.detail(route(model))?.complete, true);
    await store.readDetail(route('a/two'));
    assert.equal(reads.length, 1);
});

test('the account Main is pinned to is the settings document\'s word, or a write\'s own answer; unknown before either', () => {
    const store = createModelCatalogStore();
    let heard = 0;
    store.subscribe(() => { heard += 1; });
    assert.equal(store.mainAccount(), null, 'not read: what follows Main is read as Auto, and the settings may still say');
    store.adoptSettings({ OUROBOROS_MODEL_ACCOUNTS: { main: 'work', light: 'home' } });
    assert.equal(store.mainAccount(), 'work');
    store.adoptSettings({ OUROBOROS_MODEL_ACCOUNTS: '{"main":"home"}' });
    assert.equal(store.mainAccount(), 'home', 'the stored spelling of the map is read too');
    store.adoptSettings({ OUROBOROS_MODEL_ACCOUNTS: '{not json' });
    assert.equal(store.mainAccount(), '', 'a map that cannot be read pins nothing');
    store.adoptSettings({});
    assert.equal(store.mainAccount(), '', 'no pin is Auto');
    const before = heard;
    store.adoptMainAccount('work');
    assert.deepEqual([store.mainAccount(), heard], ['work', before + 1], 'Main\'s own write says its account at once');
    store.adoptMainAccount('');
    assert.equal(store.mainAccount(), '');
});

test('another connection under the same routed value drops what its endpoints and transport said; a late answer of the earlier one is not kept', async () => {
    const waiting = [];
    let catalogReads = 0;
    const store = createModelCatalogStore({ readProcessing: (value) => new Promise((resolve) => { waiting.push({ value, resolve }); }),
        readCatalog: async () => { catalogReads += 1; return { items: [{ ...item('z-ai/glm-5.3'), processing_transport: ['standard', 'fast', 'ultrafast'] },
            { ...item('openai::gpt-6'), provider_id: 'openai', processing_transport: ['standard', 'fast'] }] }; } });
    const settings = (revision, light = 'light-1') => ({ OUROBOROS_CONNECTIONS: [], OPENROUTER_API_KEY: 'sk-or-v1...', OUROBOROS_MODEL_LIGHT: 'x',
        _meta: { tokens: { OUROBOROS_CONNECTIONS: 'rows-1', OUROBOROS_MODEL_LIGHT: light },
            connections: [{ id: 'openrouter', revision }, { id: 'openai', revision: 'openai#same' }] } });
    store.adoptSettings(settings('openrouter#r1'));
    await store.read();
    assert.equal(store.revisionOf(route('z-ai/glm-5.3')), 'openrouter#r1');
    assert.equal(store.revisionOf(route('claudexor::codex=gpt-6-astra')), '', 'a subscription has no API connection');
    const glm = route('z-ai/glm-5.3');
    const first = store.readDetail(glm);
    await Promise.resolve();
    waiting.shift().resolve(endpoints('z-ai/glm-5.3', ['standard', 'fast']));
    await first;
    assert.equal(store.detail(glm).state, 'ok');
    // An unrelated setting keeps the answer and the catalog.
    store.adoptSettings(settings('openrouter#r1', 'light-2'));
    assert.deepEqual([store.detail(glm).state, catalogReads], ['ok', 1], 'an unrelated edit is not another source');
    // The connection's revision moved (a rotated key shows the same masked prefix): nothing of the earlier one is told.
    const epoch = store.detailEpoch();
    store.adoptSettings(settings('openrouter#r2'));
    assert.deepEqual([store.detail(glm), store.detailEpoch(), store.revisionOf(glm)], [null, epoch + 1, 'openrouter#r2']);
    const rows = store.catalog().items;
    assert.ok(!('processing_transport' in rows.find((row) => row.value === 'z-ai/glm-5.3')), 'the changed source\'s transport is no limit now');
    assert.deepEqual(rows.find((row) => row.value === 'openai::gpt-6').processing_transport, ['standard', 'fast'], 'another source keeps its own');
    await flush();
    assert.equal(catalogReads, 2, 'and the catalog is read again');
    // A read out when the connection changes answers for the earlier one: not kept; the next call asks again.
    const late = store.readDetail(glm);
    await Promise.resolve();
    store.adoptSettings(settings('openrouter#r3'));
    waiting.shift().resolve(endpoints('z-ai/glm-5.3', ['standard', 'fast', 'ultrafast']));
    await late;
    assert.equal(store.detail(glm), null, 'a late answer of the earlier connection stands nowhere');
    const again = store.readDetail(glm);
    await Promise.resolve();
    assert.equal(waiting.length, 1, 'asked again under the connection now');
    waiting.shift().resolve(endpoints('z-ai/glm-5.3', ['standard']));
    await again;
    assert.deepEqual([...store.detail(glm).modes], ['standard']);
});

test('Main\'s binding read: the four tokens and the document whose connections say which source a route is shown on', async () => {
    const tokens = { OUROBOROS_MODEL: 'm', USE_LOCAL_MAIN: 'l', OUROBOROS_MODEL_ACCOUNTS: 'a', OUROBOROS_MODEL_PROCESSING_PREFERENCES: 'p' };
    const doc = { OUROBOROS_MODEL: 'openai@work::gpt-6', USE_LOCAL_MAIN: 'false', OUROBOROS_MODEL_ACCOUNTS: '{}',
        OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}', _meta: { tokens, connections: [{ id: 'openai@work', revision: 'openai@work#w1' }] } };
    let answer = doc;
    const store = createModelCatalogStore({ readSettings: async () => answer });
    const read = await store.readMainTierSettings();
    assert.deepEqual([read.route, read.tier, read.binding], [route('openai@work::gpt-6'), 'fast', tokens]);
    assert.deepEqual([revisionIn(read.settings, read.route), revisionIn(read.settings, route('openai::gpt-6')), revisionIn(read.settings, route('qwen3', true))],
        ['openai@work#w1', null, ''], 'the named key\'s own revision; another key the read does not name; the local model none');
    assert.deepEqual([apiConnectionOf(route('z-ai/glm-5.3')), apiConnectionOf(route('claudexor::codex=gpt-6-astra'))], ['openrouter', '']);
    answer = { ...doc, _meta: { tokens: { OUROBOROS_MODEL: 'm' } } };
    await assert.rejects(store.readMainTierSettings(), /Could not verify Main/);
    const catalog = { items: [{ value: 'openai@work::gpt-6', processing_transport: ['standard'] }, { value: 'qwen3', use_local: true, processing_transport: ['standard'] }] };
    assert.deepEqual(dropTransport(catalog, new Set(['openai@work'])).items.map((row) => 'processing_transport' in row), [false, true],
        'the local adapter is no API connection that changed');
});

test('a catalog read out when a key rotates never revives the earlier source\'s transport; a read of the new one follows, others keep theirs', async () => {
    const answers = [];
    const store = createModelCatalogStore({ readCatalog: () => new Promise((resolve) => { answers.push(resolve); }) });
    const rows = (glm, gpt) => ({ items: [{ ...item('z-ai/glm-5.3'), processing_transport: glm },
        { ...item('openai::gpt-6'), provider_id: 'openai', processing_transport: gpt }] });
    const settings = (revision) => ({ _meta: { tokens: {}, connections: [{ id: 'openrouter', revision }, { id: 'openai', revision: 'openai#same' }] } });
    const transport = (value) => store.catalog()?.items.find((row) => row.value === value)?.processing_transport;
    // What the tier strip and Settings actually consume: the index's word on the route (its raw `current` answer).
    const ceiling = (value) => store.index().tiers(route(value))?.ceiling ?? null;
    store.adoptSettings(settings('openrouter#r1'));
    const first = store.read();
    answers.shift()(rows(['standard', 'fast', 'ultrafast'], ['standard', 'fast']));
    await first;
    assert.deepEqual([ceiling('z-ai/glm-5.3'), ceiling('openai::gpt-6')], [['standard', 'fast', 'ultrafast'], ['standard', 'fast']]);
    // A refresh goes out under r1 and is held; the key rotates to r2 while it is out.
    const held = store.read({ force: true });
    store.adoptSettings(settings('openrouter#r2'));
    let refreshed = false;
    const next = store.read({ force: true }).then(() => { refreshed = true; });
    assert.equal(transport('z-ai/glm-5.3'), undefined, 'nothing of the earlier source is told at once');
    assert.deepEqual([ceiling('z-ai/glm-5.3'), ceiling('openai::gpt-6')], [null, ['standard', 'fast']],
        'the index too: the changed key\'s ceiling is unknown (never Standard-only), the unchanged one keeps its own');
    answers.shift()(rows(['standard', 'fast', 'ultrafast'], ['standard', 'fast']));  // the held answer: still r1's
    await held;
    assert.equal(transport('z-ai/glm-5.3'), undefined, 'the held answer does not bring the earlier transport back');
    assert.deepEqual(transport('openai::gpt-6'), ['standard', 'fast'], 'a source that did not change keeps its own');
    assert.deepEqual([ceiling('z-ai/glm-5.3'), ceiling('openai::gpt-6')], [null, ['standard', 'fast']], 'the index after the held answer');
    // Another reader's answer (Settings' refresh, by its event) while the new source is still owed: the same.
    store.adopt(rows(['standard', 'fast', 'ultrafast'], ['standard', 'fast']));
    assert.equal(transport('z-ai/glm-5.3'), undefined);
    assert.equal(ceiling('z-ai/glm-5.3'), null, 'nor through another reader\'s answer');
    await flush();
    assert.equal(answers.length, 1, 'a read under the new source follows the held one');
    assert.equal(refreshed, false, 'callers recovering from a refusal wait for the queued read, not the one already out');
    answers.shift()(rows(['standard'], ['standard', 'fast']));
    await next;
    assert.deepEqual(transport('z-ai/glm-5.3'), ['standard'], 'and its answer is told whole');
    assert.deepEqual([ceiling('z-ai/glm-5.3'), ceiling('openai::gpt-6')], [['standard'], ['standard', 'fast']]);
});

test('a settings read out when a newer document is adopted never writes the older one back', async () => {
    let answer = () => {};
    const store = createModelCatalogStore({ readSettings: () => new Promise((resolve) => { answer = resolve; }) });
    const model = 'claudexor::codex=gpt-6-astra';
    const settings = (revision, main) => ({ OUROBOROS_MODEL: model, USE_LOCAL_MAIN: false, OUROBOROS_MODEL_ACCOUNTS: { main },
        _meta: { tokens: {}, connections: [{ id: 'openrouter', revision }] } });
    const reading = store.readConnections({ fresh: true });  // sent while the key is r1
    await flush();  // the read is out
    store.adoptSettings(settings('openrouter#r2', 'work'));  // the Settings page (or confirmation) delivered a newer document
    answer(settings('openrouter#r1', 'home'));
    await reading;
    assert.equal(store.revisionOf(route('z-ai/glm-5.3')), 'openrouter#r2', 'the older answer does not stand over the newer');
    assert.equal(store.mainAccount(route(model)), 'work', 'not even for a different pin on the exact same route');
    const again = store.readConnections({ fresh: true });  // a read sent after it is adopted
    await flush();
    answer(settings('openrouter#r3', ''));
    await again;
    assert.equal(store.revisionOf(route('z-ai/glm-5.3')), 'openrouter#r3');
});

test('first settings adoption discards unbound endpoint facts, including outstanding detail and catalog reads', async () => {
    const pending = [], catalogs = [];
    const model = 'vendor/model';
    const store = createModelCatalogStore({
        readProcessing: () => new Promise((resolve) => pending.push(resolve)),
        readCatalog: () => new Promise((resolve) => catalogs.push(resolve)),
    });
    const settled = route('another/model');
    const first = store.readDetail(settled);
    await Promise.resolve();
    pending.shift()(endpoints(settled.model, ['standard', 'fast']));
    await first;
    const late = store.readDetail(route(model));
    const catalog = store.read();
    await Promise.resolve();
    store.adoptSettings({ _meta: { connections: [{ id: 'openrouter', revision: 'openrouter#first' }] } });
    assert.equal(store.detail(settled), null);
    pending.shift()(endpoints(model, ['standard', 'ultrafast']));
    catalogs.shift()({ items: [{ ...item(model), processing_transport: ['standard', 'ultrafast'] }] });
    await Promise.all([late, catalog]);
    assert.equal(store.detail(route(model)), null);
    assert.equal(store.index().tiers(route(model))?.ceiling ?? null, null);
    catalogs.shift()({ items: [{ ...item(model), processing_transport: ['standard'] }] });
    await flush();
    assert.deepEqual(store.index().tiers(route(model)).ceiling, ['standard']);
});

for (const shape of ['model row', 'rowless source']) test(`the first settings revision rereads a ${shape}’s transport`, async () => {
    const rowless = shape === 'rowless source';
    const id = rowless ? 'openrouter@work' : 'openrouter';
    const model = rowless ? `${id}::vendor/model` : 'vendor/model';
    const response = (tiers) => rowless ? { items: [], api_sources: [{ id, label: 'Work', processing_transport: tiers }] }
        : { items: [{ ...item(model), processing_transport: tiers }] };
    let finish;
    const store = createModelCatalogStore({ readCatalog: () => new Promise((resolve) => { finish = resolve; }) });
    const read = store.read();
    finish(response(['standard', 'ultrafast']));
    await read;
    const ceiling = () => store.index().tiers(route(model)).ceiling;
    assert.deepEqual(ceiling(), ['standard', 'ultrafast']);
    store.adoptSettings({ _meta: { connections: [{ id, revision: `${id}#first` }] } });
    assert.equal(ceiling(), null);
    if (rowless) assert.equal(store.index().source(id).count, 0);  // the source survives, independently of its transport fact
    else assert.equal(store.catalog().items[0].value, model);  // older catalogs keep their model row too
    finish(response(['standard']));
    await flush();
    assert.deepEqual(ceiling(), ['standard']);
});

test('a work following Main gets the account read for its exact route, not an earlier one', async () => {
    const a = 'claudexor::codex=gpt-6-astra', b = 'claudexor::codex=gpt-6.2-sol';
    const settings = (model, pin, local = false) => ({ OUROBOROS_MODEL: model, USE_LOCAL_MAIN: String(local),
        OUROBOROS_MODEL_ACCOUNTS: { main: pin } });
    let answer = settings(a, 'account-a');
    const store = createModelCatalogStore({ readSettings: async () => answer });
    await store.readConnections();
    assert.equal(store.mainAccount(), 'account-a');
    assert.equal(store.mainAccount(route(a)), 'account-a');
    assert.equal(store.mainAccount(route(b)), null); // pending model B, but Settings still only names A
    assert.equal(store.mainAccount(route(a, true)), null);

    answer = settings(b, '');
    await store.readConnections({ fresh: true });
    assert.equal(store.mainAccount(route(b)), ''); // read Auto, not unknown
    assert.equal(store.mainAccount(route(a)), null);
    store.adoptSettings(settings(b, 'account-b', true));
    assert.equal(store.mainAccount(route(b)), null);
    assert.equal(store.mainAccount(route(b, true)), 'account-b');
    store.adoptMainAccount('');
    assert.equal(store.mainAccount(), '');
    assert.equal(store.mainAccount(route(b, true)), null); // a write's unscoped answer proves no read binding
    answer = settings(b, 'stale');
    const old = store.readConnections({ fresh: true }); // the preceding Settings document is still in flight
    store.adoptMainAccount('', route(a)); // the latest write confirmed Auto for A, not B
    await old;
    assert.equal(store.mainAccount(route(a)), '');
    assert.equal(store.mainAccount(route(b)), null);
});

test('a refresh failure handed over without items keeps the rows and says the read failed; a new success brings it back', async () => {
    const store = createModelCatalogStore({ readCatalog: async () => ({ items: [item('a/one')] }) });
    await store.read();
    assert.equal(store.catalog().read_state, 'ok');
    // What a failed refresh hands the store: the outcome, with no rows of its own.
    store.adopt({ read_state: 'transport', errors: [{ provider_id: 'catalog', error: 'offline' }] });
    assert.equal(store.catalog().read_state, 'transport', 'the read outcome is the price\'s own state');
    assert.deepEqual(store.index().rows.map((row) => row.id), ['a/one'], 'the rows it listed stay for the chooser');
    store.adopt({ read_state: 'ok', items: [item('a/one')] });
    assert.equal(store.catalog().read_state, 'ok', 'a new success says the price again');
    assert.deepEqual(store.index().rows.map((row) => row.id), ['a/one']);
});

test('a handed-over failure keeps the choices, says the read failed and leaves the next opening to read again', async () => {
    let reads = 0;
    const store = createModelCatalogStore({ readCatalog: async () => { reads += 1; return { items: [item('a/one')] }; } });
    await store.read();
    assert.deepEqual([reads, store.status().failed], [1, false]);
    store.adopt({ read_state: 'transport', errors: [{ provider_id: 'catalog', error: 'offline' }] });
    assert.equal(store.status().failed, true, 'the handed-over failure is a failure of the store too');
    assert.equal(store.catalog().read_state, 'transport');
    assert.deepEqual(store.index().rows.map((row) => row.id), ['a/one'], 'the choices stay for the chooser');
    await store.read();
    assert.equal(reads, 2, 'the next opening reads again: the provider may have come back');
    assert.equal(store.status().failed, false);
    // A cold failure lists nothing and still leaves the next opening to read.
    let cold = 0;
    const empty = createModelCatalogStore({ readCatalog: async () => { cold += 1; throw new Error('offline'); } });
    await empty.read();
    assert.deepEqual([empty.status().failed, empty.index().rows.length], [true, 0]);
    assert.equal(empty.catalog().read_state, 'transport');
    await empty.read();
    assert.equal(cold, 2, 'a cold failure is read again on the next opening');
});

test('a cold chooser that answers HTTP 200 with a stated failure reads again; an empty or partial answer is not one', async () => {
    let reads = 0;
    const answers = [
        { read_state: 'failed', items: [], errors: [{ provider_id: 'openrouter', error: 'no catalog' }] },
        { read_state: 'ok', items: [item('a/one')] },
    ];
    const store = createModelCatalogStore({ readCatalog: async () => answers[reads++] });
    await store.read();
    assert.deepEqual([reads, store.status().failed], [1, true],
        'an empty answer that STATES its failure is not a fresh catalog');
    await store.read();
    assert.deepEqual([reads, store.status().failed], [2, false], 'the next opening reads again');
    // A successful empty read is not a failure: the catalog really lists nothing, and it keeps its life.
    let emptyReads = 0;
    const empty = createModelCatalogStore({ readCatalog: async () => { emptyReads += 1; return { read_state: 'ok', items: [] }; } });
    await empty.read();
    await empty.read();
    assert.deepEqual([emptyReads, empty.status().failed], [1, false], 'an empty catalog is fresh, not a failure');
    // A partial read kept what it could read and is not a failure either.
    let partialReads = 0;
    const partial = createModelCatalogStore({ readCatalog: async () => {
        partialReads += 1;
        return { read_state: 'partial', items: [item('a/one')], errors: [{ error: 'one source failed' }] };
    } });
    await partial.read();
    await partial.read();
    assert.deepEqual([partialReads, partial.status().failed], [1, false], 'a partial read is not a failure');
});

// #3493: the corrective read is per attempted read. Two ways a held figure can end up of another tier without
// this store having read for it: the tier swings away and back across one read (the server answers under the
// middle name), and another reader's answer of an older tier is handed over late. Each owes exactly one read;
// where that read cannot bring the named tier's figure the catalog keeps its usual life.

test('a tier that swings away and back across a read owes one corrective read, then the catalog keeps its life', async () => {
    const priced = (figure, tier) => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: figure, cache_write_ttl: tier } }] });
    const said = (store) => describeRoute(store.index(), { model: 'a/one' }).cacheWrite;
    let reads = 0;
    let release = null;
    const answers = [() => priced(8, '1h'), () => new Promise((resolve) => { release = () => resolve(priced(5, '5m')); }),
        () => priced(8, '1h')];
    const store = createModelCatalogStore({ readCatalog: () => { reads += 1; return answers[reads - 1](); } });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    await store.read();
    assert.deepEqual([reads, said(store)], [1, 'Cache write: $8 per 1M tokens.']);
    const late = store.read({ force: true });
    await Promise.resolve();
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });   // 1h→5m→1h across the read
    release();
    await late;
    assert.equal(said(store), '', 'the read answered under the middle name: its figure is unsaid under 1h');
    await store.read();
    assert.deepEqual([reads, said(store)], [3, 'Cache write: $8 per 1M tokens.'], 'exactly one corrective read');
    await store.read();
    assert.equal(reads, 3, 'and no read per opened capsule');
});

test('a corrective read that still cannot bring the named tier\'s figure is the one owed read', async () => {
    const priced = (figure, tier) => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: figure, cache_write_ttl: tier } }] });
    const said = (store) => describeRoute(store.index(), { model: 'a/one' }).cacheWrite;
    let reads = 0;
    let release = null;
    const answers = [() => priced(8, '1h'), () => new Promise((resolve) => { release = () => resolve(priced(5, '5m')); }),
        () => priced(5, '5m')];
    const store = createModelCatalogStore({ readCatalog: () => { reads += 1; return answers[reads - 1](); } });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    await store.read();
    const late = store.read({ force: true });
    await Promise.resolve();
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    release();
    await late;
    await store.read();                       // the corrective read answers 5m's figure again
    assert.deepEqual([reads, said(store)], [3, '']);
    await store.read();
    await store.read();
    assert.equal(reads, 3, 'the one owed read was made; the unsaid figure keeps the catalog\'s usual life');
});

test('a late answer handed over by another reader owes one corrective read even where this store read under the named tier', async () => {
    const priced = (figure, tier) => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: figure, cache_write_ttl: tier } }] });
    const said = (store) => describeRoute(store.index(), { model: 'a/one' }).cacheWrite;
    let reads = 0;
    const store = createModelCatalogStore({ readCatalog: async () => { reads += 1; return priced(8, '1h'); } });
    store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    await store.read();
    assert.deepEqual([reads, said(store)], [1, 'Cache write: $8 per 1M tokens.']);
    // Settings' own reader answers late with figures fetched under an older tier name.
    store.adopt(priced(5, '5m'));
    assert.equal(said(store), '', 'another tier\'s figure is unsaid at once');
    await store.read();
    assert.deepEqual([reads, said(store)], [2, 'Cache write: $8 per 1M tokens.'], 'exactly one corrective read');
    await store.read();
    assert.equal(reads, 2);
});

// The same sequences with an answer that states no cache-write figure, which cannot show the tier it was read under.
// 'a/one' states a five-minute write figure and no hourly one, so an answer read under 1h carries no figure for it.
const figureless = () => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20 } }] });
const stamped = (figure, tier) => ({ items: [{ ...item('a/one'), price: { input: 4, output: 20, cache_write: figure, cache_write_ttl: tier } }] });
const saidBy = (store) => describeRoute(store.index(), { model: 'a/one' }).cacheWrite;
const MINUTE = 60 * 1000;

// On the store's own clock, which starts at one second (a stamp of 0 would pass for fresh there): the answer that
// lands leaves the catalog stale at that very instant, the next ordinary open reads once under the named tier, and
// that read is fresh within the same window and for the catalog's usual five minutes, never longer.
function noFigureRig() {
    const rig = { clock: 1_000, reads: 0, release: null, next: () => stamped(5, '5m') };
    rig.store = createModelCatalogStore({ now: () => rig.clock, readCatalog: () => { rig.reads += 1; return rig.next(); } });
    // A read begun now stays out until `release(answer)`; the reads after it answer under the five-minute tier.
    rig.outstanding = () => {
        rig.next = () => new Promise((resolve) => { rig.release = resolve; });
        const reading = rig.store.read();
        rig.next = () => stamped(5, '5m');
        return reading;
    };
    rig.readUnder1h = async () => {
        rig.next = figureless;
        await rig.store.read();
        rig.next = () => stamped(5, '5m');
    };
    return rig;
}
const NO_FIGURE_SEQUENCES = {
    'the first catalog read lands before Settings names a tier': async (rig) => {
        const opened = rig.outstanding();
        rig.release(figureless());
        await opened;
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
    },
    'Settings first names a tier while the initial catalog read is out': async (rig) => {
        const opened = rig.outstanding();
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
        rig.release(figureless());
        await opened;
    },
    'a sheet\'s read is out when the tier moves': async (rig) => {
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
        const opened = rig.outstanding();                              // a Model sheet opens: its read is begun under 1h
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' }); // the owner saves 5 min while it is out
        rig.release(figureless());
        await opened;
    },
    'the tier swings away and back across that read': async (rig) => {
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
        await rig.store.read();
        rig.clock += 6 * MINUTE;                                       // its life is over: the next open reads
        const opened = rig.outstanding();
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' }); // back, before the read lands under the middle name
        rig.release(figureless());
        await opened;
    },
    'another reader\'s answer begun under the old tier lands after the move': async (rig) => {
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
        await rig.readUnder1h();
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' }); // Settings › Models' refresh is out meanwhile
        rig.store.adopt(figureless());
    },
    'that answer lands after a read under the new tier': async (rig) => {
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
        await rig.readUnder1h();
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
        await rig.store.read();                                        // a Model capsule already read under 5 min
        assert.equal(saidBy(rig.store), 'Cache write: $5 per 1M tokens.');
        rig.store.adopt(figureless());                                 // and the late refresh unsays its figure
    },
};
for (const [sequence, land] of Object.entries(NO_FIGURE_SEQUENCES)) {
    test(`an answer that states no figure leaves the catalog stale where ${sequence}; one read follows`, async () => {
        const rig = noFigureRig();
        await land(rig);
        const landed = rig.reads;
        assert.equal(saidBy(rig.store), '', 'the answer that landed states no figure');
        await rig.store.read();                                        // an open at the very instant it landed
        assert.deepEqual([rig.reads - landed, saidBy(rig.store)], [1, 'Cache write: $5 per 1M tokens.'],
            'it never stamped the catalog fresh: one read under the named tier');
        rig.clock += 5 * 1000;
        await rig.store.read();                                        // the same window, seconds later
        assert.equal(rig.reads - landed, 1, 'that read made it fresh');
        rig.clock += 4 * MINUTE;
        await rig.store.read();
        assert.equal(rig.reads - landed, 1, 'and keeps it fresh from its own answer');
        rig.clock += 2 * MINUTE;
        await rig.store.read();
        assert.equal(rig.reads - landed, 2, 'for the catalog\'s usual life, never longer');
    });
}

for (const prior of ['1h', null]) {
    test(`an open while a read begun under ${prior ?? 'no tier'} is out waits for it, then reads once under the named tier`, async () => {
        const rig = noFigureRig();
        if (prior) rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: prior });
        const opened = rig.outstanding();
        rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '5m' });
        const meanwhile = [rig.store.read(), rig.store.read()];       // the sheet opened again, twice, before it lands
        assert.equal(rig.reads, 1, 'the new opens join the old read before correcting it');
        rig.release(figureless());
        await Promise.all([opened, ...meanwhile]);
        assert.deepEqual([rig.reads, saidBy(rig.store)], [2, 'Cache write: $5 per 1M tokens.'], 'one read for every open meanwhile');
        await rig.store.read();
        assert.equal(rig.reads, 2);
    });
}

test('with the tier unchanged an answer that states no figure, this store\'s or handed over, keeps the usual life', async () => {
    const rig = noFigureRig();
    rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });
    await rig.readUnder1h();
    rig.store.adopt(figureless());                                     // Settings › Models' own refresh, same tier
    rig.store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h' });     // a save of another setting
    rig.clock += 4 * MINUTE;
    await rig.store.read();
    assert.equal(rig.reads, 1, 'no read the tier does not owe');
    rig.clock += 2 * MINUTE;
    await rig.store.read();
    assert.equal(rig.reads, 2, 'and the usual one when its life ends');
});

test('a settings read out when Settings applies a newer document never stands over it, answered or failed', async () => {
    const connection = (name, enabled) => ({ id: `openai@${name}`, kind: 'openai', name, display: `OpenAI · ${name}`, enabled,
        configured: true });
    for (const late of ['answers', 'fails']) {
        let settle;
        let reads = 0;
        const store = createModelCatalogStore({ readCatalog: async () => stamped(5, '1h'), readSettings: () => {
            reads += 1;
            return new Promise((resolve, reject) => { settle = { resolve, reject }; });
        } });
        const reading = store.readConnections();
        await flush();
        assert.equal(reads, 1, 'the store\'s own first read is out');
        store.adoptSettings({ OUROBOROS_PROMPT_CACHE_TTL: '1h', OUROBOROS_PROCESSING_PREFERENCE: 'economy',
            OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}', _meta: { connections: [connection('new', true)] } });
        if (late === 'answers') {
            settle.resolve({ OUROBOROS_PROMPT_CACHE_TTL: '5m', OUROBOROS_PROCESSING_PREFERENCE: 'standard',
                _meta: { connections: [connection('old', false)] } });
        } else settle.reject(new Error('offline'));
        await reading;
        await store.read();
        assert.deepEqual(store.tiers(), { main: 'fast', default: 'economy' }, `${late}: the applied document's tiers stay`);
        assert.deepEqual(store.index().connections.map((entry) => [entry.id, entry.enabled]), [['openai@new', true]], late);
        assert.equal(saidBy(store), 'Cache write: $5 per 1M tokens.', `${late}: the applied cache tier keeps its figure said`);
        await store.readConnections();
        assert.equal(reads, 1, `${late}: the document is held, so nothing is read again`);
    }
});

test('with no newer document the store\'s own settings read is the one adopted', async () => {
    let settle;
    const store = createModelCatalogStore({ readSettings: () => new Promise((resolve) => { settle = resolve; }) });
    const reading = store.readConnections();
    await flush();
    settle({ OUROBOROS_PROCESSING_PREFERENCE: 'standard', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}' });
    await reading;
    assert.deepEqual(store.tiers(), { main: 'fast', default: 'standard' });
});
