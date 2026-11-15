// The composer's one client store over /api/model-catalog (model_catalog_store.js): a read when asked and then only
// when stale or failed, a failed read keeping what an earlier one listed, the owner's recent and pinned models from the
// preferences (none while the server does not carry them), a pin as one addressed write, the processing tiers Main and
// a lead take, and the settings document's connections read once.
import assert from 'node:assert/strict';
import test from 'node:test';
import { describeRoute } from '../modules/chat_model_routes.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';

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
