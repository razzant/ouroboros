// The composer's one client store over /api/model-catalog (model_catalog_store.js): a read when asked and then only
// when stale or failed, a failed read keeping what an earlier one listed, the owner's recent models from the
// preferences (none while the server does not carry them), and the settings document's connections read once.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';

const item = (id) => ({ provider_id: 'openrouter', provider: 'OpenRouter', id, name: id, value: id });

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
