// A review slot's abort ends the card-detail GET it started; a read it only joined stays with its owner.

import assert from 'node:assert/strict';
import test from 'node:test';

import { cardDetailReader, readMissingCardDetail } from '../modules/settled_card.js';

// Let a settled mocked fetch travel the real promise chain; no timers, so nothing can hang the run.
const flush = async () => { for (let i = 0; i < 60; i += 1) await Promise.resolve(); };

function harness() {
    const prior = globalThis.fetch, signals = [], fail = [];
    globalThis.fetch = (_url, init = {}) => new Promise((_, reject) => {
        signals.push(init.signal);
        fail.push(() => reject(new Error('end of test')));
        init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('x'), { name: 'AbortError' })));
    });
    const record = { groupId: 'g1', root: { dataset: {} } };
    return { signals, fail, record, read: cardDetailReader(new Map([['g1', record]]), () => assert.fail('plain')),
        restore: () => { globalThis.fetch = prior; } };
}

test('the initiating review slot abort cancels the physical GET', async () => {
    const h = harness();
    try {
        const slot = new AbortController();
        const done = h.read('g1', { signal: slot.signal, revision: 'r1' }).then(() => 'ok', (e) => e.name);
        await new Promise((r) => setTimeout(r, 0));
        assert.equal(h.signals.length, 1);
        slot.abort();
        assert.equal(await done, 'StateReadEnded');
        assert.equal(h.signals[0].aborted, true);
    } finally { h.restore(); }
});

test('a review that joined another reader does not cancel that read', async () => {
    const h = harness();
    try {
        const owner = h.read('g1', {}); // no signal: the census-style owner
        void owner.catch(() => {});
        await new Promise((r) => setTimeout(r, 0));
        const slot = new AbortController();
        void h.read('g1', { signal: slot.signal }).catch(() => {});
        await new Promise((r) => setTimeout(r, 0));
        slot.abort();
        await new Promise((r) => setTimeout(r, 0));
        assert.equal(h.signals.length, 1);
        assert.equal(h.signals[0].aborted, false);
    } finally { h.fail.forEach((end) => end()); h.restore(); }
});

// The inverse witness, on the real consumer seam: once the census has joined the read a review slot started,
// that slot's abort must leave one live request whose terminal answer still reaches the census with no false Retry.
test('an aborting review slot leaves the joined census read to deliver its terminal detail', async () => {
    const prior = globalThis.fetch, transports = [], settle = [];
    globalThis.fetch = (_url, init = {}) => new Promise((resolve, reject) => {
        transports.push(init.signal);
        settle.push(() => resolve({ ok: true, json: async () => ({ status: 'completed' }) }));
        init.signal?.addEventListener('abort', () => reject(Object.assign(new Error('x'), { name: 'AbortError' })));
    });
    const record = { groupId: 'g1', root: { dataset: {} } }, applied = [], reports = [];
    const read = cardDetailReader(new Map([['g1', record]]), () => assert.fail('a card must not read plain'));
    try {
        const slot = new AbortController();
        void read('g1', { signal: slot.signal, revision: 'r1' }).catch(() => {});
        await flush();
        readMissingCardDetail(record, () => true, (detail) => applied.push(detail), (kind) => reports.push(kind));
        await flush();
        assert.equal(transports.length, 1, 'the census joins the in-flight read instead of racing a second GET');
        slot.abort();
        await flush();
        assert.deepEqual(applied, [], 'the abandoned slot pre-empts nothing: the read is still owed its answer');
        settle[0]();
        await flush();
        assert.equal(applied.length, 1, 'the terminal detail still reaches the census');
        assert.equal(applied[0].status, 'completed');
        assert.equal(transports.length, 1, 'the applied answer came from the one request that stayed alive');
        assert.deepEqual(reports, ['idle'], 'a delivered read is not reported as a failure needing Retry');
    } finally {
        globalThis.fetch = prior;
        settle.forEach((release) => release());
    }
});
