import test from 'node:test';
import assert from 'node:assert/strict';

import { createStateSnapshotSequencer } from '../modules/chat_activity.js';

// The reader below mirrors chat.js::refreshHeaderControlState and
// app.js::refreshProjectsNav: admission through gate(force), then one fetch,
// then apply on success or fail on any error. The fake fetch is settled by the
// test so overlap is deterministic.
function harness() {
    const applied = [];
    const events = [];
    const sequencer = createStateSnapshotSequencer(
        (data, requestedAt, generation) => applied.push({ data, requestedAt, generation }),
        () => 1000 + events.length, () => events.push('unavailable'));
    const fetches = [];
    const fetchState = () => new Promise((resolve, reject) => { fetches.push({ resolve, reject }); });
    async function refresh(force = false) {
        const request = await sequencer.gate(force);
        if (!request) return 'skipped';
        try {
            const resp = await fetchState();
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            sequencer.apply(request, resp.data);
            return 'applied';
        } catch {
            sequencer.fail(request);
            return 'failed';
        }
    }
    return { sequencer, applied, events, fetches, refresh };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('periodic ticks while a read is in flight are skipped, not queued', async () => {
    const h = harness();
    const first = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 1);
    const second = h.refresh(false);
    const third = h.refresh(false);
    let secondSettled = false;
    second.then(() => { secondSettled = true; });
    await tick();
    assert.equal(h.fetches.length, 1, 'ticks during a read start no fetch');
    assert.equal(secondSettled, false, 'a tick joins the in-flight read instead of resolving early');
    for (const f of h.fetches) f.resolve({ ok: true, data: 'one' });
    assert.deepEqual(await Promise.all([first, second, third]), ['applied', 'skipped', 'skipped']);
    assert.deepEqual(h.applied.map((a) => a.data), ['one']);
    // The gate is free again: the next tick reads.
    const next = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 2);
    h.fetches[1].resolve({ ok: true, data: 'two' });
    assert.equal(await next, 'applied');
});

test('forced refreshes during a read coalesce into one follow-up that resolves after it applies', async () => {
    const h = harness();
    const periodic = h.refresh(false);
    await tick();
    const forcedA = h.refresh(true);
    const forcedB = h.refresh(true);
    const laterTick = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 1, 'forced callers wait for the in-flight read');
    h.fetches[0].resolve({ ok: true, data: 'stale-ish' });
    assert.equal(await periodic, 'applied');
    assert.equal(await laterTick, 'skipped');
    await tick();
    assert.equal(h.fetches.length, 2, 'exactly one follow-up read starts after settle');
    let settledA = false;
    let settledB = false;
    forcedA.then(() => { settledA = true; });
    forcedB.then(() => { settledB = true; });
    await tick();
    assert.equal(settledA, false);
    assert.equal(settledB, false);
    h.fetches[1].resolve({ ok: true, data: 'fresh' });
    const outcomes = await Promise.all([forcedA, forcedB]);
    assert.deepEqual(outcomes.sort(), ['applied', 'skipped']);
    assert.deepEqual(h.applied.map((a) => a.data), ['stale-ish', 'fresh']);
    assert.equal(settledA && settledB, true, 'both forced callers observe the completed follow-up');
    await tick();
    assert.equal(h.fetches.length, 2, 'no third read was queued');
});

test('an HTTP error and a thrown fetch both release the gate', async () => {
    const h = harness();
    const first = h.refresh(false);
    await tick();
    h.fetches[0].resolve({ ok: false, status: 503 });
    assert.equal(await first, 'failed');
    assert.deepEqual(h.events, ['unavailable']);
    const second = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 2, 'HTTP error released the gate');
    h.fetches[1].reject(new Error('network down'));
    assert.equal(await second, 'failed');
    const third = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 3, 'thrown fetch released the gate');
    h.fetches[2].resolve({ ok: true, data: 'back' });
    assert.equal(await third, 'applied');
    assert.deepEqual(h.applied.map((a) => a.data), ['back']);
});

test('a forced caller coalesced behind a failing read still gets its follow-up', async () => {
    const h = harness();
    const first = h.refresh(false);
    await tick();
    const forced = h.refresh(true);
    await tick();
    h.fetches[0].reject(new Error('abort'));
    assert.equal(await first, 'failed');
    await tick();
    assert.equal(h.fetches.length, 2);
    h.fetches[1].resolve({ ok: true, data: 'after-failure' });
    assert.equal(await forced, 'applied');
    assert.deepEqual(h.applied.map((a) => a.data), ['after-failure']);
});

test('generation ordering is unchanged: a synthetic bump makes the in-flight read stale but still settles it', async () => {
    const h = harness();
    const inflight = h.refresh(false);
    await tick();
    // app.js does this on socket open/close: an ungated generation bump.
    h.sequencer.fail(h.sequencer.begin());
    assert.deepEqual(h.events, ['unavailable']);
    h.fetches[0].resolve({ ok: true, data: 'older' });
    assert.equal(await inflight, 'applied');
    assert.deepEqual(h.applied, [], 'the older response cannot apply after a newer generation');
    const next = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 2, 'the stale settle still released the gate');
    h.fetches[1].resolve({ ok: true, data: 'newer' });
    await next;
    assert.deepEqual(h.applied.map((a) => a.data), ['newer']);
});

test('gate requests carry the same clock as begin() and settle through apply of the same object only', async () => {
    const h = harness();
    const a = await h.sequencer.gate(true);
    assert.deepEqual(a, { generation: 1, requestedAt: 1000 });
    const pending = () => Promise.race([h.sequencer.gate(false).then(() => 'settled'), tick().then(() => 'pending')]);
    assert.equal(await pending(), 'pending');
    // A different object with the same generation does not settle the gate.
    assert.equal(h.sequencer.apply({ ...a }, 'copy'), true);
    assert.equal(await pending(), 'pending');
    assert.equal(h.sequencer.apply(a, 'same'), false);
    // Settled: the next tick opens a fresh read instead of joining.
    assert.deepEqual(await h.sequencer.gate(false), { generation: 2, requestedAt: 1000 });
});
