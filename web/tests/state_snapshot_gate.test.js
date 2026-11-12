import test from 'node:test';
import assert from 'node:assert/strict';

import { createStateSnapshotSequencer } from '../modules/chat_activity.js';

// The reader below keeps the page readers' shape (chat.js::refreshHeaderControlState,
// app.js::refreshProjectsNav): admission through gate(force), then one read,
// then apply on success or fail on any error. The fake fetch is settled by the
// test so overlap is deterministic; state_read_recovery.test.js runs the real
// shared reader with its lifetime.
function harness() {
    const applied = [];
    const events = [];
    const requests = [];
    const sequencer = createStateSnapshotSequencer(
        (data, requestedAt, generation) => applied.push({ data, requestedAt, generation }),
        () => 1000 + events.length, () => events.push('unavailable'));
    const fetches = [];
    const fetchState = () => new Promise((resolve, reject) => { fetches.push({ resolve, reject }); });
    async function refresh(force = false) {
        const request = await sequencer.gate(force);
        if (!request) return 'skipped';
        requests.push(request);
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
    return { sequencer, applied, events, fetches, requests, refresh };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('unavailable fan-out carries request time and cannot supersede a newer applied census', () => {
    const unavailable = [];
    let now = 100;
    const sequencer = createStateSnapshotSequencer(() => {}, () => now++,
        (...args) => unavailable.push(args));
    const older = sequencer.begin(), newer = sequencer.begin();
    assert.equal(sequencer.apply(newer, { current_questions: ['fresh'] }), true);
    assert.equal(sequencer.fail(older), false);
    assert.deepEqual(unavailable, []);
    assert.deepEqual(sequencer.latest(), { current_questions: ['fresh'] });
    const failed = sequencer.begin();
    assert.equal(sequencer.fail(failed), true);
    assert.deepEqual(unavailable, [[failed.requestedAt, failed.generation]]);
    assert.equal(sequencer.apply(older, { current_questions: [] }), false);
    assert.equal(sequencer.latest(), null);
});

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

test('generation ordering is unchanged: a synthetic bump makes the in-flight read stale and detaches it', async () => {
    const h = harness();
    const inflight = h.refresh(false);
    await tick();
    // app.js does this on socket open/close: an ungated generation bump, not a failed read.
    h.sequencer.retire();
    assert.deepEqual(h.events, [], 'retirement fans out no unavailability');
    assert.equal(h.sequencer.latest(), null, 'the late-mount seed retires');
    assert.equal(h.requests[0].signal.aborted, true, 'the stale read is cancelled');
    const next = h.refresh(false);
    await tick();
    assert.equal(h.fetches.length, 2, 'the retired read no longer holds the gate');
    h.fetches[0].resolve({ ok: true, data: 'older' });
    assert.equal(await inflight, 'applied');
    assert.deepEqual(h.applied, [], 'the older response cannot apply after a newer generation');
    const pending = await Promise.race([next.then(() => 'settled'), tick().then(() => 'pending')]);
    assert.equal(pending, 'pending', 'nor can it settle the newer read');
    h.fetches[1].resolve({ ok: true, data: 'newer' });
    await next;
    assert.deepEqual(h.applied.map((a) => a.data), ['newer']);
});

test('gate requests carry the same clock as begin() and settle through apply of the same object only', async () => {
    const h = harness();
    const a = await h.sequencer.gate(true);
    const { signal, ...fields } = a;
    assert.deepEqual(fields, { generation: 1, requestedAt: 1000, path: '/api/state' });
    assert.equal(signal.aborted, false, 'a gated request carries the live signal of its own read');
    const pending = () => Promise.race([h.sequencer.gate(false).then(() => 'settled'), tick().then(() => 'pending')]);
    assert.equal(await pending(), 'pending');
    // A different object with the same generation does not settle the gate.
    assert.equal(h.sequencer.apply({ ...a }, 'copy'), true);
    assert.equal(await pending(), 'pending');
    assert.equal(h.sequencer.apply(a, 'same'), false);
    // Settled: the next tick opens a fresh read instead of joining.
    const { signal: nextSignal, ...next } = await h.sequencer.gate(false);
    assert.deepEqual(next, { generation: 2, requestedAt: 1000, path: '/api/state' });
    assert.notEqual(nextSignal, signal, 'each read has its own signal');
});

test('only an actual failed read fans out, and a throwing fan-out still releases the gate', async () => {
    let throwing = true;
    const unavailable = [];
    const sequencer = createStateSnapshotSequencer(() => {}, () => 7, (...args) => {
        unavailable.push(args);
        if (throwing) throw new Error('one window failed to paint');
    });
    const retired = await sequencer.gate(false);
    sequencer.retire();
    assert.equal(sequencer.fail(retired), false, 'a retired read cannot report unavailability');
    assert.deepEqual(unavailable, []);
    const failed = await sequencer.gate(false);
    assert.throws(() => sequencer.fail(failed), /failed to paint/);
    assert.deepEqual(unavailable, [[7, failed.generation]]);
    throwing = false;
    const next = await Promise.race([sequencer.gate(false), tick().then(() => 'stuck')]);
    assert.notEqual(next, 'stuck', 'the failed request settled despite the throwing fan-out');
    assert.equal(sequencer.fail(next), true);
    assert.equal(unavailable.length, 2);
});

// The bindings map is held with its revision; `?bindings_since` lets the server answer "unchanged".
const bound = { t1: { project_id: 'alpha', chat_id: 5, project_name: 'Alpha' } };

test('a held bindings map fills an unchanged body for its own revision only', () => {
    const applied = [];
    const sequencer = createStateSnapshotSequencer((data) => applied.push(data));
    const first = sequencer.begin();
    assert.equal(first.path, '/api/state');
    sequencer.apply(first, { task_bindings: bound, task_bindings_revision: 'r1', task_bindings_unchanged: false });
    const second = sequencer.begin();
    assert.equal(second.path, '/api/state?bindings_since=r1');
    sequencer.apply(second, { task_bindings: null, task_bindings_revision: 'r1', task_bindings_unchanged: true });
    assert.deepEqual(applied[1].task_bindings, bound);
    assert.deepEqual(sequencer.latest().task_bindings, bound, 'the late-mount seed carries the filled map');
    const third = sequencer.begin();
    sequencer.apply(third, { task_bindings: null, task_bindings_revision: 'r2', task_bindings_unchanged: true });
    assert.equal(applied[2].task_bindings, null, 'a revision this page never held is never filled');
    assert.equal(sequencer.begin().path, '/api/state', 'and the next read asks for the whole map');
});

test('an old server, a failed bindings read or a retire hold nothing', () => {
    const sequencer = createStateSnapshotSequencer(() => {});
    sequencer.apply(sequencer.begin(), { task_bindings: bound });
    assert.equal(sequencer.begin().path, '/api/state');
    sequencer.apply(sequencer.begin(), { task_bindings: {}, task_bindings_revision: '', task_bindings_unchanged: false });
    assert.equal(sequencer.begin().path, '/api/state');
    sequencer.apply(sequencer.begin(), { task_bindings: bound, task_bindings_revision: 'r1' });
    sequencer.retire();
    assert.equal(sequencer.begin().path, '/api/state', 'a socket change or a confirmed mutation reads the map whole');
});

test('an older answer can neither fill nor replace the held map', () => {
    const applied = [];
    const sequencer = createStateSnapshotSequencer((data) => applied.push(data));
    const older = sequencer.begin();
    const newer = sequencer.begin();
    assert.equal(sequencer.apply(newer, { task_bindings: bound, task_bindings_revision: 'r2' }), true);
    assert.equal(sequencer.apply(older, { task_bindings: {}, task_bindings_revision: 'r1' }), false);
    assert.equal(sequencer.begin().path, '/api/state?bindings_since=r2');
    assert.equal(applied.length, 1);
});
