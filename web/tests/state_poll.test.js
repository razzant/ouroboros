import assert from 'node:assert/strict';
import test from 'node:test';

import {
    STATE_POLL_CHAT_MS,
    STATE_POLL_IDLE_MS,
    createStatePoll,
    statePollIntervalMs,
} from '../modules/state_poll.js';

/**
 * A deterministic stand-in for setTimeout/clearTimeout: nothing fires until the test
 * says so, so "the timer re-armed" is an assertion rather than a wait.
 */
function fakeClock() {
    const pending = new Map();
    let nextId = 1;
    return {
        armed: [],
        cleared: [],
        setTimer(fn, ms) {
            const id = nextId++;
            pending.set(id, fn);
            this.armed.push({ id, ms });
            return id;
        },
        clearTimer(id) {
            this.cleared.push(id);
            pending.delete(id);
        },
        get pendingCount() { return pending.size; },
        /** Fire the most recently armed timer, as a real event loop eventually would. */
        async fire() {
            const last = this.armed[this.armed.length - 1];
            const fn = pending.get(last.id);
            assert.ok(fn, `timer ${last.id} is not pending`);
            pending.delete(last.id);
            await fn();
        },
        lastMs() { return this.armed.length ? this.armed[this.armed.length - 1].ms : null; },
    };
}

function harness({ page = 'chat', hidden = false, read } = {}) {
    const clock = fakeClock();
    const env = { page, hidden, reads: 0 };
    const poll = createStatePoll({
        read: read || (async () => { env.reads += 1; return { tick: env.reads }; }),
        activePage: () => env.page,
        hidden: () => env.hidden,
        setTimer: (fn, ms) => clock.setTimer(fn, ms),
        clearTimer: (id) => clock.clearTimer(id),
    });
    return { poll, clock, env };
}

test('the cadence is Chat-vs-elsewhere, read at decision time', () => {
    assert.equal(statePollIntervalMs('chat'), STATE_POLL_CHAT_MS);
    assert.equal(statePollIntervalMs('files'), STATE_POLL_IDLE_MS);
    assert.equal(statePollIntervalMs('changes'), STATE_POLL_IDLE_MS);
    assert.equal(statePollIntervalMs(undefined), STATE_POLL_IDLE_MS);
    // 3s on Chat is live budget/mode feedback; 20s is the resting rate.
    assert.ok(STATE_POLL_CHAT_MS < STATE_POLL_IDLE_MS);

    // The page is a GETTER, so navigating changes the NEXT arming, not a value
    // captured when the poll was built.
    const { poll, clock, env } = harness({ page: 'chat' });
    poll.schedule();
    assert.equal(clock.lastMs(), STATE_POLL_CHAT_MS);
    env.page = 'dashboard';
    poll.schedule();
    assert.equal(clock.lastMs(), STATE_POLL_IDLE_MS);
    assert.equal(poll.intervalMs(), STATE_POLL_IDLE_MS);
});

test('concurrent callers are ONE request, not three', async () => {
    let resolveRead;
    let calls = 0;
    const { poll, clock } = harness({
        read: () => {
            calls += 1;
            return new Promise((res) => { resolveRead = () => res({ ok: true }); });
        },
    });

    // The startup barrier, a projects_changed refresh and a timer tick, together.
    const a = poll.refresh();
    const b = poll.refresh();
    const c = poll.refresh();
    assert.equal(calls, 1, 'a second caller must join the in-flight read');
    assert.equal(a, b);
    assert.equal(b, c);

    resolveRead();
    await Promise.all([a, b, c]);
    assert.equal(calls, 1);

    // Once it SETTLED the guard is released, so a later caller really does read.
    const seen = [];
    poll.subscribe((data) => seen.push(data));
    const d = poll.refresh();
    assert.equal(calls, 2);
    resolveRead();
    await d;
    assert.equal(clock.pendingCount, 1, 'the timer is re-armed after each settle');
});

test('the timer re-arms on SETTLE, so a slow read cannot stack requests', async () => {
    const { poll, clock, env } = harness();
    await poll.refresh();
    assert.equal(env.reads, 1);
    assert.equal(clock.armed.length, 1, 'armed once, after the read settled');

    // Firing the timer performs the next read, which arms the next timer: one live
    // timer at a time, forever.
    await clock.fire();
    assert.equal(env.reads, 2);
    assert.equal(clock.pendingCount, 1);
    await clock.fire();
    assert.equal(env.reads, 3);
    assert.equal(clock.pendingCount, 1);
});

test('a REJECTED read still re-arms, and never wedges the guard', async () => {
    let calls = 0;
    const { poll, clock } = harness({
        read: async () => { calls += 1; throw new Error('network down'); },
    });
    await assert.rejects(poll.refresh(), /network down/);
    assert.equal(clock.pendingCount, 1, 'a failed read must not stop the poll');
    // The in-flight guard was released, so the app is not permanently stuck.
    await assert.rejects(poll.refresh(), /network down/);
    assert.equal(calls, 2);
});

test('a hidden document PAUSES the timer instead of backing it off', async () => {
    const { poll, clock, env } = harness();
    env.hidden = true;
    poll.schedule();
    assert.equal(clock.pendingCount, 0, 'no hidden-tab spend at any interval');
    // Not a longer interval — no timer at all.
    assert.equal(clock.armed.length, 0);

    // A read while hidden still publishes (an explicit refresh is honoured), but
    // arms nothing behind it.
    await poll.refresh();
    assert.equal(env.reads, 1);
    assert.equal(clock.pendingCount, 0);

    env.hidden = false;
    poll.schedule();
    assert.equal(clock.pendingCount, 1);
    poll.stop();
    assert.equal(clock.pendingCount, 0);
});

test('the visibilitychange sequence a live tab performs: armed -> stop -> re-arm', async () => {
    // The exact shape of app.js's `visibilitychange` handler. It matters as its own
    // case because the handler used to reach for a module-scope timer variable that
    // consolidation deleted: `stop()` is the seam the core exports, and it has to
    // clear a timer that is ALREADY ARMED (hiding a tab mid-interval), not just
    // decline to arm a new one.
    const { poll, clock, env } = harness({ page: 'chat' });
    poll.schedule();
    const armed = clock.armed[clock.armed.length - 1];
    assert.equal(clock.pendingCount, 1);
    assert.equal(armed.ms, STATE_POLL_CHAT_MS);

    // Tab hidden: the live handle is cleared, so the pending tick never fires.
    env.hidden = true;
    poll.stop();
    assert.deepEqual(clock.cleared.includes(armed.id), true, 'the ARMED handle was cleared');
    assert.equal(clock.pendingCount, 0);
    assert.equal(env.reads, 0, 'a hidden tab spends nothing');

    // Idempotent: a second hide event (or a stop with no timer) is a harmless noop.
    poll.stop();
    assert.equal(clock.pendingCount, 0);

    // Tab visible again: the catch-up read re-arms on SETTLE, exactly as the
    // handler's `refreshState()` relies on.
    env.hidden = false;
    await poll.refresh();
    assert.equal(env.reads, 1);
    assert.equal(clock.pendingCount, 1);
    assert.equal(clock.lastMs(), STATE_POLL_CHAT_MS);
});

test('every subscriber sees the same snapshot, and a late one is replayed', async () => {
    const { poll } = harness();
    const early = [];
    poll.subscribe((data) => early.push(data));
    await poll.refresh();
    assert.deepEqual(early, [{ tick: 1 }]);

    // A project panel created after startup must not wait a full interval for its
    // first paint.
    const late = [];
    poll.subscribe((data) => late.push(data));
    assert.deepEqual(late, [{ tick: 1 }], 'late subscriber replayed the last snapshot');

    await poll.refresh();
    assert.deepEqual(early, [{ tick: 1 }, { tick: 2 }]);
    assert.deepEqual(late, [{ tick: 1 }, { tick: 2 }]);
});

test('unsubscribe really stops delivery, and is idempotent', async () => {
    const { poll } = harness();
    const seen = [];
    const off = poll.subscribe((data) => seen.push(data));
    await poll.refresh();
    assert.equal(seen.length, 1);
    off();
    await poll.refresh();
    assert.equal(seen.length, 1, 'no delivery after unsubscribe');
    off();  // second call must not throw or remove someone else's handler
    const other = [];
    poll.subscribe((data) => other.push(data));
    await poll.refresh();
    assert.equal(seen.length, 1);
    assert.equal(other.length, 2, 'replay + the fresh snapshot');

    // A non-function is refused with a no-op unsubscribe rather than an exception.
    assert.equal(typeof poll.subscribe(null), 'function');
    poll.subscribe(null)();
});

test('one throwing subscriber cannot starve the others or the timer', async () => {
    const { poll, clock } = harness();
    const after = [];
    poll.subscribe(() => { throw new Error('bad consumer'); });
    poll.subscribe((data) => after.push(data));
    await poll.refresh();
    assert.deepEqual(after, [{ tick: 1 }], 'the next consumer still got the snapshot');
    assert.equal(clock.pendingCount, 1, 'and the poll is still alive');

    // The same containment applies to the late-subscriber replay.
    assert.doesNotThrow(() => poll.subscribe(() => { throw new Error('bad late'); }));
});

test('a subscriber that unsubscribes DURING fan-out does not skip its neighbour', async () => {
    const { poll } = harness();
    const order = [];
    const off = poll.subscribe(() => { order.push('first'); off(); });
    poll.subscribe(() => order.push('second'));
    await poll.refresh();
    // Iterating a live array while splicing it would have skipped 'second'.
    assert.deepEqual(order, ['first', 'second']);
});
