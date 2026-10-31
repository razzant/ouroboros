// ws.js's own state reads (the socket-down recovery probe and the post-open read) under a request or body
// that never settles (#2343). Virtual time; the transport ignores cancellation. The served-SHA contract
// itself stays in ws_recovery.test.js.
import assert from 'node:assert/strict';
import test from 'node:test';

import { STATE_READ_LIFETIME_MS as LIFETIME } from '../modules/api_client.js';
import { WS } from '../modules/ws.js';
import { clock as virtualClock, flush } from './state_read_fixture.js';

const clock = (t) => virtualClock(t, 500);

const OPEN = 1;

/** A page without a real socket: scripted /api/state reads (the last repeats) and the reloads they cause. */
function installEnv(t, script) {
    const prior = { WebSocket: globalThis.WebSocket, window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
    const replaced = [];
    const calls = [];
    const queue = [...script];
    globalThis.WebSocket = { OPEN, CONNECTING: 0, CLOSING: 2, CLOSED: 3 };
    globalThis.window = { location: { href: 'http://127.0.0.1:8000/', replace: (url) => replaced.push(String(url)) } };
    globalThis.document = { getElementById: () => null };
    globalThis.fetch = async (url, init = {}) => {
        const entry = queue.length > 1 ? queue.shift() : queue[0];
        calls.push({ url: String(url), signal: init.signal });
        if (entry.hang === 'fetch') return new Promise(() => {});
        if (entry.hang === 'body') return { ok: true, status: 200, json: () => new Promise(() => {}) };
        if (entry.malformed) return { ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected token <'); } };
        const status = entry.status ?? 200;
        return { ok: status < 300, status, json: async () => entry.body ?? {} };
    };
    t.after(() => Object.assign(globalThis, prior));
    return { replaced, calls };
}

function socketPage(t) {
    const ws = new WS('ws://unused');
    t.after(() => {
        ws._clearUiRecoveryTimer();
        ws._clearReconnectTimer();
        ws._clearWatchdogTimer();
        ws._clearPostOpenRetryTimer();
    });
    ws._wasConnected = true;
    ws._lastSha = 'abc';
    return ws;
}

for (const hang of ['fetch', 'body']) {
    test(`a recovery probe whose ${hang} never answers fails at its lifetime and its one chain re-arms`, async (t) => {
        const advance = clock(t);
        const env = installEnv(t, [{ hang }, { body: { sha: 'abc' } }]);
        const ws = socketPage(t);
        ws._scheduleUiRecovery('socket-disconnect', 15000);
        await advance(15000);
        assert.equal(env.calls.length, 1, 'the probe is in flight');
        ws._scheduleUiRecovery('socket-disconnect', 0);  // send() or a reconnect while it hangs: still one chain
        await advance(LIFETIME - 500);
        assert.deepEqual([env.calls.length, ws._uiRecoveryProbeInFlight], [1, true], 'inside its lifetime it still waits');
        await advance(500);
        assert.equal(ws._uiRecoveryProbeInFlight, false, 'the expired probe is a failed probe');
        assert.equal(ws._recoveryHealthyProbes, 0, 'which counts nothing toward the reload fuse');
        await advance(22500);  // the chain's next pause: 15 s x 1.5
        assert.equal(env.calls.length, 2, 'exactly one re-armed probe');
        assert.equal(ws._recoveryHealthyProbes, 1);
        assert.deepEqual(env.replaced, []);
    });
}

test('a recovery probe that expires after its episode ended discards itself', async (t) => {
    const advance = clock(t);
    const env = installEnv(t, [{ hang: 'fetch' }]);
    const ws = socketPage(t);
    ws._scheduleUiRecovery('socket-disconnect', 0);
    await advance(500);
    assert.equal(env.calls.length, 1);
    ws._recoveryEpisode += 1;  // a reconnect ended the episode while the probe hung
    ws._uiRecoveryProbeInFlight = true;  // the new episode's own probe is in flight
    await advance(LIFETIME + 30000);
    assert.equal(ws._uiRecoveryProbeInFlight, true, 'the stale probe neither clears the new flag');
    assert.equal(env.calls.length, 1, 'nor re-arms a chain');
});

for (const hang of ['fetch', 'body']) {
    test(`a post-open read whose ${hang} never answers is a non-answer: repeated, never decided as unknown`, async (t) => {
        const advance = clock(t);
        const env = installEnv(t, [{ hang }, { body: { sha: 'abc' } }]);
        const ws = socketPage(t);
        ws.ws = { readyState: OPEN };
        ws._refreshStateAfterOpen(true);
        await advance(LIFETIME - 500);
        assert.deepEqual([env.calls.length, env.replaced.length], [1, 0]);
        await advance(500 + 2000);  // the lifetime, then the first retry pause
        assert.equal(env.calls.length, 2, 'the expired read is repeated like a failed request');
        assert.deepEqual(env.replaced, [], 'the repeated read keeps the unchanged page; the expired one decided nothing');
        assert.equal(ws._lastSha, 'abc');
    });
}

test('a post-open body that does not parse is still decided as unknown', async (t) => {
    const env = installEnv(t, [{ malformed: true }]);
    const ws = socketPage(t);
    ws.ws = { readyState: OPEN };
    ws._refreshStateAfterOpen(true);
    await flush();
    assert.equal(env.replaced.length, 1);
    assert.match(env.replaced[0], /_ouro_reason=sha-unknown/);
});

test('a post-open read that expires after its socket was replaced decides nothing and is not repeated', async (t) => {
    const advance = clock(t);
    const env = installEnv(t, [{ hang: 'body' }]);
    const ws = socketPage(t);
    ws.ws = { readyState: OPEN };
    ws._refreshStateAfterOpen(true);
    await advance(500);
    ws.ws = { readyState: OPEN };  // a newer socket owns the decision now
    await advance(LIFETIME + 5000);
    assert.deepEqual([env.calls.length, env.replaced.length, ws._postOpenRetryTimer], [1, 0, null]);
});
