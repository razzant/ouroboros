import assert from 'node:assert/strict';
import test from 'node:test';

import { WS, decide, decideServedPage, pageOwnedByCore, RECOVERY_HEALTHY_PROBE_LIMIT, SERVER_ANSWERS_EVENT } from '../modules/ws.js';

// Behavior coverage for the served-SHA reconciliation contract (netres Lane B):
// one SSOT decision drives both the post-open refresh and the socket-down
// recovery probe. These tests run the real WS code under a mocked browser
// environment (same pattern as client_surface.test.js).

class FakeSocket {
    static OPEN = 1;
    static CONNECTING = 0;
    static CLOSING = 2;
    static CLOSED = 3;
    static instances = [];

    constructor(url) {
        this.url = url;
        this.readyState = FakeSocket.CONNECTING;
        this.sent = [];
        FakeSocket.instances.push(this);
    }

    send(data) { this.sent.push(data); }

    close() {}
}

function installEnv(responses = []) {
    const replaced = [];
    let calls = 0;
    globalThis.WebSocket = FakeSocket;
    FakeSocket.instances = [];
    globalThis.window = {
        location: {
            href: 'http://127.0.0.1:8000/',
            replace: (url) => replaced.push(String(url)),
        },
    };
    globalThis.document = { getElementById: () => null };
    globalThis.fetch = async () => {
        calls += 1;
        // The last scripted response repeats for every later probe.
        const script = responses.length > 1 ? responses.shift() : responses[0];
        if (!script || script.reject) throw new Error('network down');
        return {
            ok: script.ok !== false,
            status: script.status ?? (script.ok === false ? 500 : 200),
            json: async () => {
                if (script.badJson) throw new Error('malformed body');
                return script.body ?? {};
            },
        };
    };
    return {
        replaced,
        fetchCalls: () => calls,
        setFetch: (fn) => { globalThis.fetch = async (...args) => { calls += 1; return fn(...args); }; },
    };
}

function settle(turns = 4) {
    let p = Promise.resolve();
    for (let i = 0; i < turns; i += 1) {
        p = p.then(() => new Promise((resolve) => setTimeout(resolve, 0)));
    }
    return p;
}

async function waitFor(cond, what, maxTurns = 400) {
    for (let i = 0; i < maxTurns; i += 1) {
        if (cond()) return;
        await settle(1);
    }
    throw new Error(`condition not reached: ${what}`);
}

// Stops any chained delay-0 recovery timers so node:test can exit: an OPEN
// socket makes an in-flight probe bail without re-arming, then clear timers.
async function teardown(ws) {
    ws.ws = { readyState: FakeSocket.OPEN };
    await settle(6);
    ws._clearUiRecoveryTimer();
    ws._clearReconnectTimer();
    ws._clearWatchdogTimer();
    ws._clearPostOpenRetryTimer();
    ws.ws = null;
}

// ---------------------------------------------------------------------------
// decide(): the pure SSOT contract.
// ---------------------------------------------------------------------------

test('decide keeps the page on the first-ever connection regardless of SHA state', () => {
    assert.equal(decide(null, 'abc', false), 'keep');
    assert.equal(decide(null, undefined, false), 'keep');
    assert.equal(decide('abc', 'def', false), 'keep');
});

test('decide keeps the page when the served SHA is unchanged', () => {
    assert.equal(decide('abc', 'abc', true), 'keep');
    assert.equal(decide(' abc ', 'abc', true), 'keep');
});

test('decide reloads as changed when the served SHA differs', () => {
    assert.equal(decide('abc', 'def', true), 'reload_changed');
});

test('decide reloads as unknown when a previously-known SHA disappears or garbles', () => {
    assert.equal(decide('abc', undefined, true), 'reload_unknown');
    assert.equal(decide('abc', null, true), 'reload_unknown');
    assert.equal(decide('abc', '', true), 'reload_unknown');
    assert.equal(decide('abc', '   ', true), 'reload_unknown');
    assert.equal(decide('abc', 12345, true), 'reload_unknown');
    assert.equal(decide('abc', { sha: 'abc' }, true), 'reload_unknown');
    // An unproveable PREVIOUS SHA equally forbids the keep claim once the
    // server does serve one.
    assert.equal(decide(null, 'abc', true), 'reload_unknown');
    assert.equal(decide('', 'abc', true), 'reload_unknown');
});

test('decide keeps ONLY on an exact empty-string served SHA with nothing remembered (Q19)', () => {
    // Owner-selected default under uncertainty (netres Q19), narrowed: the
    // server explicitly serves sha="" (unversioned /api/state while
    // current_sha is unset) and no non-empty SHA was ever remembered.
    assert.equal(decide(null, '', true), 'keep');
    assert.equal(decide('', '', true), 'keep');
    assert.equal(decide(undefined, '', true), 'keep');
});

test('decide reloads as unknown for absent/malformed served values even with nothing remembered', () => {
    // Only the exact empty string is the unversioned-install fact; a missing
    // field, parse failure, non-string or whitespace-only value after a
    // previous connection cannot prove the page current.
    assert.equal(decide(null, undefined, true), 'reload_unknown');
    assert.equal(decide(null, null, true), 'reload_unknown');
    assert.equal(decide('', '   ', true), 'reload_unknown');
    assert.equal(decide(undefined, null, true), 'reload_unknown');
    assert.equal(decide(null, 12345, true), 'reload_unknown');
});

// ---------------------------------------------------------------------------
// _refreshStateAfterOpen: post-open reconciliation through the same SSOT.
// ---------------------------------------------------------------------------

test('first open remembers the served SHA without reloading; later opens compare against it', async () => {
    const env = installEnv([{ body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._refreshStateAfterOpen(false);
        await settle();
        assert.equal(env.replaced.length, 0);

        // Same SHA after a reconnect: keep.
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 0);

        // Changed SHA after a reconnect: reload (proves the first open stored 'abc').
        env.setFetch(async () => ({ ok: true, json: async () => ({ sha: 'def' }) }));
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-change/);
    } finally {
        await teardown(ws);
    }
});

test('an OK state response without a SHA after a reconnect reloads (the closed keep-hole)', async () => {
    const env = installEnv([{ body: {} }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-unknown/);
    } finally {
        await teardown(ws);
    }
});

test('a malformed state body after a reconnect reloads as unknown', async () => {
    const env = installEnv([{ badJson: true }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-unknown/);
    } finally {
        await teardown(ws);
    }
});

test('a non-OK state response after open never reloads', async () => {
    const env = installEnv([{ ok: false, body: {} }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 0);
    } finally {
        await teardown(ws);
    }
});

// A non-answer is no "keep": the comparison a reconnect owes is repeated until the
// server answers or the socket is gone. Delay 0 chains the retries for the test.
test('a non-answer after open is retried until the server answers; a changed SHA then reloads once', async () => {
    const env = installEnv([{ ok: false, body: {} }, { reject: true }, { body: { sha: 'def' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true, 0);
        await waitFor(() => env.replaced.length === 1, 'reload after the third read');
        assert.match(env.replaced[0], /_ouro_reason=sha-change/);
        assert.equal(env.fetchCalls(), 3, 'a 500, then a failed request, then the answer');
        await settle(6);
        assert.equal(env.fetchCalls(), 3, 'the answer ends the retries');
        assert.equal(env.replaced.length, 1);
    } finally {
        await teardown(ws);
    }
});

test('a retried post-open read that finds the same SHA keeps the page and stops', async () => {
    const env = installEnv([{ ok: false, body: {} }, { body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true, 0);
        await waitFor(() => env.fetchCalls() === 2, 'second read');
        await settle(6);
        assert.equal(env.fetchCalls(), 2);
        assert.equal(env.replaced.length, 0);
        assert.equal(ws._postOpenRetryTimer, null);
    } finally {
        await teardown(ws);
    }
});

test('a first-open non-answer is retried and stores the baseline without reloading', async () => {
    const env = installEnv([{ ok: false, body: {} }, { body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._refreshStateAfterOpen(false, 0);
        await waitFor(() => ws._lastSha === 'abc', 'baseline stored by the retry');
        assert.equal(env.replaced.length, 0);
    } finally {
        await teardown(ws);
    }
});

test('post-open retries belong to their socket: a cycled connection ends them', async () => {
    const env = installEnv([{ ok: false, body: {} }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws.ws = { readyState: FakeSocket.OPEN };
        ws._refreshStateAfterOpen(true, 0);
        await waitFor(() => env.fetchCalls() >= 3, 'retries chain while the socket lives');
        ws.ws = { readyState: FakeSocket.OPEN };
        await settle(6);
        const calls = env.fetchCalls();
        await settle(6);
        assert.equal(env.fetchCalls(), calls, 'the old socket no longer reads');
        assert.equal(env.replaced.length, 0);
    } finally {
        await teardown(ws);
    }
});

test('a disconnect clears the pending post-open retry', async () => {
    const env = installEnv([{ ok: false, body: {} }]);
    const ws = new WS('ws://unused');
    try {
        ws.connect();
        const socket = FakeSocket.instances[0];
        socket.readyState = FakeSocket.OPEN;
        socket.onopen();
        await waitFor(() => ws._postOpenRetryTimer !== null, 'retry armed after the 500');
        socket.onclose();
        assert.equal(ws._postOpenRetryTimer, null);
        assert.equal(env.replaced.length, 0);
    } finally {
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// _scheduleUiRecovery: probe decisions, fuse, race bail, queue survival.
// ---------------------------------------------------------------------------

test('healthy same-SHA probes reconnect in place; the fuse reloads exactly on the Nth probe', async () => {
    const env = installEnv([{ body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'fuse reload');
        // The fuse fired on the Nth probe — earlier healthy probes kept the page.
        assert.equal(env.fetchCalls(), RECOVERY_HEALTHY_PROBE_LIMIT);
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/);
    } finally {
        await teardown(ws);
    }
});

test('a failed probe resets the consecutive healthy count', async () => {
    const env = installEnv([
        { body: { sha: 'abc' } },
        { body: { sha: 'abc' } },
        { ok: false },
        { body: { sha: 'abc' } },
    ]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'fuse reload after reset');
        // Two healthy, one failed (reset), then a fresh full run of healthy probes.
        assert.equal(env.fetchCalls(), 3 + RECOVERY_HEALTHY_PROBE_LIMIT);
    } finally {
        await teardown(ws);
    }
});

test('a recovery probe seeing a changed SHA reloads immediately', async () => {
    const env = installEnv([{ body: { sha: 'def' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'sha-change reload');
        assert.equal(env.fetchCalls(), 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-change/);
    } finally {
        await teardown(ws);
    }
});

test('a recovery probe with a missing SHA after a connection reloads as unknown', async () => {
    const env = installEnv([{ body: {} }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'sha-unknown reload');
        assert.equal(env.fetchCalls(), 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-unknown/);
    } finally {
        await teardown(ws);
    }
});

test('a probe resolving after the socket reopened bails without reloading or re-arming', async () => {
    const env = installEnv([]);
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    env.setFetch(async () => {
        await gate;
        // A changed SHA that would reload were the bail missing.
        return { ok: true, json: async () => ({ sha: 'zzz' }) };
    });
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() === 1, 'probe dispatched');
        ws.ws = { readyState: FakeSocket.OPEN };
        release();
        await settle();
        assert.equal(env.replaced.length, 0);
        await settle();
        assert.equal(env.fetchCalls(), 1, 'a bailed probe must not re-arm recovery');
    } finally {
        await teardown(ws);
    }
});

test('the fuse fires at most once per disconnect episode and resets on reconnect', async () => {
    const env = installEnv([{ body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';

        // Episode 1: fuse fires once.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'first fuse');

        // Still down, recovery re-armed by the reconnect cycle: healthy probes
        // continue but the episode fuse stays latched.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        const seen = env.fetchCalls();
        await waitFor(
            () => env.fetchCalls() >= seen + RECOVERY_HEALTHY_PROBE_LIMIT + 2,
            'post-fuse probes',
        );
        assert.equal(env.replaced.length, 1, 'the fuse must not fire twice in one episode');

        // Successful reconnect ends the episode.
        ws.ws = { readyState: FakeSocket.OPEN };
        await settle(6);
        ws.ws = null;
        ws.connect();
        const sock = FakeSocket.instances.at(-1);
        sock.readyState = FakeSocket.OPEN;
        sock.onopen();
        await settle();
        assert.equal(env.replaced.length, 1, 'a same-SHA reconnect must not reload');

        // Episode 2 after a fresh disconnect: the fuse is armed again AND the
        // healthy counter restarted from zero — the episode must consume a
        // full run of healthy probes before its own fuse (pins the onopen
        // counter reset).
        const episode2Base = env.fetchCalls();
        ws.ws = null;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 2, 'second-episode fuse');
        assert.ok(
            env.fetchCalls() - episode2Base >= RECOVERY_HEALTHY_PROBE_LIMIT,
            'episode 2 fired its fuse without a full run of healthy probes',
        );
    } finally {
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// In-flight probe gating: at most one recovery probe chain per episode.
// ---------------------------------------------------------------------------

test('arm attempts while a probe hangs in flight dispatch no extra probe', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => {
        releases.push(() => resolve({ ok: true, json: async () => ({ sha: 'abc' }) }));
    }));
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() === 1, 'first probe dispatched');
        // send()/_scheduleReconnect would re-arm during the hung probe: gated.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await settle(8);
        assert.equal(env.fetchCalls(), 1, 'no second probe while one is in flight');
        releases.shift()();
        // The single chain resumes with exactly one follow-up probe.
        await waitFor(() => env.fetchCalls() === 2, 'chain resumed after resolution');
        await settle(4);
        assert.equal(env.fetchCalls(), 2, 'still one chain after the resolution');
        assert.equal(env.replaced.length, 0);
    } finally {
        while (releases.length) releases.shift()();
        await teardown(ws);
    }
});

test('hung probes cannot multi-count the healthy fuse', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => {
        releases.push(() => resolve({ ok: true, json: async () => ({ sha: 'abc' }) }));
    }));
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        for (let i = 0; i < RECOVERY_HEALTHY_PROBE_LIMIT; i += 1) {
            await waitFor(() => env.fetchCalls() === i + 1, `probe ${i + 1} dispatched`);
            // Spam arm attempts while the probe hangs — none may stack.
            ws._scheduleUiRecovery('socket-disconnect', 0);
            ws._scheduleUiRecovery('socket-disconnect', 0);
            assert.equal(env.fetchCalls(), i + 1);
            releases.shift()();
            await settle(4);
        }
        // The fuse needed LIMIT distinct sequential probes — one healthy count
        // per physical probe — and fired exactly once.
        assert.equal(env.replaced.length, 1, 'fuse fired exactly once');
        assert.equal(env.fetchCalls(), RECOVERY_HEALTHY_PROBE_LIMIT);
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/);
    } finally {
        while (releases.length) releases.shift()();
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Run-from-source installs (served sha "") and probe-side SHA adoption.
// ---------------------------------------------------------------------------

test('a source install (served sha "") keeps the page on reconnect, post-open path', async () => {
    const env = installEnv([{ body: { sha: '' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 0, 'no SHA was ever provable: keep');
        assert.equal(ws._lastSha, null, 'an empty served SHA is never stored');
    } finally {
        await teardown(ws);
    }
});

test('a source install keeps on recovery probes and the fuse still heals a frozen runtime', async () => {
    // Interplay B2+B3: never-any-sha probes keep (no reload_unknown storm) and
    // store nothing; the healthy-probe fuse remains the one reload that heals
    // a frozen WebSocket runtime.
    const env = installEnv([{ body: { sha: '' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'fuse reload');
        assert.equal(env.fetchCalls(), RECOVERY_HEALTHY_PROBE_LIMIT,
            'every pre-fuse probe kept the page');
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/,
            'the reload is the runtime fuse, not a SHA decision');
        assert.equal(ws._lastSha, null);
    } finally {
        await teardown(ws);
    }
});

test('known-SHA-then-empty still reloads as unknown on the recovery probe', async () => {
    const env = installEnv([{ body: { sha: '' } }]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'sha-unknown reload');
        assert.equal(env.fetchCalls(), 1);
        assert.match(env.replaced[0], /_ouro_reason=sha-unknown/);
    } finally {
        await teardown(ws);
    }
});

test('a recovery probe never stores the served SHA; only the post-open refresh does', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => {
        releases.push(() => resolve({ ok: true, json: async () => ({ sha: 'X' }) }));
    }));
    const ws = new WS('ws://unused');
    try {
        // Never-connected client probing a restarted server: keep, store nothing.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() === 1, 'pre-connect probe dispatched');
        releases.shift()();
        await settle();
        assert.equal(env.replaced.length, 0);
        assert.equal(ws._lastSha, null, 'the probe must not adopt the served SHA');

        // Park the probe chain, then let the first open remember the SHA.
        ws.ws = { readyState: FakeSocket.OPEN };
        await settle(6);
        ws._clearUiRecoveryTimer();
        while (releases.length) releases.shift()();
        env.setFetch(async () => ({ ok: true, json: async () => ({ sha: 'X' }) }));
        ws._refreshStateAfterOpen(false);
        await settle();
        assert.equal(env.replaced.length, 0);
        assert.equal(ws._lastSha, 'X', 'the post-open refresh stores the served SHA');

        // A later probe seeing different assets heals the page (as base did).
        ws._wasConnected = true;
        ws.ws = null;
        env.setFetch(async () => ({ ok: true, json: async () => ({ sha: 'Y' }) }));
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'changed-SHA reload');
        assert.match(env.replaced[0], /_ouro_reason=sha-change/);
    } finally {
        while (releases.length) releases.shift()();
        await teardown(ws);
    }
});

test('keep-recovery preserves the outbound queue and the reconnect flush delivers it', async () => {
    const env = installEnv([
        { body: { sha: 'abc' } },
        { ok: false },
    ]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        const outbound = [];
        ws.on('outbound_queued', (e) => outbound.push(['queued', e.clientMessageId]));
        ws.on('outbound_sent', (e) => outbound.push(['sent', e.clientMessageId, e.queued]));

        const result = ws.send({ type: 'chat', text: 'hello' });
        assert.equal(result.status, 'queued');
        // send() armed the slow default timers; drive recovery deterministically.
        ws._clearUiRecoveryTimer();
        ws._clearReconnectTimer();
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() >= 2, 'keep probe plus follow-up');
        assert.equal(env.replaced.length, 0, 'a healthy same-SHA probe must not reload');

        const sock = ws.ws;
        sock.readyState = FakeSocket.OPEN;
        sock.onopen();
        await settle();
        assert.equal(env.replaced.length, 0);
        assert.equal(sock.sent.length, 1, 'the queued message must flush on reconnect');
        const frame = JSON.parse(sock.sent[0]);
        assert.equal(frame.type, 'chat');
        assert.equal(frame.text, 'hello');
        assert.deepEqual(outbound[0], ['queued', result.clientMessageId]);
        assert.deepEqual(outbound[1], ['sent', result.clientMessageId, true]);
    } finally {
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Disconnect-episode generations: a probe belongs to the episode it was armed
// in; a reconnect ends the episode and stale resolutions are discarded.
// ---------------------------------------------------------------------------

test('a probe hung across reconnect + second disconnect neither disarms the new episode nor acts stale', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => { releases.push(resolve); }));
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';

        // Episode 1: probe A dispatches and hangs.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() === 1, 'probe A dispatched');

        // Reconnect while A is still in flight (its post-open refresh fetch
        // also hangs — call 2).
        ws.connect();
        const sock = FakeSocket.instances.at(-1);
        sock.readyState = FakeSocket.OPEN;
        sock.onopen();
        await settle();

        // Second disconnect: the NEW episode must be able to arm its own
        // probe (with a global in-flight flag it would stay disarmed forever
        // while A hangs).
        ws.ws = null;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() === 3, 'new episode armed its own probe');

        // Late resolution of stale probe A: a changed SHA that would reload
        // were the generation guard missing, with the healthy counter one
        // short of the fuse so a stale healthy count would fire it.
        ws._recoveryHealthyProbes = RECOVERY_HEALTHY_PROBE_LIMIT - 1;
        releases[0]({ ok: true, json: async () => ({ sha: 'zzz' }) });
        await settle(6);
        assert.equal(env.replaced.length, 0, 'a stale probe must not reload or fire the fuse');
        assert.equal(ws._recoveryHealthyProbes, RECOVERY_HEALTHY_PROBE_LIMIT - 1,
            'a stale probe must not touch the new episode\'s counters');
        assert.equal(ws._uiRecoveryProbeInFlight, true,
            'a stale probe must not clear the newer episode\'s in-flight flag');

        // The stale resolution must not have re-armed a second chain either.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await settle(4);
        assert.equal(env.fetchCalls(), 3, 'no extra probe after the stale resolution');

        // The live probe B still owns the episode: releasing it healthy
        // same-SHA reaches the fuse threshold set above.
        releases[2]({ ok: true, json: async () => ({ sha: 'abc' }) });
        await waitFor(() => env.replaced.length === 1, 'live probe reaches the fuse');
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/);
    } finally {
        releases.forEach((release) => release({ ok: false, json: async () => ({}) }));
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Captive-portal probe responses: 200 without a parseable object body is a
// FAILED probe, never health.
// ---------------------------------------------------------------------------

test('a 200 non-JSON probe body counts as a failed probe: reset and re-arm, no reload', async () => {
    // A captive portal / interposed proxy answers 200 HTML for a remote
    // browser: with a remembered SHA the old code reloaded straight into the
    // portal, destroying queued outbound messages.
    const env = installEnv([
        { body: { sha: 'abc' } },
        { body: { sha: 'abc' } },
        { badJson: true },
        { body: { sha: 'abc' } },
    ]);
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'fuse reload after the portal reset');
        // Two healthy, the portal response (failed → counter reset), then a
        // fresh full run of healthy probes before the fuse. Had the portal
        // response counted as healthy, the reload would have come earlier and
        // with a sha-unknown reason.
        assert.equal(env.fetchCalls(), 3 + RECOVERY_HEALTHY_PROBE_LIMIT);
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/,
            'the only reload is the runtime fuse, never a portal-driven SHA decision');
    } finally {
        await teardown(ws);
    }
});

test('a 200 probe whose JSON body is not an object is equally a failed probe', async () => {
    const env = installEnv([]);
    env.setFetch(async () => ({ ok: true, json: async () => 'signin required' }));
    const ws = new WS('ws://unused');
    try {
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() >= RECOVERY_HEALTHY_PROBE_LIMIT + 2,
            'probe chain keeps re-arming');
        assert.equal(env.replaced.length, 0, 'non-object bodies must never reload');
        assert.equal(ws._recoveryHealthyProbes, 0, 'failed probes keep the counter at zero');
    } finally {
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Post-open refresh socket scoping: a stale fetch must not overwrite _lastSha
// or reload after the connection cycled (mirror of the probe's OPEN bail).
// ---------------------------------------------------------------------------

test('a post-open refresh resolving after the connection cycled neither stores nor reloads', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => { releases.push(resolve); }));
    const ws = new WS('ws://unused');
    try {
        // First-open refresh dispatches against socket 1 and hangs.
        const socket1 = { readyState: FakeSocket.OPEN };
        ws.ws = socket1;
        ws._refreshStateAfterOpen(false);
        await waitFor(() => env.fetchCalls() === 1, 'refresh dispatched');

        // The connection cycles: a newer socket owns reconciliation now.
        ws.ws = { readyState: FakeSocket.OPEN };
        releases[0]({ ok: true, json: async () => ({ sha: 'stale' }) });
        await settle();
        assert.equal(ws._lastSha, null, 'a stale refresh must not adopt the served SHA');
        assert.equal(env.replaced.length, 0);

        // Same for a reconnect refresh that would otherwise reload.
        ws._wasConnected = true;
        ws._lastSha = 'abc';
        const socket3 = ws.ws;
        ws._refreshStateAfterOpen(true);
        await waitFor(() => env.fetchCalls() === 2, 'reconnect refresh dispatched');
        assert.equal(ws.ws, socket3);
        ws.ws = { readyState: FakeSocket.OPEN };
        releases[1]({ ok: true, json: async () => ({ sha: 'zzz' }) });
        await settle();
        assert.equal(env.replaced.length, 0, 'a stale refresh must not reload the page');
        assert.equal(ws._lastSha, 'abc');
    } finally {
        releases.forEach((release) => release({ ok: false, json: async () => ({}) }));
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Page ownership: who refreshes this page when the core behind it changes.
// `served_ui` is the interface the core serves; the marker is the core's
// per-response stamp on its own built-in page.
// ---------------------------------------------------------------------------

const SHA_CASES = [[null, 'abc', false], ['abc', 'abc', true], ['abc', 'def', true], ['abc', undefined, true],
    ['abc', '', true], [null, '', true], [null, 'abc', true], ['abc', 12345, true]];

test('an old core (no served_ui) keeps the exact SHA decision on every page, marked or not', () => {
    for (const [prev, sha, connected] of SHA_CASES) {
        for (const marked of [true, false]) {
            assert.equal(decideServedPage(prev, sha === undefined ? {} : { sha }, connected, marked),
                decide(prev, sha, connected), `prev=${prev} sha=${sha} connected=${connected} marked=${marked}`);
        }
    }
    for (const body of [undefined, null, 'portal text']) assert.equal(decideServedPage('abc', body, true, false), 'reload_unknown');
});

test('the core\'s own page keeps the SHA decision while built-in is served, a refused directory included', () => {
    for (const [prev, sha, connected] of SHA_CASES) {
        for (const servedUi of [{ source: 'builtin' }, { source: 'builtin', error: 'no index.html' }]) {
            assert.equal(decideServedPage(prev, { sha, served_ui: servedUi }, connected, true), decide(prev, sha, connected));
        }
    }
});

test('the core\'s own page reloads once into a directory or none, even on the same SHA, never before connecting', () => {
    for (const servedUi of [{ source: 'directory' }, { source: 'none' }, null, {}]) {
        assert.equal(decideServedPage('abc', { sha: 'abc', served_ui: servedUi }, true, true), 'reload_source');
        assert.equal(decideServedPage(null, { sha: 'abc', served_ui: servedUi }, false, true), 'keep', 'a fresh page cannot loop');
    }
});

test('an unmarked page under a new core owns its refresh: core changes never reload it', () => {
    for (const servedUi of [{ source: 'builtin' }, { source: 'directory' }, { source: 'none' }, null]) {
        for (const sha of ['def', '', undefined, 12345]) {
            assert.equal(decideServedPage('abc', { sha, served_ui: servedUi }, true, false), 'keep');
        }
    }
});

const markedDocument = (content = 'core') => ({
    getElementById: () => null,
    querySelector: (selector) => (selector === 'meta[name="ouroboros-ui-owner"]' && content !== null
        ? { getAttribute: (name) => (name === 'content' ? content : null) } : null),
});

test('only the exact core marker claims the page', () => {
    assert.equal(pageOwnedByCore(markedDocument('core')), true);
    for (const doc of [markedDocument('Core'), markedDocument(''), markedDocument(null), { getElementById: () => null }, undefined]) {
        assert.equal(pageOwnedByCore(doc), false);
    }
});

test('post-open: the marked page reloads once into the newly selected interface; the first open keeps', async () => {
    const env = installEnv([{ body: { sha: 'abc', served_ui: { source: 'builtin' } } }]);
    globalThis.document = markedDocument();
    const ws = new WS('ws://unused');
    try {
        ws._refreshStateAfterOpen(false);
        await settle();
        assert.equal(ws._lastSha, 'abc');
        env.setFetch(async () => ({ ok: true, status: 200, json: async () => ({ sha: 'abc', served_ui: { source: 'directory' } }) }));
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 1);
        assert.match(env.replaced[0], /_ouro_reason=ui-source/);
    } finally {
        await teardown(ws);
    }
});

test('post-open: an unmarked page under a new core stays on a changed SHA and remembers it', async () => {
    const env = installEnv([{ body: { sha: 'def', served_ui: { source: 'builtin' } } }]);
    const ws = new WS('ws://unused');
    try {
        ws._lastSha = 'abc';
        ws._refreshStateAfterOpen(true);
        await settle();
        assert.equal(env.replaced.length, 0, 'the custom or external host owns its assets');
        assert.equal(ws._lastSha, 'def', 'kept for a later old-core compare');
    } finally {
        await teardown(ws);
    }
});

test('recovery: the marked page reloads into a new interface; the healthy-socket fuse is unaffected on an unmarked one', async () => {
    let env = installEnv([{ body: { sha: 'abc', served_ui: { source: 'none' } } }]);
    globalThis.document = markedDocument();
    let ws = new WS('ws://unused');
    try {
        Object.assign(ws, { _wasConnected: true, _lastSha: 'abc' });
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'source reload');
        assert.equal(env.fetchCalls(), 1);
        assert.match(env.replaced[0], /_ouro_reason=ui-source/);
    } finally {
        await teardown(ws);
    }
    env = installEnv([{ body: { sha: 'def', served_ui: { source: 'none' } } }]);
    ws = new WS('ws://unused');
    try {
        Object.assign(ws, { _wasConnected: true, _lastSha: 'abc' });
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'fuse reload');
        assert.equal(env.fetchCalls(), RECOVERY_HEALTHY_PROBE_LIMIT);
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/);
    } finally {
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// Signed out: the gate refuses the socket before accepting it, so the browser
// reports only an ordinary close; the state probe's 401 is the legible fact.
// ---------------------------------------------------------------------------

test('a 401 recovery probe hands over to the gate sign-in once and stops probing', async () => {
    const env = installEnv([{ ok: false, status: 401, body: { error: 'Authentication required.' } }]);
    globalThis.window.location.href = 'http://127.0.0.1:8000/?_ouro_refresh=9&_ouro_reason=sha-change&room=2#settings';
    const ws = new WS('ws://unused');
    try {
        Object.assign(ws, { _wasConnected: true, _lastSha: 'abc' });
        ws._pendingMessages.push({ type: 'chat', text: 'queued', client_message_id: 'msg-1' });
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'sign-in handoff');
        await settle(6);
        assert.deepEqual(env.replaced, ['/auth/login?next=%2F%3Froom%3D2%23settings']);
        assert.equal(env.fetchCalls(), 1, 'no further probe after the handoff');
        assert.equal(ws._uiRecoveryProbeInFlight, false);
        assert.equal(ws._uiRecoveryTimer, null);
        assert.equal(ws._pendingMessages.length, 1, 'nothing queued is sent on the way out');
    } finally {
        await teardown(ws);
    }
});

test('a 401 probe hands over without a previous connection too; other refusals keep probing', async () => {
    const env = installEnv([{ ok: false, status: 401 }]);
    const ws = new WS('ws://unused');
    try {
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'handoff without a previous connection');
        assert.match(env.replaced[0], /^\/auth\/login\?next=/);
    } finally {
        await teardown(ws);
    }
    for (const status of [403, 404, 502]) {
        const other = installEnv([{ ok: false, status }]);
        const probe = new WS('ws://unused');
        try {
            Object.assign(probe, { _wasConnected: true, _lastSha: 'abc' });
            probe._scheduleUiRecovery('socket-disconnect', 0);
            await waitFor(() => other.fetchCalls() >= 3, `HTTP ${status} re-arms`);
            assert.deepEqual(other.replaced, [], `HTTP ${status} is not a sign-in`);
        } finally {
            await teardown(probe);
        }
    }
});

test('a 401 that resolves after the socket reopened, or in a dead episode, is discarded', async () => {
    const env = installEnv([]);
    const releases = [];
    env.setFetch(() => new Promise((resolve) => { releases.push(resolve); }));
    const ws = new WS('ws://unused');
    const refused = { ok: false, status: 401, json: async () => ({}) };
    try {
        ws._wasConnected = true;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => releases.length === 1, 'probe in flight');
        ws.ws = { readyState: FakeSocket.OPEN };
        releases[0](refused);
        await settle(6);
        assert.deepEqual(env.replaced, [], 'the reopened socket proves the session');
        ws.ws = null;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => releases.length === 2, 'second probe in flight');
        ws._recoveryEpisode += 1;
        releases[1](refused);
        await settle(6);
        assert.deepEqual(env.replaced, [], 'a stale episode decides nothing');
    } finally {
        releases.forEach((release) => release({ ok: false, status: 500, json: async () => ({}) }));
        await teardown(ws);
    }
});

// ---------------------------------------------------------------------------
// The server answers while the socket stays down: an observation for the
// room's status (never a cause: a handshake refused before accept, a dropped
// upgrade and a frozen runtime all reach the page as the same 1006).
// ---------------------------------------------------------------------------

test('a socket attempt failing after the core answered is said; an unanswered probe, sign-in, a reload and an open end it', async () => {
    const env = installEnv([]);
    const answers = [];
    env.setFetch(() => new Promise((resolve) => { answers.push(resolve); }));
    const answer = async (script) => {
        await waitFor(() => answers.length > 0, 'a probe in flight');
        answers.shift()({ ok: script.ok !== false, status: script.status ?? 200, json: async () => script.body ?? {} });
        await settle(6);
    };
    const ws = new WS('ws://unused');
    const said = [];
    ws.on(SERVER_ANSWERS_EVENT, ({ serverAnswers }) => said.push(serverAnswers));
    const failAttempt = () => {
        ws.connect();
        FakeSocket.instances.at(-1).onclose();
        ws._clearReconnectTimer();  // the test starts every attempt itself
    };
    try {
        Object.assign(ws, { _wasConnected: true, _lastSha: 'abc', _recoveryReloadFired: true });  // the fuse stays out
        failAttempt();
        ws._clearUiRecoveryTimer();
        ws.connect();  // still under way when the core answers
        const early = FakeSocket.instances.at(-1);
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await answer({ body: { sha: 'abc' } });
        assert.deepEqual([said, ws.serverAnswersWhileDown()], [[], false], 'an answer alone proves nothing about the socket');
        early.onclose();
        ws._clearReconnectTimer();
        assert.deepEqual(said, [], 'an attempt begun before the answer may have met a core that was down');
        ws.connect();
        const slow = FakeSocket.instances.at(-1);  // begun after the first successful probe
        await answer({ body: { sha: 'abc' } });  // another success while this attempt is still connecting
        slow.onclose();
        ws._clearReconnectTimer();
        assert.deepEqual([said, ws.serverAnswersWhileDown()], [[true], true],
            'a later healthy probe must not erase the observation preceding this slow attempt');
        await answer({ ok: false, status: 502 });
        assert.deepEqual([said, ws.serverAnswersWhileDown()], [[true, false], false], 'the core no longer answers');
        failAttempt();
        assert.deepEqual(said, [true, false], 'without a new answer a failure is only Reconnecting');
        await answer({ body: { sha: 'abc' } });
        failAttempt();
        assert.deepEqual(said, [true, false, true]);
        // An open ends it with the episode (the open event re-derives every status); the opened socket's
        // later close proves nothing either.
        ws.connect();
        const opened = FakeSocket.instances.at(-1);
        opened.readyState = FakeSocket.OPEN;
        opened.onopen();
        assert.equal(ws.serverAnswersWhileDown(), false);
        opened.readyState = FakeSocket.CLOSED;
        opened.onclose();
        ws._clearReconnectTimer();
        assert.deepEqual([said, ws.serverAnswersWhileDown()], [[true, false, true], false]);
    } finally {
        while (answers.length) answers.shift()({ ok: false, status: 500, json: async () => ({}) });
        await teardown(ws);
    }
    for (const [what, script, reload] of [['sign-in', { ok: false, status: 401 }, /^\/auth\/login\?next=/],
        ['a changed build', { body: { sha: 'def' } }, /_ouro_reason=sha-change/]]) {
        const other = installEnv([]);
        const pending = [];
        other.setFetch(() => new Promise((resolve) => { pending.push(resolve); }));
        const probe = new WS('ws://unused');
        const heard = [];
        probe.on(SERVER_ANSWERS_EVENT, ({ serverAnswers }) => heard.push(serverAnswers));
        try {
            Object.assign(probe, { _wasConnected: true, _lastSha: 'abc', _serverAnswers: true, _answeredAtAttempt: 0 });
            probe._scheduleUiRecovery('socket-disconnect', 0);
            await waitFor(() => pending.length === 1, `${what}: probe in flight`);
            pending.shift()({ ok: script.ok !== false, status: script.status ?? 200, json: async () => script.body ?? {} });
            await waitFor(() => other.replaced.length === 1, `${what}: the page leaves`);
            assert.match(other.replaced[0], reload);
            assert.deepEqual(heard, [false], `${what} ends the observation before the page leaves`);
        } finally {
            while (pending.length) pending.shift()({ ok: false, status: 500, json: async () => ({}) });
            await teardown(probe);
        }
    }
});

test('a page whose socket never opened is not reloaded by the fuse; after its first open the next episode has one', async () => {
    const env = installEnv([{ body: { sha: 'abc' } }]);
    const ws = new WS('ws://unused');
    try {
        // Never connected (a refused socket, or the fuse's own reload of one): the same served page, a socket that
        // stays down, and no reload every few minutes.
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.fetchCalls() >= RECOVERY_HEALTHY_PROBE_LIMIT + 3, 'probes past the limit');
        assert.deepEqual(env.replaced, []);
        // A connection ends the episode; the next one has its own fuse.
        ws.ws = { readyState: FakeSocket.OPEN };
        await settle(6);
        ws.ws = null;
        ws._clearUiRecoveryTimer();
        ws.connect();
        const sock = FakeSocket.instances.at(-1);
        sock.readyState = FakeSocket.OPEN;
        sock.onopen();
        await settle();
        ws.ws = null;
        ws._scheduleUiRecovery('socket-disconnect', 0);
        await waitFor(() => env.replaced.length === 1, 'the next episode\'s fuse');
        assert.match(env.replaced[0], /_ouro_reason=socket-disconnect/);
    } finally {
        await teardown(ws);
    }
});
