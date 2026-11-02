// One stuck /api/state read must not own the page's state (#2343): the bounded read, the sequencer's
// retirement and the page's real readers under a request or body that never settles. Time is virtual;
// STATE_READ_LIFETIME_MS is an engineering tolerance, not a target. The transport below ignores
// cancellation, the worst case: the read must settle by itself.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { STATE_READ_LIFETIME_MS as LIFETIME, readState } from '../modules/api_client.js';
import { createStateSnapshotSequencer, readStateSnapshot } from '../modules/state_snapshots.js';
import { createChatInstance } from '../modules/chat.js';
import { initEvolution } from '../modules/evolution.js';
import { WS } from '../modules/ws.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { installDomDouble } from './dom_double.js';
import { clock, flush, observe } from './state_read_fixture.js';

/** Scripted /api/state reads, the last entry repeating: {hang: 'fetch'|'body'} waits for `release` and ignores
 * its signal; {status}, {reject}, {malformed} and {body} answer at once. Other URLs answer `other(url)`. */
function transport(script, other = () => ({})) {
    const calls = [];
    const queue = [...script];
    const answer = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
    async function fetchImpl(url, init = {}) {
        url = String(url);
        if (!url.startsWith('/api/state')) return answer(200, other(url));
        const entry = queue.length > 1 ? queue.shift() : queue[0];
        const call = { url, signal: init.signal, bodyReads: 0, release: () => {} };
        calls.push(call);
        if (entry.hang === 'fetch') return new Promise((resolve) => { call.release = (body) => resolve(answer(200, body)); });
        if (entry.hang === 'body') {
            const body = new Promise((resolve) => { call.release = resolve; });
            return { ok: true, status: 200, json: () => body };
        }
        if (entry.reject) throw new TypeError('Failed to fetch');
        if (entry.malformed) return { ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected token <'); } };
        const status = entry.status ?? 200;
        return { ok: status < 300, status, json: async () => { call.bodyReads += 1; return entry.body ?? {}; } };
    }
    return { fetch: fetchImpl, calls };
}

function withFetch(t, fetchImpl) {
    const prior = globalThis.fetch;
    globalThis.fetch = fetchImpl;
    t.after(() => { globalThis.fetch = prior; });
}

const FRESH = { accounting: { available: true }, spent_usd: 1.25, marker: 'fresh' };
const FIRST = { accounting: { available: true }, spent_usd: 0.5, marker: 'first' };

// ---------------------------------------------------------------- the bounded read (api_client.readState)

for (const hang of ['fetch', 'body']) {
    test(`a ${hang} that never settles ends the read as expired at its lifetime and cancels its transport`, async (t) => {
        const advance = clock(t);
        const tr = transport([{ hang }]);
        withFetch(t, tr.fetch);
        const read = observe(readState());
        await advance(LIFETIME - 1000);
        assert.equal(read.settled, false, 'inside its lifetime the read still waits');
        await advance(1000);
        assert.deepEqual([read.error?.name, read.error?.reason], ['StateReadEnded', 'expired']);
        assert.equal(tr.calls[0].signal.aborted, true);
    });
}

test('a body that does not parse is an answer without state; a refusal is not read; a transport error stays itself', async (t) => {
    const tr = transport([{ malformed: true }, { status: 503 }, { reject: true }]);
    withFetch(t, tr.fetch);
    const malformed = await readState();
    assert.deepEqual([malformed.response.ok, malformed.state], [true, undefined]);
    const refused = await readState();
    assert.deepEqual([refused.response.status, refused.state, tr.calls[1].bodyReads], [503, undefined, 0]);
    await assert.rejects(readState(), (error) => error instanceof TypeError);
});

test('the caller\'s signal ends the read at once although the transport ignores it; an aborted one sends nothing', async (t) => {
    const tr = transport([{ hang: 'fetch' }]);
    withFetch(t, tr.fetch);
    const controller = new AbortController();
    const read = observe(readState('/api/state', { signal: controller.signal }));
    await flush();
    controller.abort();
    await flush();
    assert.equal(read.error?.reason, 'aborted');
    assert.equal(tr.calls[0].signal.aborted, true);
    await assert.rejects(readState('/api/state', { signal: controller.signal }), { reason: 'aborted' });
    assert.equal(tr.calls.length, 1, 'an already aborted signal sends nothing');
});

test('a settled read leaves neither its lifetime timer nor its abort listener behind', async (t) => {
    const timers = new Set();
    const { setTimeout: realSet, clearTimeout: realClear } = globalThis;
    globalThis.setTimeout = (fn, ms) => { const id = realSet(fn, ms); timers.add(id); return id; };
    globalThis.clearTimeout = (id) => { timers.delete(id); realClear(id); };
    t.after(() => { globalThis.setTimeout = realSet; globalThis.clearTimeout = realClear; });
    withFetch(t, transport([{ body: FRESH }, { status: 500 }, { reject: true }]).fetch);
    const { signal } = new AbortController();
    const listeners = new Set();
    const [add, remove] = [signal.addEventListener.bind(signal), signal.removeEventListener.bind(signal)];
    signal.addEventListener = (type, fn, options) => { listeners.add(fn); add(type, fn, options); };
    signal.removeEventListener = (type, fn) => { listeners.delete(fn); remove(type, fn); };
    assert.equal((await readState('/api/state', { signal })).state.marker, 'fresh');
    assert.equal((await readState('/api/state', { signal })).response.status, 500);
    await assert.rejects(readState('/api/state', { signal }));
    assert.deepEqual([timers.size, listeners.size], [0, 0]);
});

// ---------------------------------------------------------------- the sequencer and the shared reader

function page() {
    const applied = [];
    const unavailable = [];
    const snapshots = createStateSnapshotSequencer((data) => applied.push(data?.marker), undefined,
        (requestedAt, generation) => unavailable.push(generation));
    const read = async (force = false) => {
        const request = await snapshots.gate(force);
        if (request) await readStateSnapshot(snapshots, request);
    };
    return { snapshots, applied, unavailable, read };
}

test('retire detaches a hung read: its reader and the readers joined to it settle, and nothing is painted unavailable', async (t) => {
    const tr = transport([{ hang: 'fetch' }]);
    withFetch(t, tr.fetch);
    const p = page();
    const initiating = observe(p.read(false));
    await flush();
    const joined = observe(p.read(false));
    const forced = observe(p.read(true));
    await flush();
    assert.equal(tr.calls.length, 1, 'a tick and a forced caller wait behind the read in flight');
    p.snapshots.retire();
    await flush();
    assert.deepEqual([initiating.settled, joined.settled, initiating.error, joined.error], [true, true, undefined, undefined]);
    assert.deepEqual(p.unavailable, [], 'retirement is not a failed read');
    assert.equal(tr.calls[0].signal.aborted, true, 'the retired read is cancelled');
    assert.equal(tr.calls.length, 2, 'the coalesced forced caller reads at once');
    // The retired transport answers late, while the newer read is in flight: it paints and frees nothing.
    tr.calls[0].release(FIRST);
    await flush();
    const tick = observe(p.read(false));
    await flush();
    assert.deepEqual(p.applied, []);
    assert.equal(tr.calls.length, 2, 'a tick still joins the newer read');
    assert.equal(forced.settled || tick.settled, false);
    tr.calls[1].release(FRESH);
    await flush();
    assert.deepEqual(p.applied, ['fresh']);
    assert.equal(forced.settled && tick.settled, true);
});

test('a read whose body never arrives fails once at its lifetime, releasing its waiters and one coalesced follow-up', async (t) => {
    const advance = clock(t);
    const tr = transport([{ hang: 'body' }, { body: FRESH }]);
    withFetch(t, tr.fetch);
    const p = page();
    const initiating = observe(p.read(false));
    await flush();
    const joined = observe(p.read(false));
    const forced = [observe(p.read(true)), observe(p.read(true))];
    await advance(LIFETIME - 1000);
    assert.deepEqual([p.unavailable.length, initiating.settled, joined.settled], [0, false, false]);
    await advance(1000);
    assert.equal(p.unavailable.length, 1, 'the expired read is a failed read');
    assert.equal(initiating.settled && joined.settled, true);
    assert.equal(tr.calls.length, 2, 'both forced callers share one follow-up');
    assert.deepEqual(p.applied, ['fresh']);
    assert.equal(forced.every((box) => box.settled), true);
    await advance(2 * LIFETIME);
    assert.deepEqual([p.unavailable.length, tr.calls.length], [1, 2], 'no timer of a settled read fires later');
});

test('a reconnect burst admits one new read per logical request, never one for the retired read', async (t) => {
    const tr = transport([{ hang: 'fetch' }]);  // every read waits for the test
    withFetch(t, tr.fetch);
    const p = page();
    observe(p.read(false));  // the periodic read in flight when the socket reopens; it never answers
    await flush();
    // ws 'open' in app.js listener order: Main's forced read (initChat), app.js's retire() and forced read,
    // then Evolution's forced read; a poll tick lands in the same turn.
    const main = observe(p.read(true));
    p.snapshots.retire();
    const app = observe(p.read(true));
    const evolution = observe(p.read(true));
    const tick = observe(p.read(false));
    await flush();
    assert.equal(tr.calls[0].signal.aborted, true);
    assert.equal(tr.calls.length, 2, 'Main\'s forced read, queued behind the retired one, starts at once');
    tr.calls[1].release(FIRST);
    await flush();
    assert.equal(main.settled && tick.settled, true);
    assert.equal(tr.calls.length, 3, 'the two forced callers after the retirement share one successor');
    tr.calls[2].release(FRESH);
    await flush();
    assert.equal(app.settled && evolution.settled, true);
    assert.deepEqual(p.applied, ['first', 'fresh']);
    assert.equal(tr.calls.length, 3, 'no further read is owed');
});

// ---------------------------------------------------------------- the page's real readers

const APP_SOURCE = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const BOOT = 'refreshProjectsNav(false).finally(() => ws.connect());';

/** app.js cannot load in Node: its refreshProjectsNav runs from the current source text, so a change there is tested. */
function appRefreshProjectsNav(stateSnapshots) {
    const start = APP_SOURCE.indexOf('\nasync function refreshProjectsNav(');
    assert.notEqual(start, -1, 'app.js defines refreshProjectsNav at top level');
    const text = APP_SOURCE.slice(start + 1, APP_SOURCE.indexOf('\n}\n', start) + 2);
    return new Function('stateSnapshots', 'readStateSnapshot', `${text}\nreturn refreshProjectsNav;`)(stateSnapshots, readStateSnapshot);
}

function mountMain(t, tr) {
    const env = installDom(tr.fetch);
    const handlers = new Map();
    const socket = {
        connects: 0,
        connect() { this.connects += 1; },
        on(type, fn) { (handlers.get(type) || handlers.set(type, new Set()).get(type)).add(fn); return () => handlers.get(type)?.delete(fn); },
        emit(type, msg) { for (const fn of handlers.get(type) || []) fn(msg); },
        isConnected: () => true,
        send: () => ({ status: 'sent', clientMessageId: 'cm-1' }),
    };
    const unavailable = [];
    let chat = null;
    const stateSnapshots = createStateSnapshotSequencer((data, at) => chat?.hydrateStateSnapshot(data, at), undefined,
        (at) => { unavailable.push(at); chat?.hydrateStateSnapshot(null, at); });
    const state = { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 };
    chat = createChatInstance({ ws: socket, state, updateUnreadBadge() {}, stateSnapshots, chatId: 1, idPrefix: 'chat', mountEl: env.mount });
    t.after(() => { chat.destroy(); restoreDom(env.prior); });  // hooks run in order: the instance first
    return { socket, stateSnapshots, state, unavailable };
}

const chatHistory = (url) => (url.startsWith('/api/chat/history') ? { messages: [], window: { complete: true } } : { active_direct_turns: [] });
const assistantText = () => (globalThis.document.byId.get('chat-messages')?.children || [])
    .filter((node) => node.classList?.contains('chat-bubble') && node.classList.contains('assistant'))
    .map((node) => node.innerHTML).join('\n');

for (const hang of ['fetch', 'body']) {
    test(`boot: when Main's first read's ${hang} never answers the socket still opens, and a stamped Project frame stays out of Main`, async (t) => {
        const advance = clock(t);
        const tr = transport([{ hang }, { body: FRESH }], chatHistory);
        const main = mountMain(t, tr);
        assert.equal(APP_SOURCE.split(BOOT).length, 2, 'app.js boots with exactly this statement');
        new Function('refreshProjectsNav', 'ws', BOOT)(appRefreshProjectsNav(main.stateSnapshots), main.socket);
        await advance(LIFETIME - 1000);
        assert.deepEqual([tr.calls.length, main.socket.connects], [1, 0], 'the prefetch joined Main\'s read and waits on it');
        await advance(1000);
        assert.equal(main.socket.connects, 1, 'the expired read settles the prefetch: the socket opens as after a failed one');
        assert.equal(main.unavailable.length, 1);
        assert.equal(main.state.projectChatIds.size, 0, 'nothing has taught the page its Projects yet');
        main.socket.emit('chat', { chat_id: 77, project_thread: true, role: 'assistant', system_type: 'proactive_message',
            content: 'Project-only words', markdown: false, ts: '2026-10-29T00:00:00Z' });
        main.socket.emit('chat', { chat_id: 1, role: 'assistant', system_type: 'proactive_message',
            content: 'Main words', markdown: false, ts: '2026-10-29T00:00:01Z' });
        assert.match(assistantText(), /Main words/);
        assert.doesNotMatch(assistantText(), /Project-only words/, 'the server stamp keeps an unlearned Project\'s frame out of Main');
        await advance(3000);
        assert.equal(main.stateSnapshots.latest()?.marker, 'fresh', 'Main\'s own poll then reads and applies a newer state');
    });
}

test('Main\'s socket-open read, queued behind a hung read, runs at once when app.js retires on open', async (t) => {
    const tr = transport([{ hang: 'fetch' }, { body: FRESH }], chatHistory);
    const main = mountMain(t, tr);
    await flush();
    assert.equal(tr.calls.length, 1, 'mount issues Main\'s forced header read, which hangs');
    main.socket.emit('open', { previouslyConnected: true });  // the instance's own open handler: a forced read
    await flush();
    assert.equal(tr.calls.length, 1, 'it queues behind the read in flight');
    main.stateSnapshots.retire();  // app.js's open handler
    await flush();
    assert.equal(tr.calls.length, 2);
    assert.equal(main.stateSnapshots.latest()?.marker, 'fresh');
    assert.deepEqual(main.unavailable, [], 'retirement painted nothing unavailable');
});

test('Evolution reads the request the sequencer names and recovers after a read whose body never arrives', async (t) => {
    const advance = clock(t);
    const runtime = { ...FRESH, runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } };
    const noHistory = { versions: [], saved_at: '', refresh: { state: 'idle' }, install: {} };
    const tr = transport([{ hang: 'body' }, { body: runtime }], (url) => (url.includes('evolution-data') ? noHistory : {}));
    const env = installDomDouble();
    withFetch(t, tr.fetch);
    let evolution = null;
    const snapshots = createStateSnapshotSequencer((data) => evolution?.hydrateStateSnapshot(data), undefined,
        () => evolution?.hydrateStateSnapshot(null));
    snapshots.apply(snapshots.begin(), { task_bindings: { t1: { project_id: 'a' } }, task_bindings_revision: 'r1' });
    const ws = new WS('ws://unused');
    const mount = env.doc.createElement('div');
    env.doc.body.append(mount);
    evolution = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' }, stateSnapshots: snapshots });
    t.after(() => { evolution.dispose(); env.restore(); });
    const note = () => env.doc.getElementById('evo-campaign-note').textContent;
    ws.emit('open');
    await flush();
    assert.equal(tr.calls[0].url, '/api/state?bindings_since=r1', 'its read asks what the sequencer asked');
    await advance(LIFETIME);
    assert.match(note(), /could not be read/);
    ws.emit('open');  // "It is read again when this tab opens."
    await flush();
    assert.equal(tr.calls.length, 2);
    assert.doesNotMatch(note(), /could not be read|Reading the runtime state/);
});
