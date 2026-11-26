// W3: the page's first socket attempt fails before any open while the room's first newest read is already out.
// Through the real client modules (ws.js, the room's socket view and the chat instance) over a fictional history.
// That read runs over HTTP and works; with no connection that ever opened there is nothing for it to be older than,
// so it finishes and paints (no cancelled page, no loading pill left with nothing to retry). A close after a
// connection did happen still ends the read it interrupts, and the reconnect reads again.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { roomSocket } from '../modules/room_retire.js';
import { WS } from '../modules/ws.js';
import { ElementStub, PREVIEW_SCOPE, installDom, restoreDom } from './chat_dom_fixture.js';

const quiet = async (ticks = 12) => {
    for (let i = 0; i < ticks; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

const until = async (predicate, limit = 60) => {
    for (let i = 0; i < limit; i += 1) {
        if (predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    throw new Error('condition did not settle');
};

const isoAt = (sec) => new Date(Date.UTC(2026, 10, 26, 10, 0, sec)).toISOString();
const CHAT_ID = 2;

// `openedBeforeRoom`: the page's socket opens first and the room mounts on it (a Project opened later).
function makeEntry(t, { openedBeforeRoom = false } = {}) {
    const reads = [];     // every newest/page/after read the room made: held until the test answers it, cancelled by its signal
    const sockets = [];   // every socket attempt the page's client made
    let readingArrivals = 0;
    class FakeSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        constructor(url) { this.url = url; this.readyState = FakeSocket.CONNECTING; sockets.push(this); }
        send() {}
        close() { this.readyState = FakeSocket.CLOSED; }
    }
    const { prior, mount } = installDom(async (url, init = {}) => {
        const address = new URL(String(url), 'http://local');
        if (address.pathname === '/api/chat/history') {
            return new Promise((resolve, reject) => {
                const read = {
                    cursor: address.searchParams.get('cursor'),
                    mode: address.searchParams.get('mode') || (address.searchParams.get('cursor') ? 'page' : 'newest'),
                    reason: address.searchParams.get('reason') || '',
                    decision: address.searchParams.get('decision') || '',
                    aborted: false,
                    release: (body) => resolve({ ok: true, status: 200, json: async () => ({ profile_scope: PREVIEW_SCOPE, ...body }) }),
                };
                init.signal?.addEventListener?.('abort', () => {
                    read.aborted = true;
                    reject(new DOMException('Aborted', 'AbortError'));
                }, { once: true });
                reads.push(read);
            });
        }
        if (address.pathname === '/api/state') {
            return { ok: true, status: 200, json: async () => ({ sha: 'sha-1', active_direct_turns: [] }) };
        }
        return { ok: true, status: 200, json: async () => ({}) };
    });
    const priorSocket = globalThis.WebSocket;
    const priorQuery = ElementStub.prototype.querySelector;
    // Allow nested lookups (status badge) in the flat DOM stub.
    ElementStub.prototype.querySelector = function (selector) {
        const direct = priorQuery.call(this, selector);
        if (direct) return direct;
        for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
        return null;
    };
    globalThis.WebSocket = FakeSocket;
    const ws = new WS('ws://test/ws');
    const open = (index) => { sockets[index].readyState = FakeSocket.OPEN; sockets[index].onopen(); };
    if (openedBeforeRoom) { ws.connect(); open(0); }
    const instance = createChatInstance({
        ws: roomSocket(ws),
        state: { activePage: 'chat', projectChatIds: new Set([CHAT_ID]), unreadCount: 0 },
        updateUnreadBadge() {},
        onReadingLatest: () => { readingArrivals += 1; },
        stateSnapshots: {
            begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true,
            apply() {},
            fail() {},
            latest: () => ({ supervisor_ready: true }),
        },
        chatId: CHAT_ID,
        projectId: 'proj-w3',
        idPrefix: 'chat',
        mountEl: mount,
        asPanel: true,
    });
    t.after(() => {
        instance.destroy();
        for (const clear of ['_clearReconnectTimer', '_clearUiRecoveryTimer', '_clearWatchdogTimer', '_clearPostOpenRetryTimer']) ws[clear]();
        ElementStub.prototype.querySelector = priorQuery;
        globalThis.WebSocket = priorSocket;
        restoreDom(prior);
    });
    const messages = globalThis.document.byId.get('chat-messages');
    const composer = globalThis.document.byId.get('chat-input-area');
    const scrollBtn = globalThis.document.byId.get('chat-scroll-bottom');
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 100, width: 100, height: bottom - top });
    messages.getBoundingClientRect = () => box(0, 20);
    composer.getBoundingClientRect = () => box(20, 60);
    const placeAtTop = (id) => {
        const rect = ElementStub.prototype.getBoundingClientRect;
        ElementStub.prototype.getBoundingClientRect = function () {
            if (this.dataset?.historyId !== id) return rect.call(this);
            const top = -messages.scrollTop;
            return { top, bottom: top + 20, left: 0, right: 100, width: 100, height: 20 };
        };
        t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
    };
    const row = (id, role, text, n) => ({ role, chat_id: CHAT_ID, source: 'web', text, content: text, ts: isoAt(n), history_id: id,
        history_position: { source: 'chat', offset: n } });
    const history = (extra = {}) => ({
        messages: [row('chat:q1', 'user', 'Where did we stop yesterday?', 1), row('chat:a1', 'assistant', 'At the release notes draft.', 2)],
        progress: [], has_more: false, page_cursor: 'page:1', next_cursor: null, after_cursor: null,
        window: { complete: true, truncated_by: [] }, coverage: null,
        ...extra,
    });
    return {
        ws, reads, sockets, instance, open, history, row, messages, placeAtTop,
        connect() { ws.connect(); },
        // A failed or lost socket attempt, as the browser reports it: an error, then the close.
        close(index) { sockets[index].readyState = FakeSocket.CLOSED; sockets[index].onerror?.(); sockets[index].onclose(); },
        status: () => mount.querySelector('.status-badge')?.textContent || '',
        scrollLabel: () => scrollBtn?.getAttribute('aria-label') || '',
        arrivals: () => readingArrivals,
        shown: () => messages.querySelectorAll('[data-history-id]').map((n) => n.dataset.historyId),
        buttons: () => messages.querySelectorAll('button').map((b) => b.textContent),
    };
}

test('W3: a first socket attempt that fails before any open leaves the first newest read to paint', async (t) => {
    const r = makeEntry(t);
    r.connect();
    await until(() => r.reads.length === 1);

    r.close(0);  // the page's first close, before any open
    await quiet();
    assert.equal(r.status(), 'Reconnecting…', 'the outage itself is still shown');
    assert.equal(r.reads[0].aborted, false, 'the read over working HTTP is not cancelled');

    r.reads[0].release(r.history());
    await quiet();
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.deepEqual(r.shown(), ['chat:q1', 'chat:a1']);
    assert.equal(r.buttons().includes('Loading saved history…'), false, 'no loading pill is left with nothing behind it');
});

test('W3: the first open after such a close finds the history read and asks for no second newest page', async (t) => {
    const r = makeEntry(t);
    r.connect();
    await until(() => r.reads.length === 1);

    r.close(0);
    await quiet();
    r.reads[0].release(r.history());
    await quiet();

    r.connect();      // the next attempt
    r.open(1);        // the page's first open
    await quiet();
    assert.equal(r.reads.length, 1, 'the loaded history is shared, not read again');
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.deepEqual(r.shown(), ['chat:q1', 'chat:a1']);
});

// The control: what the first close must not change. Once a connection happened, a close supersedes the read it
// interrupts (a late answer draws nothing) and the reconnect reads again.
test('W3 control: after a connection happened, a close still cancels the read in flight and the reconnect reads again', async (t) => {
    const r = makeEntry(t);
    r.connect();
    await until(() => r.reads.length === 1);
    r.open(0);        // the page has connected
    await quiet();

    r.close(0);       // and then loses it
    await quiet();
    assert.equal(r.reads[0].aborted, true, 'the interrupted read is cancelled');
    r.reads[0].release(r.history());
    await quiet();
    assert.equal(r.instance.hasPaintedHistory(), false, 'a late answer to the cancelled read draws nothing');
    assert.deepEqual(r.shown(), []);

    r.connect();
    r.open(1);
    await until(() => r.reads.length === 2);
    r.reads[1].release(r.history());
    await quiet();
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.deepEqual(r.shown(), ['chat:q1', 'chat:a1'], 'the reconnect read paints the history');
});

test('W3 control: a room mounted on an already-open socket has its read cancelled by the next close', async (t) => {
    const r = makeEntry(t, { openedBeforeRoom: true });
    await until(() => r.reads.length === 1);

    r.close(0);       // the room never heard an open, but the page's socket had opened
    await quiet();
    assert.equal(r.reads[0].aborted, true, 'the page had connected: this close is an outage');
    r.reads[0].release(r.history());
    await quiet();
    assert.equal(r.instance.hasPaintedHistory(), false);
});

test('cold boot before first socket opens preserves one-shot unread landing over HTTP and waits for connected paint for read credit', async (t) => {
    const r = makeEntry(t);
    r.placeAtTop('chat:late');
    const lateRow = { ...r.row('chat:late', 'assistant', 'Arrived last, written earlier.', 0), ts: isoAt(0) };
    await until(() => r.reads.length === 1);

    void r.instance.showLatest({ arrive: true });
    const boot = r.instance.refreshHistory({ revision: 1 });
    r.reads[0].release(r.history({
        messages: [lateRow, r.row('chat:q1', 'user', 'Follow up', 1), r.row('chat:a1', 'assistant', 'Earlier reply', 2)],
        after_cursor: 'after:1',
        window: { complete: true, truncated_by: [], latest_message: { history_id: 'chat:late', out_of_order: true } },
    }));
    await until(() => r.reads.length === 2);
    assert.equal(r.reads[1].mode, 'after');
    r.reads[1].release({ messages: [], after_cursor: 'after:2', window: { complete: false, truncated_by: ['page'] } });
    const bootReceipt = await boot;
    await quiet();

    assert.equal(bootReceipt.painted, false, 'read credit waits until socket connects and paints');
    assert.equal(r.arrivals(), 0, 'onReadingLatest is withheld while unopened');
    assert.equal(r.messages.scrollTop, 0, 'one-shot unread landing still lands on the newest message over healthy HTTP');
    assert.notEqual(r.scrollLabel(), 'Retry loading latest messages');

    // First real socket open shares initial HTTP and credits on connected paint without an extra whole newest.
    r.connect();
    r.open(0);
    await quiet();
    const afterOpen = r.instance.refreshHistory({ revision: 1 });
    await until(() => r.reads.length === 3);
    assert.equal(r.reads[2].mode, 'after');
    assert.equal(r.reads[2].cursor, 'after:2');
    r.reads[2].release({ messages: [], after_cursor: 'after:3', window: { complete: false, truncated_by: ['page'] } });
    const connected = await afterOpen;
    assert.deepEqual([connected.painted, connected.read], [true, true]);
    assert.equal(r.reads.filter((e) => e.mode === 'newest').length, 1, 'no extra whole newest after first open');
});

test('a failed first socket attempt before any open leaves no fake retry, walks older pages to land unread, and avoids an extra newest on first open', async (t) => {
    const r = makeEntry(t);
    r.placeAtTop('chat:old');
    r.connect();
    await until(() => r.reads.length === 1);

    r.close(0);  // first socket attempt fails before any open
    await quiet();
    void r.instance.showLatest({ arrive: true });
    const boot = r.instance.refreshHistory({ revision: 1 });
    r.reads[0].release(r.history({
        has_more: true, next_cursor: 'older:1', after_cursor: 'after:1',
        window: { complete: false, truncated_by: ['page'], latest_message: { history_id: 'chat:old', out_of_order: true } },
    }));
    await until(() => r.reads.length === 2);
    assert.equal(r.reads[1].mode, 'after', 'first close before open must not set fake catchUp.owed');
    r.reads[1].release({ messages: [], after_cursor: 'after:2', window: { complete: false, truncated_by: ['page'] } });
    await until(() => r.reads.length === 3);
    assert.equal(r.reads[2].cursor, 'older:1', 'unread opening walks older pages over healthy HTTP');
    r.reads[2].release({
        messages: [{ ...r.row('chat:old', 'assistant', 'Older latest arrival', 0), ts: isoAt(0) }],
        has_more: false, page_cursor: 'page:older:1', next_cursor: null,
        window: { complete: false, truncated_by: ['page'] },
    });
    const bootReceipt = await boot;
    await quiet();

    assert.equal(r.reads[0].aborted, false);
    assert.equal(bootReceipt.painted, false, 'read credit is withheld before first socket open');
    assert.equal(r.arrivals(), 0);
    assert.ok(r.shown().includes('chat:old'), 'older page delivered the newest message');
    assert.equal(r.messages.scrollTop, 0, 'one-shot unread landing reached the older newest message');
    assert.notEqual(r.scrollLabel(), 'Retry loading latest messages', 'first close before open leaves no fake Retry button');

    r.connect();
    r.open(1);
    await quiet();
    const afterOpen = r.instance.refreshHistory({ revision: 1 });
    await until(() => r.reads.length === 4);
    assert.equal(r.reads[3].mode, 'after', 'first open uses after, not a second whole newest');
    assert.equal(r.reads[3].cursor, 'after:2');
    r.reads[3].release({ messages: [], after_cursor: 'after:3', window: { complete: false, truncated_by: ['page'] } });
    const connected = await afterOpen;
    assert.deepEqual([connected.painted, connected.read], [true, true]);
    assert.equal(r.reads.filter((e) => e.mode === 'newest').length, 1, 'no extra whole newest forced by fake catchUp.owed');
});

test('control: a post-open disconnect during unread opening starts no reads, preserves the one-shot landing, and withholds credit if disconnect precedes paint', async (t) => {
    const r = makeEntry(t, { openedBeforeRoom: true });
    r.placeAtTop('chat:late2');
    await until(() => r.reads.length === 1);
    r.reads[0].release(r.history({ after_cursor: 'after:1' }));
    await quiet();

    // 1. Real disconnect after a prior open: 0 reads started, one-shot landing preserved for reconnect.
    r.close(0);
    await quiet();
    const readsBefore = r.reads.length;
    void r.instance.showLatest({ arrive: true });
    const offline = await r.instance.refreshHistory({ revision: 2 });
    assert.equal(offline.painted, false);
    assert.equal(r.reads.length, readsBefore, 'post-open disconnect starts 0 reads');

    r.connect();
    r.open(1);
    await until(() => r.reads.length === readsBefore + 1);
    assert.equal(r.reads[readsBefore].mode, 'newest');
    const reconn = r.instance.refreshHistory({ revision: 2 });
    r.reads[readsBefore].release(r.history({
        messages: [{ ...r.row('chat:late2', 'assistant', 'Late after reconnect', 0), ts: isoAt(0) },
            r.row('chat:q1', 'user', 'q1', 1), r.row('chat:a1', 'assistant', 'a1', 2)],
        after_cursor: 'after:2',
        window: { complete: true, truncated_by: [], latest_message: { history_id: 'chat:late2', out_of_order: true } },
    }));
    await until(() => r.reads.length === readsBefore + 2);
    r.reads[readsBefore + 1].release({ messages: [], after_cursor: 'after:3', window: { complete: false, truncated_by: ['page'] } });
    const afterReconn = await reconn;
    await quiet();
    assert.deepEqual([afterReconn.painted, afterReconn.read], [true, true]);
    assert.equal(r.messages.scrollTop, 0, 'one-shot unread landing survived the post-open disconnect and landed on reconnect');

    // 2. Disconnect after read completes but before the 2-rAF paint finishes grants no read credit.
    const priorRaf = globalThis.requestAnimationFrame;
    const queuedRafs = [];
    globalThis.requestAnimationFrame = (fn) => { queuedRafs.push(fn); return queuedRafs.length; };
    t.after(() => { globalThis.requestAnimationFrame = priorRaf; });

    const prePaint = r.instance.refreshHistory({ revision: 3 });
    await until(() => r.reads.length === readsBefore + 3);
    r.reads[readsBefore + 2].release({
        messages: [r.row('chat:a3', 'assistant', 'New message', 3)],
        after_cursor: 'after:4',
        window: { complete: false, truncated_by: ['page'], latest_message: { history_id: 'chat:a3', out_of_order: false } },
    });
    await until(() => queuedRafs.length > 0);
    r.close(1);  // disconnect BEFORE paint frames run
    while (queuedRafs.length) queuedRafs.shift()();
    const disconnectedBeforePaint = await prePaint;
    assert.equal(disconnectedBeforePaint.painted, false, 'read credit requires connected paint');
});
