// Preference-independent history start, waiter wakeup on
// cancellation/supersession, no new reads on close, proper return/provisional
// semantics, and one fresh newest read on ordinary close/open with a usable cursor.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createChatHistoryPager } from '../modules/chat_history.js';
import { messageRow } from '../modules/message_actions.js';
import { createCatchUp, roomSocket } from '../modules/room_retire.js';
import { createCallCards } from '../../skills/live_conversation/ui/call_card.js';
import { ElementStub, PREVIEW_SCOPE, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

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

const ts = (n) => new Date(Date.UTC(2026, 10, 26, 12, 0, n)).toISOString();

// Honor AbortSignal like native fetch().
function withSignal(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
    return new Promise((resolve, reject) => {
        const onAbort = () => reject(new DOMException('Aborted', 'AbortError'));
        signal.addEventListener('abort', onAbort, { once: true });
        Promise.resolve(promise).then(
            (val) => { signal.removeEventListener('abort', onAbort); resolve(val); },
            (err) => { signal.removeEventListener('abort', onAbort); reject(err); },
        );
    });
}

function makeReconnectRoom(t, { chatId = 2, hangPreferences = false, onPreferences = null, initialRows = null, hasMore = false, afterCursor = true } = {}) {
    const rows = initialRows ? [...initialRows] : [{
        role: 'assistant', chat_id: chatId, text: 'initial 1', content: 'initial 1',
        ts: ts(1), history_id: 'chat:1', history_position: { source: 'chat', offset: 1 },
    }];
    const reads = [];
    const held = new Map();
    let prefRequests = 0;
    let cursorSeq = 1;
    let emitAfterCursor = afterCursor;
    let windowOverride = null;

    const historyResponse = (address, init = {}) => {
        const cursor = address.searchParams.get('cursor');
        const mode = cursor ? (cursor.startsWith('after:') ? 'after' : 'older') : 'newest';
        reads.push({
            mode,
            cursor,
            reason: address.searchParams.get('reason') || '',
            decision: address.searchParams.get('decision') || '',
            detail: address.searchParams.get('detail') || '',
        });
        // Model server snapshot time at request start, not response release.
        const seen = mode === 'after' ? Number(cursor.split(':')[1] || 0) : 0;
        const slice = mode === 'after' ? rows.slice(seen) : [...rows];
        const nextAfter = emitAfterCursor ? `after:${rows.length}` : null;
        const latest = [...slice].reverse().find((row) => row.role === 'assistant' && !row.is_progress);
        const win = windowOverride ? (typeof windowOverride === 'function' ? windowOverride(mode, slice) : windowOverride) : {
            complete: !hasMore && mode === 'newest',
            truncated_by: hasMore ? ['quota'] : mode === 'after' ? ['page'] : [],
            ...(latest ? { latest_message: { history_id: latest.history_id, out_of_order: false } } : {}),
        };
        const pageCursor = `page:${++cursorSeq}`;
        const deliver = () => ({
            ok: true, status: 200,
            json: async () => ({
                profile_scope: PREVIEW_SCOPE,
                messages: slice,
                progress: [],
                has_more: mode === 'newest' ? hasMore : false,
                page_cursor: pageCursor,
                next_cursor: mode === 'newest' && hasMore ? 'older:1' : null,
                after_cursor: nextAfter,
                window: win,
                coverage: null,
            }),
        });
        const waiter = held.get(mode);
        if (waiter) {
            held.delete(mode);
            return withSignal(waiter.promise.then(deliver), init.signal);
        }
        return withSignal(Promise.resolve().then(deliver), init.signal);
    };

    const { prior, mount } = installDom(async (url, init = {}) => {
        const address = new URL(String(url), 'http://local');
        if (address.pathname === '/api/ui/preferences') {
            prefRequests += 1;
            if (hangPreferences) return new Promise(() => {});
            if (onPreferences) {
                const body = await onPreferences(prefRequests);
                return { ok: true, status: 200, json: async () => body };
            }
            return { ok: true, status: 200, json: async () => ({}) };
        }
        if (address.pathname === '/api/chat/history') return historyResponse(address, init);
        if (address.pathname === '/api/state') {
            return { ok: true, status: 200, json: async () => ({ active_direct_turns: [] }) };
        }
        return { ok: true, status: 200, json: async () => ({}) };
    });
    const priorSocket = globalThis.WebSocket;
    const priorQuery = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function (selector) {
        if (String(selector).startsWith(':scope >')) {
            const parts = String(selector).slice(':scope >'.length).split('>').map((s) => s.trim()).filter(Boolean);
            let cur = this;
            for (const part of parts) {
                const cls = part.startsWith('.') ? part.slice(1).split(/[ :>\[]/)[0].split('.') : [];
                cur = (cur?.children || []).find((c) => cls.length && cls.every((n) => c.classList?.contains(n))) || null;
                if (!cur) return null;
            }
            return cur;
        }
        const direct = priorQuery.call(this, selector);
        if (direct) return direct;
        for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
        return null;
    };
    globalThis.WebSocket = { OPEN: 1 };

    let connected = true, opened = true, seq = 0;
    const listeners = new Map();
    const rawWs = {
        ws: { readyState: 1 },
        on(type, fn) {
            if (!listeners.has(type)) listeners.set(type, new Set());
            listeners.get(type).add(fn);
            return () => listeners.get(type).delete(fn);
        },
        emit(type, payload) { for (const fn of [...(listeners.get(type) || [])]) fn(payload); },
        isConnected: () => connected,
        previouslyConnected: () => opened,
        serverAnswersWhileDown: () => false,
        foreignProfile: () => false,
        servingProfileScope: () => PREVIEW_SCOPE,
        canShowSavedProfile: () => true,
        send: () => ({ status: connected ? 'sent' : 'queued', clientMessageId: `cm-${++seq}` }),
    };
    const ws = roomSocket(rawWs);
    const instance = createChatInstance({
        ws,
        state: { activePage: 'chat', projectChatIds: new Set([chatId]), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: {
            begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true,
            apply() {},
            fail() {},
            latest: () => ({ supervisor_ready: true }),
        },
        chatId,
        projectId: 'proj-r2',
        idPrefix: 'chat',
        mountEl: mount,
        asPanel: true,
    });
    t.after(() => {
        instance.destroy();
        ElementStub.prototype.querySelector = priorQuery;
        globalThis.WebSocket = priorSocket;
        restoreDom(prior);
    });
    const messages = globalThis.document.byId.get('chat-messages');
    const composer = globalThis.document.byId.get('chat-input-area');
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 100, width: 100, height: bottom - top });
    messages.getBoundingClientRect = () => box(0, 20);
    composer.getBoundingClientRect = () => box(20, 60);
    const status = () => mount.querySelector('.status-badge')?.textContent || '';
    const send = (text) => {
        const input = globalThis.document.byId.get('chat-input');
        input.value = text;
        globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        return `cm-${seq}`;
    };
    const hold = (mode) => {
        let release;
        const promise = new Promise((resolve) => { release = resolve; });
        held.set(mode, { promise });
        return release;
    };
    return {
        instance, ws, rawWs, rows, reads, messages, status, send, hold,
        prefRequests: () => prefRequests,
        shown: () => messages.querySelectorAll('[data-history-id]').map((n) => n.dataset.historyId),
        pushAssistant(n, extra = {}) {
            rows.push({
                role: 'assistant', chat_id: chatId, text: `msg ${n}`, content: `msg ${n}`,
                ts: ts(n), history_id: `chat:${n}`, history_position: { source: 'chat', offset: n }, ...extra,
            });
        },
        pushUser(cmid, text, n, extra = {}) {
            rows.push({
                role: 'user', chat_id: chatId, text, content: text, client_message_id: cmid,
                ingress_accepted: true, ingress_dispatched: true,
                ts: ts(n), history_id: `chat:${n}`, history_position: { source: 'chat', offset: n }, ...extra,
            });
        },
        disconnect() { connected = false; rawWs.ws.readyState = 3; rawWs.emit('close'); },
        reconnect() { connected = true; opened = true; rawWs.ws.readyState = 1; rawWs.emit('open', { previouslyConnected: true }); },
        setAfterCursor(enabled) { emitAfterCursor = Boolean(enabled); },
        setWindow(next) { windowOverride = next; },
    };
}

test('hung preferences never stall initial open, reconnect recovery, or read receipt', async (t) => {
    const r = makeReconnectRoom(t, { hangPreferences: true });
    await until(() => r.reads.length === 1);
    await quiet();
    assert.ok(r.prefRequests() >= 1, 'initial preferences request was issued and left hanging');
    assert.equal(r.instance.hasPaintedHistory(), true, 'initial history hydrated without waiting on preferences');
    assert.deepEqual(r.reads.map((e) => e.mode), ['newest']);
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true, 'initial receipt resolves without preferences');
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Confirm across reconnect');
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    assert.equal(r.status(), 'Reconnecting…');
    const beforeReconnect = r.reads.length;
    r.pushUser(cmid, 'Confirm across reconnect', 2);
    r.pushAssistant(3);
    r.reconnect();
    const receipt = await r.instance.refreshHistory({ revision: 2 });
    await until(() => r.shown().includes('chat:3'));
    await quiet();

    assert.ok(r.prefRequests() >= 2, 'reconnect also refreshed preferences independently');
    assert.equal(receipt.painted, true, 'receipt resolves without waiting on hung preferences');
    assert.equal(receipt.read, true);
    assert.equal(r.status(), 'Starting…', 'R1 Sending -> Starting… settlement preserved');
    const afterReconnect = r.reads.slice(beforeReconnect);
    assert.equal(afterReconnect.filter((e) => e.mode === 'newest').length, 1, 'exactly one whole reconnect read, no duplicate whole');
    assert.deepEqual(afterReconnect.map(({ mode, reason, decision }) => ({ mode, reason, decision })), [
        { mode: 'newest', reason: 'reconnect', decision: 'explicit_whole' },
        { mode: 'after', reason: 'read_receipt', decision: '' },
    ]);
});

test('ordinary close/open requests one fresh newest even with a usable cursor, then resumes on-connected after', async (t) => {
    const r = makeReconnectRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();

    // On-connected arrival uses `after`.
    r.pushAssistant(2);
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
    assert.equal(r.reads.at(-1).mode, 'after');
    assert.equal(r.reads.at(-1).cursor, 'after:1');

    // Real close/open requests exactly one fresh newest even with a usable cursor.
    r.disconnect();
    r.pushAssistant(3);
    r.reconnect();
    await until(() => r.shown().includes('chat:3'));
    await quiet();
    assert.deepEqual(r.reads.slice(2).map(({ mode, reason, decision }) => ({ mode, reason, decision })), [
        { mode: 'newest', reason: 'reconnect', decision: 'explicit_whole' },
    ]);

    // Subsequent on-connected arrival resumes `after` from the cursor established by reconnect newest.
    r.pushAssistant(4);
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true);
    assert.equal(r.reads.at(-1).mode, 'after');
    assert.equal(r.reads.at(-1).cursor, 'after:3');
});

test('close aborts in-flight reads, wakes waiting receipt/showLatest immediately, and starts 0 reads while closed', async (t) => {
    const r = makeReconnectRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();

    // 1. Receipt on an already-closed socket starts 0 reads and grants no credit.
    r.disconnect();
    const beforeClosed = r.reads.length;
    const closedResult = await r.instance.refreshHistory({ revision: 10 });
    assert.equal(closedResult.painted, false);
    assert.equal(Boolean(closedResult.read), false);
    assert.equal(r.reads.length, beforeClosed, 'receipt on closed socket starts 0 history reads');

    // 2. Close aborts an in-flight `after` read cleanly with 0 fallback `newest` reads.
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    r.hold('after');
    const inflightAfter = r.instance.refreshHistory({ revision: 11 });
    await until(() => r.reads.length === 3);
    assert.equal(r.reads[2].mode, 'after');
    r.disconnect();
    const afterRes = await inflightAfter;
    assert.equal(afterRes.painted, false);
    assert.equal(r.reads.length, 3, 'aborted in-flight after read does not fall back to newest');

    // 3. Close during a held reconnect `newest` aborts it and immediately wakes waiting receipt and showLatest with 0 new reads.
    r.hold('newest');
    r.reconnect();
    await until(() => r.reads.length === 4);
    assert.equal(r.reads[3].mode, 'newest');
    const waitingReceipt = r.instance.refreshHistory({ revision: 12 });
    const waitingShowLatest = r.instance.showLatest({ arrive: true });
    let receiptSettled = false, latestSettled = false;
    void waitingReceipt.then(() => { receiptSettled = true; });
    void waitingShowLatest.then(() => { latestSettled = true; });
    await quiet();
    assert.equal(receiptSettled, false, 'receipt waits while reconnect catch-up is active');
    assert.equal(latestSettled, false, 'showLatest waits while reconnect catch-up is active');

    const countBeforeClose = r.reads.length;
    r.disconnect();
    await until(() => receiptSettled && latestSettled);
    const receiptRes = await waitingReceipt;
    assert.equal(receiptRes.painted, false);
    assert.equal(r.reads.length, countBeforeClose, 'woken receipt and showLatest on closed socket start 0 new reads');

    // 4. Subsequent reconnect recovers cleanly with one fresh newest.
    r.pushAssistant(2);
    r.reconnect();
    await until(() => r.reads.length === countBeforeClose + 1 && r.shown().includes('chat:2'));
    await quiet();
    assert.equal(r.reads.length, countBeforeClose + 1);
    assert.deepEqual(r.reads.at(-1), { mode: 'newest', cursor: null, reason: 'reconnect', decision: 'explicit_whole', detail: '' });
});

test('return shares one newest with its receipt and ordinary after preserves provisional windowReturn bubbles', async (t) => {
    const r = makeReconnectRoom(t, { hasMore: true });
    await until(() => r.reads.length === 1);
    await quiet();

    // Hide window, capture a provisional assistant message while hidden, then return on the same connection.
    const provisionalAddress = 'row:2@2026-11-26T12:00:05.000Z#0123456789ab';
    globalThis.document.hidden = true;
    r.instance.leave(true);
    r.ws.pause();
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'assistant', content: 'Provisional words', ts: ts(5), row_address: provisionalAddress,
    });
    globalThis.document.hidden = false;
    r.instance.stay();
    const provisional = r.messages.querySelector('[data-window-words]');
    assert.ok(provisional, 'windowReturn mounted provisional bubble on stay');
    // Hold focus on the provisional bubble so a bounded newest page that omits it keeps it mounted.
    globalThis.document.activeElement = provisional;

    // Return + immediate paint receipt share one newest read (no duplicate whole).
    r.ws.resume();
    const returnReceipt = await r.instance.refreshHistory({ revision: 2 });
    await quiet();
    assert.equal(returnReceipt.painted, true);
    assert.equal(returnReceipt.read, true);
    assert.equal(r.reads.slice(1).filter((e) => e.mode === 'newest').length, 1, 'return and receipt share one newest read');
    assert.equal(provisional.isConnected, true, 'bounded newest keeps held provisional bubble');

    // Keep focus held; an ordinary `after` read for another message must preserve the provisional bubble.
    r.pushAssistant(3);
    const afterReceipt = await r.instance.refreshHistory({ revision: 3 });
    await quiet();
    assert.equal(afterReceipt.painted, true);
    assert.equal(afterReceipt.read, true);
    assert.equal(r.reads.at(-1).mode, 'after');
    assert.equal(provisional.isConnected, true, 'ordinary after does not clear away provisional bubble');
    assert.ok(r.shown().includes('chat:3'));

    // When the provisional message's own canonical row arrives via `after`, it adopts the bubble in place.
    r.pushAssistant(4, { text: 'Provisional words', content: 'Provisional words', ts: ts(5), row_address: provisionalAddress });
    const adoptedReceipt = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.equal(adoptedReceipt.painted, true);
    assert.equal(adoptedReceipt.read, true);
    assert.equal(r.reads.at(-1).mode, 'after');
    assert.equal(provisional.isConnected, true, 'canonical row adopts provisional bubble in place');
    assert.equal(provisional.dataset.historyId, 'chat:4');
    assert.equal(Object.hasOwn(provisional.dataset, 'windowWords'), false);
});

test('late preferences expand displayed non-explicit child cards without re-reading and newest response wins', async (t) => {
    const prefResolvers = new Map();
    const nestedRows = [
        {
            role: 'assistant', chat_id: 2, is_progress: true, narration: true, task_id: 'parent1',
            text: 'Parent step', content: 'Parent step', ts: ts(1),
            history_id: 'progress:1', history_position: { source: 'progress', offset: 1 },
        },
        {
            role: 'assistant', chat_id: 2, is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'parent1', subagent_task_id: 'child1', task_id: 'child1',
            subagent_role: 'researcher', subagent_event: 'running',
            text: 'Child 1 running', content: 'Child 1 running', ts: ts(2),
            history_id: 'progress:2', history_position: { source: 'progress', offset: 2 },
        },
        {
            role: 'assistant', chat_id: 2, is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'parent1', subagent_task_id: 'child1', task_id: 'child1',
            subagent_role: 'researcher', narration: true,
            text: 'Child 1 note', content: 'Child 1 note', ts: ts(3),
            history_id: 'progress:3', history_position: { source: 'progress', offset: 3 },
        },
        {
            role: 'assistant', chat_id: 2, is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'child1', subagent_task_id: 'grandchild1', task_id: 'grandchild1',
            subagent_role: 'verifier', subagent_event: 'completed', result: 'Verified',
            text: 'Grandchild done', content: 'Grandchild done', ts: ts(4),
            history_id: 'progress:4', history_position: { source: 'progress', offset: 4 },
        },
        {
            role: 'assistant', chat_id: 2, is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'parent1', subagent_task_id: 'child2', task_id: 'child2',
            subagent_role: 'reviewer', subagent_event: 'running',
            text: 'Child 2 running', content: 'Child 2 running', ts: ts(5),
            history_id: 'progress:5', history_position: { source: 'progress', offset: 5 },
        },
    ];
    const r = makeReconnectRoom(t, {
        initialRows: nestedRows,
        onPreferences: (seq) => new Promise((resolve) => { prefResolvers.set(seq, resolve); }),
    });
    await until(() => r.reads.length === 1 && r.prefRequests() === 1);
    await quiet();

    // History paints immediately without waiting on preferences; subagent cards start collapsed.
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.deepEqual(r.reads.map((e) => e.mode), ['newest']);
    const parent1 = walkCard(r.messages, 'parent1');
    const child1 = walkCard(r.messages, 'child1');
    const grandchild1 = walkCard(r.messages, 'grandchild1');
    const child2 = walkCard(r.messages, 'child2');
    assert.ok(parent1 && child1 && grandchild1 && child2, 'all nested cards mounted from initial history');
    assert.equal(parent1.dataset.expanded, '0');
    assert.equal(child1.dataset.expanded, '0');
    assert.equal(grandchild1.dataset.expanded, '0');
    assert.equal(child2.dataset.expanded, '0');
    assert.equal(child1.querySelector('[data-live-disclosure]').getAttribute('aria-expanded'), 'false');
    assert.equal(child1.querySelector('[data-live-timeline]').children.length, 0, 'collapsed child defers timeline render');

    // Manual collapse on child2 before preferences return must be preserved.
    const child2Toggle = child2.querySelector('[data-live-disclosure]');
    child2Toggle.click();
    assert.equal(child2.dataset.expanded, '1');
    child2Toggle.click();
    assert.equal(child2.dataset.expanded, '0');

    // Delayed initial preferences arrive: non-explicit child & grandchild expand in place, no extra newest read.
    prefResolvers.get(1)({ nested_subagents_expanded: true });
    await quiet();
    assert.equal(parent1.dataset.expanded, '0', 'root card stays collapsed');
    assert.equal(child1.dataset.expanded, '1', 'child expands when delayed preferences arrive');
    assert.equal(grandchild1.dataset.expanded, '1', 'grandchild expands when delayed preferences arrive');
    assert.equal(child1.querySelector('[data-live-disclosure]').getAttribute('aria-expanded'), 'true');
    assert.equal(grandchild1.querySelector('[data-live-disclosure]').getAttribute('aria-expanded'), 'true');
    assert.ok(child1.querySelector('[data-live-timeline]').children.length > 0, 'expanding materializes deferred timeline');
    assert.equal(child2.dataset.expanded, '0', 'manual collapse survives arrived preferences');
    assert.equal(r.reads.length, 1, 'applying arrived preferences issues no extra history read');

    // Manual expansion on child2 must survive a later false preference; crossing responses honor the newest request.
    child2Toggle.click();
    assert.equal(child2.dataset.expanded, '1');

    r.disconnect();
    r.reconnect();
    await until(() => r.reads.length === 2 && r.prefRequests() === 2);
    r.disconnect();
    r.reconnect();
    await until(() => r.reads.length === 3 && r.prefRequests() === 3);
    await quiet();

    // Newer request #3 resolves first with false -> non-explicit children collapse; manual child2 stays expanded.
    prefResolvers.get(3)({ nested_subagents_expanded: false });
    await quiet();
    assert.equal(child1.dataset.expanded, '0');
    assert.equal(grandchild1.dataset.expanded, '0');
    assert.equal(child1.querySelector('[data-live-disclosure]').getAttribute('aria-expanded'), 'false');
    assert.equal(child2.dataset.expanded, '1', 'manual expansion survives false preference');

    // Older delayed request #2 resolves late with true -> ignored so newest request wins.
    prefResolvers.get(2)({ nested_subagents_expanded: true });
    await quiet();
    assert.equal(child1.dataset.expanded, '0', 'stale earlier preference response cannot overwrite newer one');
    assert.equal(grandchild1.dataset.expanded, '0', 'stale earlier preference response cannot overwrite newer one');
    assert.equal(child2.dataset.expanded, '1');
    assert.equal(r.reads.length, 3, 'preferences resolution issues no extra history reads');
});

test('createCatchUp wakes waiters on cancel/supersession and pager drops superseded or cancelled arrivals', async () => {
    // 1. createCatchUp waiter wake on supersession and cancellation.
    const catchUp = createCatchUp(() => {}, 50);
    catchUp.begin(() => true, new Promise(() => {}), 1, true);
    let current = true, wokeOnSupersede = false, reconnectingWoke = false;
    const settledWait = catchUp.settled(() => current).then(() => { wokeOnSupersede = true; });
    void catchUp.reconnecting.then(() => { reconnectingWoke = true; });
    current = false;
    catchUp.begin(() => true, new Promise(() => {}), 2, true);
    await settledWait;
    assert.equal(wokeOnSupersede, true, 'superseding begin wakes settled() immediately');
    assert.equal(reconnectingWoke, true, 'superseding begin wakes reconnecting waiters immediately');
    let cancelledWait = false;
    void catchUp.wait.then(() => { cancelledWait = true; });
    catchUp.cancel(2, true);
    await quiet(3);
    assert.equal(cancelledWait, true, 'cancel wakes wait immediately even when read promise never settles');
    assert.equal(catchUp.wait, null);
    assert.equal(catchUp.reconnecting, null);

    // 2. Pager cancelArrivals aborts in-flight + drops queued reads with 0 readNewest fallbacks.
    let newestCalls = 0, fetchCalls = 0, alive = true;
    const pager = createChatHistoryPager({
        fetchPage: (_cursor, { signal } = {}) => {
            fetchCalls += 1;
            return withSignal(new Promise(() => {}), signal);
        },
        applyPage() {},
        releasePage() {},
        readNewest: async () => { newestCalls += 1; return true; },
        canArrive: () => true,
        isAlive: () => true,
    });
    pager.acceptRecent({ messages: [], has_more: false, page_cursor: 'p0', after_cursor: 'after:1', window: { complete: true } });
    const first = pager.read({ current: () => alive, reason: 'read_receipt' });
    const second = pager.read({ current: () => alive, reason: 'read_receipt' });
    await quiet(3);
    alive = false;
    pager.cancelArrivals();
    assert.deepEqual(await Promise.all([first, second]), [false, false]);
    assert.equal(fetchCalls, 1);
    assert.equal(newestCalls, 0, 'cancelled arrivals never fall back to readNewest');

    // 3. Queued arrival whose caller became non-current starts 0 extra fetches when the in-flight read lands.
    let releaseFirst, secondAlive = true, seqCalls = 0, seqNewest = 0;
    const seqPager = createChatHistoryPager({
        fetchPage: () => {
            seqCalls += 1;
            return new Promise((resolve) => { releaseFirst = resolve; });
        },
        applyPage() {},
        releasePage() {},
        readNewest: async () => { seqNewest += 1; return true; },
        canArrive: () => true,
        isAlive: () => true,
    });
    seqPager.acceptRecent({ messages: [], has_more: false, page_cursor: 'p0', after_cursor: 'after:1', window: { complete: true } });
    const lead = seqPager.read({ current: () => true, reason: 'read_receipt' });
    const superseded = seqPager.read({ current: () => secondAlive, reason: 'read_receipt' });
    await quiet(3);
    secondAlive = false;
    releaseFirst({ messages: [], has_more: false, page_cursor: null, after_cursor: 'after:2', window: { complete: false, truncated_by: ['page'] } });
    assert.deepEqual(await Promise.all([lead, superseded]), [true, false]);
    assert.equal(seqCalls, 1, 'superseded queued caller starts no second after read');
    assert.equal(seqNewest, 0);
});

test('overlapping unread receipt avoids a second whole newest after reconnect heals unknown latest or omits after_cursor, while later revisions still read', async (t) => {
    // 1. Prior unknown latest_message healed by reconnect's fresh newest: overlapping receipt must not force a second whole newest.
    const r = makeReconnectRoom(t);
    r.setWindow({ complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null });
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).painted, false, 'unknown latest_message leaves revision uncovered');

    r.disconnect();
    r.pushAssistant(2);
    r.setWindow(null);  // source healed before reconnect
    const beforeHealed = r.reads.length;
    r.reconnect();
    const healed = await r.instance.refreshHistory({ revision: 2 });
    await until(() => r.shown().includes('chat:2'));
    await quiet();
    assert.deepEqual([healed.painted, healed.read], [true, true]);
    const healedReads = r.reads.slice(beforeHealed);
    assert.equal(healedReads.filter((e) => e.mode === 'newest').length, 1,
        'healed latest_unknown does not force a second whole newest');
    assert.deepEqual(healedReads.map(({ mode, reason, decision }) => ({ mode, reason, decision })), [
        { mode: 'newest', reason: 'reconnect', decision: 'explicit_whole' },
        { mode: 'after', reason: 'read_receipt', decision: '' },
    ]);

    // 2. Overlapping unread receipt begun before reconnect's HTTP snapshot when server omits after_cursor: no second whole newest.
    r.disconnect();
    r.setAfterCursor(false);
    r.pushAssistant(3);
    const beforeNoCursor = r.reads.length;
    r.reconnect();
    const noCursorReceipt = await r.instance.refreshHistory({ revision: 3 });
    await until(() => r.shown().includes('chat:3'));
    await quiet();
    assert.deepEqual([noCursorReceipt.painted, noCursorReceipt.read], [true, true]);
    assert.deepEqual(r.reads.slice(beforeNoCursor).map(({ mode, reason, decision }) => ({ mode, reason, decision })), [
        { mode: 'newest', reason: 'reconnect', decision: 'explicit_whole' },
    ], 'fresh reconnect HTTP snapshot begun after receipt satisfies it without a second whole newest');

    // 3. Unhealed unknown latest_message on a fresh reconnect begun after receipt: still no false ACK and no duplicate whole newest.
    r.disconnect();
    r.setWindow({ complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null });
    const beforeStillUnknown = r.reads.length;
    r.reconnect();
    const stillUnknown = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.equal(stillUnknown.painted, false, 'unhealed unknown latest_message grants no credit');
    assert.equal(r.reads.slice(beforeStillUnknown).filter((e) => e.mode === 'newest').length, 1,
        'unhealed reconnect snapshot begun after receipt does not immediately repeat whole newest');

    // 4. Control: a later revision arriving AFTER reconnect's HTTP snapshot already started on the wire MUST still be read.
    r.disconnect();
    r.setWindow(null);
    r.pushAssistant(4);
    const beforeLater = r.reads.length;
    const releaseReconnect = r.hold('newest');
    r.reconnect();
    await until(() => r.reads.length === beforeLater + 1);  // reconnect HTTP snapshot is now in flight with rows up to chat:4
    r.pushAssistant(5);  // revision 5 arrives after reconnect's snapshot started
    const laterPromise = r.instance.refreshHistory({ revision: 5 });
    releaseReconnect();
    const laterReceipt = await laterPromise;
    await quiet();
    assert.deepEqual([laterReceipt.painted, laterReceipt.read], [true, true]);
    assert.ok(r.shown().includes('chat:5'), 'later revision after reconnect request start is fetched before ACK');
    const laterReads = r.reads.slice(beforeLater);
    assert.equal(laterReads.filter((e) => e.mode === 'newest').length, 2,
        'without after_cursor, a later revision observed after reconnect newest began still reads newest');
});

function installCaptionTestEnv(t, feed) {
    const copied = [];
    const priorClipboard = Object.getOwnPropertyDescriptor(globalThis.navigator || {}, 'clipboard');
    Object.defineProperty(globalThis.navigator, 'clipboard', {
        configurable: true,
        value: { writeText: async (value) => { copied.push(String(value)); } },
    });
    const priorTextDesc = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'textContent');
    Object.defineProperty(ElementStub.prototype, 'textContent', {
        configurable: true,
        get() {
            const own = priorTextDesc.get.call(this);
            if (own) return own;
            if (this.classList?.contains('message')) {
                const raw = this._innerHTML
                    || /<div class="message[^"]*"[^>]*>([\s\S]*?)<\/div>/.exec(this.parentNode?._innerHTML || '')?.[1]
                    || '';
                return raw.replace(/<[^>]+>/g, '');
            }
            return '';
        },
        set(value) { priorTextDesc.set.call(this, value); },
    });
    const observers = new Set();
    const priorInsert = feed.insertBefore.bind(feed);
    feed.insertBefore = function (node, before) {
        const res = priorInsert(node, before);
        for (const cb of observers) cb();
        return res;
    };
    const doc = globalThis.document;
    const priorView = doc.defaultView;
    doc.defaultView = {
        MutationObserver: class {
            constructor(cb) { this.cb = cb; }
            observe(target) { if (target === feed) observers.add(this.cb); }
            disconnect() { observers.delete(this.cb); }
        },
    };
    t.after(() => {
        if (priorClipboard) Object.defineProperty(globalThis.navigator, 'clipboard', priorClipboard);
        else delete globalThis.navigator.clipboard;
        Object.defineProperty(ElementStub.prototype, 'textContent', priorTextDesc);
        feed.insertBefore = priorInsert;
        doc.defaultView = priorView;
    });
    return { copied };
}

test('revised live-call captions after a reconnect newest update words, row address, Copy, Share/action source, Reply, and folded call card in place without duplicating or touching unchanged rows', async (t) => {
    const callTask = 'call-t1';
    const initialRows = [
        { role: 'assistant', chat_id: 2, text: 'ordinary unchanged', content: 'ordinary unchanged', ts: ts(1),
            history_id: 'chat:1', history_position: { source: 'chat', offset: 1 }, row_address: `row:2@${ts(1)}#111111111111` },
        { role: 'system', chat_id: 2, system_type: 'call_open', source: 'live_call', client_message_id: 'call:c1:open', task_id: callTask,
            text: 'Call started. Engine: Realtime.', content: 'Call started. Engine: Realtime.', ts: ts(2),
            history_id: 'chat:2', history_position: { source: 'chat', offset: 2 }, row_address: `row:2@${ts(2)}#222222222222` },
        { role: 'user', chat_id: 2, source: 'live_call', client_message_id: 'call:c1:i1', task_id: callTask,
            text: 'draft owner', content: 'draft owner', ts: ts(3),
            history_id: 'chat:3', history_position: { source: 'chat', offset: 3 }, row_address: `row:2@${ts(3)}#aaaa00000001` },
        { role: 'assistant', chat_id: 2, source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
            spoken: { call_id: 'c1', item_id: 'i2' }, task_id: callTask,
            text: 'draft spoken', content: 'draft spoken', ts: ts(4),
            history_id: 'chat:4', history_position: { source: 'chat', offset: 4 }, row_address: `row:2@${ts(4)}#bbbb00000001` },
        { role: 'system', chat_id: 2, source: 'live_call', system_type: 'call_heard', task_id: callTask,
            text: 'Heard in the call: draft heard', content: 'Heard in the call: draft heard', ts: ts(5),
            history_id: 'chat:5', history_position: { source: 'chat', offset: 5 }, row_address: `row:2@${ts(5)}#cccc00000001` },
    ];
    const r = makeReconnectRoom(t, { initialRows });
    const { copied } = installCaptionTestEnv(t, r.messages);
    const cards = createCallCards({ feed: r.messages, doc: globalThis.document });
    t.after(() => cards.destroy());

    await until(() => r.shown().includes('chat:5'));
    await quiet();
    cards.scan();

    const byHistory = (id) => r.messages.children.filter((n) => n.dataset?.historyId === id);
    const ordinaryBefore = byHistory('chat:1')[0];
    const ownerBefore = byHistory('chat:3')[0];
    const spokenBefore = byHistory('chat:4')[0];
    const heardBefore = byHistory('chat:5')[0];
    assert.ok(ordinaryBefore && ownerBefore && spokenBefore && heardBefore);
    assert.equal(ownerBefore.classList.contains('call-folded'), true, 'call rows folded into call card');
    assert.deepEqual(
        r.messages.querySelector('.call-card').querySelectorAll('.call-line-text').map((n) => n.textContent),
        ['draft owner', 'draft spoken', 'Heard in the call: draft heard'],
        'initial folded call card shows the original captions before gap',
    );

    // Spy on ordinary unchanged bubble's .message innerHTML writes to ensure zero re-renders of ordinary rows.
    const ordinaryBody = ordinaryBefore.querySelector(':scope > .message');
    let ordinaryWrites = 0;
    const htmlDesc = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
    Object.defineProperty(ordinaryBody, 'innerHTML', {
        configurable: true,
        get() { return htmlDesc.get.call(this); },
        set(v) { ordinaryWrites += 1; htmlDesc.set.call(this, v); },
    });

    // Gap: all 3 live-call captions are revised via dialogue_store.revise_text (same history_id & ts, new text & #sha12).
    r.disconnect();
    r.rows[2] = { ...r.rows[2], text: 'corrected owner', content: 'corrected owner', row_address: `row:2@${ts(3)}#aaaa00000002` };
    r.rows[3] = { ...r.rows[3], text: 'corrected spoken', content: 'corrected spoken', row_address: `row:2@${ts(4)}#bbbb00000002` };
    r.rows[4] = { ...r.rows[4], text: 'Heard in the call: corrected heard', content: 'Heard in the call: corrected heard', row_address: `row:2@${ts(5)}#cccc00000002` };
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();

    // 1. Exact DOM node identity preserved, 0 duplicates, and ordinary unchanged bubble untouched.
    assert.deepEqual(byHistory('chat:1'), [ordinaryBefore]);
    assert.deepEqual(byHistory('chat:3'), [ownerBefore]);
    assert.deepEqual(byHistory('chat:4'), [spokenBefore]);
    assert.deepEqual(byHistory('chat:5'), [heardBefore]);
    assert.equal(ordinaryWrites, 0, 'unchanged ordinary bubble is never re-rendered');

    // 2. Words and rowAddress updated in place.
    assert.equal(ownerBefore.querySelector(':scope > .message').textContent, 'corrected owner');
    assert.equal(ownerBefore.dataset.rowAddress, `row:2@${ts(3)}#aaaa00000002`);
    assert.equal(spokenBefore.querySelector(':scope > .message').textContent, 'corrected spoken');
    assert.equal(spokenBefore.dataset.rowAddress, `row:2@${ts(4)}#bbbb00000002`);
    assert.equal(heardBefore.querySelector(':scope > .message').textContent, 'Heard in the call: corrected heard');
    assert.equal(heardBefore.dataset.rowAddress, `row:2@${ts(5)}#cccc00000002`);

    // 3. Copy uses the revised words for all 3 captions.
    ownerBefore.querySelector('.chat-message-copy').click();
    spokenBefore.querySelector('.chat-message-copy').click();
    heardBefore.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['corrected owner', 'corrected spoken', 'Heard in the call: corrected heard']);

    // 4. Share / message actions source reflects the revised rowAddress and answerWords.
    const spokenActionRow = messageRow(spokenBefore);
    assert.equal(spokenActionRow.address, `row:2@${ts(4)}#bbbb00000002`);
    assert.equal(spokenActionRow.answerWords, 'corrected spoken');
    assert.equal(spokenActionRow.text, 'corrected spoken');

    // 5. Reply uses the revised rowAddress and revised snippet.
    spokenBefore.querySelector('.chat-message-reply').click();
    const replyStored = JSON.parse(globalThis.sessionStorage.getItem(`ouro_chat_reply:${PREVIEW_SCOPE}:2`));
    assert.equal(replyStored.address, `row:2@${ts(4)}#bbbb00000002`);
    assert.equal(replyStored.snippet, 'corrected spoken');

    // 6. Folded call card refreshed automatically via MutationObserver without manual scan().
    const cardTexts = r.messages.querySelector('.call-card').querySelectorAll('.call-line-text').map((n) => n.textContent);
    assert.deepEqual(cardTexts, ['corrected owner', 'corrected spoken', 'Heard in the call: corrected heard']);
});

test('live-call captions mounted from live frames before history reconcile in place after gap revision without duplicating, while normal owner echo adoption stays untouched', async (t) => {
    const r = makeReconnectRoom(t);
    const { copied } = installCaptionTestEnv(t, r.messages);
    await until(() => r.reads.length === 1);
    await quiet();

    const callTask = 'call-t2';
    // Live WS captions arrive before any history read sees them.
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'user', source: 'live_call', client_message_id: 'call:c2:i1', task_id: callTask,
        content: 'live draft owner', ts: ts(2), row_address: `row:2@${ts(2)}#dddd00000001`,
    });
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'assistant', source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
        spoken: { call_id: 'c2', item_id: 'i2' }, task_id: callTask,
        content: 'live draft spoken', ts: ts(3), row_address: `row:2@${ts(3)}#eeee00000001`,
    });
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'system', source: 'live_call', system_type: 'call_heard', task_id: callTask,
        content: 'Heard in the call: live draft heard', ts: ts(4), row_address: `row:2@${ts(4)}#ffff00000001`,
    });
    await quiet();

    const bubblesOf = (pred) => r.messages.children.filter((n) => n.classList?.contains('chat-bubble') && pred(n));
    const liveOwner = bubblesOf((n) => n.dataset.clientMessageId === 'call:c2:i1')[0];
    const liveSpoken = bubblesOf((n) => n.classList.contains('assistant') && n.dataset.systemType === 'model_narration')[0];
    const liveHeard = bubblesOf((n) => n.dataset.systemType === 'call_heard')[0];
    assert.ok(liveOwner && liveSpoken && liveHeard);

    // Live WS revision while connected updates in place with 0 duplicates.
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'user', source: 'live_call', client_message_id: 'call:c2:i1', task_id: callTask,
        content: 'live revised owner', ts: ts(2), row_address: `row:2@${ts(2)}#dddd00000002`,
    });
    await quiet();
    assert.deepEqual(bubblesOf((n) => n.dataset.clientMessageId === 'call:c2:i1'), [liveOwner]);
    assert.equal(liveOwner.querySelector(':scope > .message').textContent, 'live revised owner');

    // Control: normal web owner echo stays intact and adopts canonical row cleanly.
    const cmid = r.send('normal web message');
    const normalOwner = bubblesOf((n) => n.dataset.clientMessageId === cmid)[0];
    assert.ok(normalOwner);

    // Gap: all 3 captions revised again before reconnect newest read.
    r.disconnect();
    r.rows.push(
        { role: 'user', chat_id: 2, source: 'live_call', client_message_id: 'call:c2:i1', task_id: callTask,
            text: 'canonical owner', content: 'canonical owner', ts: ts(2),
            history_id: 'chat:2', history_position: { source: 'chat', offset: 2 }, row_address: `row:2@${ts(2)}#dddd00000003` },
        { role: 'assistant', chat_id: 2, source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
            spoken: { call_id: 'c2', item_id: 'i2' }, task_id: callTask,
            text: 'canonical spoken', content: 'canonical spoken', ts: ts(3),
            history_id: 'chat:3', history_position: { source: 'chat', offset: 3 }, row_address: `row:2@${ts(3)}#eeee00000003` },
        { role: 'system', chat_id: 2, source: 'live_call', system_type: 'call_heard', task_id: callTask,
            text: 'Heard in the call: canonical heard', content: 'Heard in the call: canonical heard', ts: ts(4),
            history_id: 'chat:4', history_position: { source: 'chat', offset: 4 }, row_address: `row:2@${ts(4)}#ffff00000003` },
        { role: 'user', chat_id: 2, source: 'web', client_message_id: cmid, ingress_accepted: true, ingress_dispatched: true,
            text: 'normal web message', content: 'normal web message', ts: ts(5),
            history_id: 'chat:5', history_position: { source: 'chat', offset: 5 }, row_address: `row:2@${ts(5)}#999900000001` },
    );
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();

    assert.deepEqual(bubblesOf((n) => n.dataset.clientMessageId === 'call:c2:i1'), [liveOwner], 'live owner reconciled in place');
    assert.deepEqual(bubblesOf((n) => n.classList.contains('assistant') && n.dataset.systemType === 'model_narration'), [liveSpoken], 'live spoken reconciled in place');
    assert.deepEqual(bubblesOf((n) => n.dataset.systemType === 'call_heard'), [liveHeard], 'live heard reconciled in place');
    assert.deepEqual(bubblesOf((n) => n.dataset.clientMessageId === cmid), [normalOwner], 'normal web owner adopted in place');

    assert.equal(liveOwner.dataset.historyId, 'chat:2');
    assert.equal(liveOwner.querySelector(':scope > .message').textContent, 'canonical owner');
    assert.equal(liveSpoken.dataset.historyId, 'chat:3');
    assert.equal(messageRow(liveSpoken).address, `row:2@${ts(3)}#eeee00000003`);
    assert.equal(messageRow(liveSpoken).answerWords, 'canonical spoken');
    assert.equal(liveHeard.dataset.historyId, 'chat:4');
    assert.equal(liveHeard.querySelector(':scope > .message').textContent, 'Heard in the call: canonical heard');
    assert.equal(normalOwner.dataset.historyId, 'chat:5');
    assert.equal(normalOwner.dataset.rowAddress, `row:2@${ts(5)}#999900000001`);
    liveSpoken.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['canonical spoken']);
});

test('late HTTP newest snapshot does not roll back newer live WS caption corrections, while a later fresh HTTP read after a gap applies in place', async (t) => {
    const callTask = 'call-w5';
    const initialRows = [
        { role: 'assistant', chat_id: 2, text: 'ordinary unchanged', content: 'ordinary unchanged', ts: ts(1),
            history_id: 'chat:1', history_position: { source: 'chat', offset: 1 }, row_address: `row:2@${ts(1)}#101010101010` },
        { role: 'system', chat_id: 2, system_type: 'call_open', source: 'live_call', client_message_id: 'call:w5:open', task_id: callTask,
            text: 'Call started. Engine: Realtime.', content: 'Call started. Engine: Realtime.', ts: ts(2),
            history_id: 'chat:2', history_position: { source: 'chat', offset: 2 }, row_address: `row:2@${ts(2)}#202020202020` },
        { role: 'user', chat_id: 2, source: 'live_call', client_message_id: 'call:w5:line', task_id: callTask,
            text: 'Meeting at ten', content: 'Meeting at ten', ts: ts(3),
            history_id: 'chat:3', history_position: { source: 'chat', offset: 3 }, row_address: `row:2@${ts(3)}#aead00000001` },
        { role: 'assistant', chat_id: 2, source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
            spoken: { call_id: 'w5', item_id: 's1' }, task_id: callTask,
            text: 'Spoken ten', content: 'Spoken ten', ts: ts(4),
            history_id: 'chat:4', history_position: { source: 'chat', offset: 4 }, row_address: `row:2@${ts(4)}#bebe00000001` },
        { role: 'system', chat_id: 2, source: 'live_call', system_type: 'call_heard', task_id: callTask,
            text: 'Heard in the call: ten', content: 'Heard in the call: ten', ts: ts(5),
            history_id: 'chat:5', history_position: { source: 'chat', offset: 5 }, row_address: `row:2@${ts(5)}#cece00000001` },
    ];
    const r = makeReconnectRoom(t, { initialRows });
    const { copied } = installCaptionTestEnv(t, r.messages);
    const cards = createCallCards({ feed: r.messages, doc: globalThis.document });
    t.after(() => cards.destroy());

    await until(() => r.shown().includes('chat:5'));
    await quiet();
    cards.scan();

    const byHistory = (id) => r.messages.children.filter((n) => n.dataset?.historyId === id);
    const ordinaryBefore = byHistory('chat:1')[0];
    const ownerBefore = byHistory('chat:3')[0];
    const spokenBefore = byHistory('chat:4')[0];
    const heardBefore = byHistory('chat:5')[0];
    assert.ok(ordinaryBefore && ownerBefore && spokenBefore && heardBefore);

    const ordinaryBody = ordinaryBefore.querySelector(':scope > .message');
    let ordinaryWrites = 0;
    const htmlDesc = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
    Object.defineProperty(ordinaryBody, 'innerHTML', {
        configurable: true,
        get() { return htmlDesc.get.call(this); },
        set(v) { ordinaryWrites += 1; htmlDesc.set.call(this, v); },
    });
    const ordinaryCopy = ordinaryBefore.querySelector('.chat-message-copy');
    ordinaryCopy.focus();
    assert.equal(globalThis.document.activeElement, ordinaryCopy);

    // Order 1: reconnect newest HTTP snapshot begins with "ten", then live WS delivers "eleven" before HTTP resolves.
    r.disconnect();
    const releaseNewest = r.hold('newest');
    r.reconnect();
    await until(() => r.reads.some((x) => x.reason === 'reconnect' && x.mode === 'newest'));

    r.rows[2] = { ...r.rows[2], text: 'Meeting at eleven', content: 'Meeting at eleven', row_address: `row:2@${ts(3)}#aead00000002` };
    r.rows[3] = { ...r.rows[3], text: 'Spoken eleven', content: 'Spoken eleven', row_address: `row:2@${ts(4)}#bebe00000002` };
    r.rows[4] = { ...r.rows[4], text: 'Heard in the call: eleven', content: 'Heard in the call: eleven', row_address: `row:2@${ts(5)}#cece00000002` };
    r.rawWs.emit('chat', {
        type: 'chat', chat_id: 2, role: 'user', source: 'live_call', client_message_id: 'call:w5:line', task_id: callTask,
        content: 'Meeting at eleven', ts: ts(3), row_address: `row:2@${ts(3)}#aead00000002`, markdown: false, is_progress: false,
    });
    r.rawWs.emit('chat', {
        type: 'chat', chat_id: 2, role: 'assistant', source: 'live_call', system_type: 'model_narration', narration: true,
        spoken: { call_id: 'w5', item_id: 's1' }, task_id: callTask,
        content: 'Spoken eleven', ts: ts(4), row_address: `row:2@${ts(4)}#bebe00000002`, markdown: true, is_progress: false,
    });
    r.rawWs.emit('chat', {
        type: 'chat', chat_id: 2, role: 'system', source: 'live_call', system_type: 'call_heard', task_id: callTask,
        content: 'Heard in the call: eleven', ts: ts(5), row_address: `row:2@${ts(5)}#cece00000002`, markdown: false, is_progress: false,
    });
    await quiet();
    assert.equal(ownerBefore.querySelector(':scope > .message').textContent, 'Meeting at eleven');

    // Late HTTP newest snapshot ("ten") resolves after the WS corrections: must not roll back anything.
    releaseNewest();
    await quiet();

    assert.deepEqual(byHistory('chat:1'), [ordinaryBefore]);
    assert.deepEqual(byHistory('chat:3'), [ownerBefore]);
    assert.deepEqual(byHistory('chat:4'), [spokenBefore]);
    assert.deepEqual(byHistory('chat:5'), [heardBefore]);
    assert.equal(ordinaryWrites, 0, 'unrelated row is never re-rendered');
    assert.equal(globalThis.document.activeElement, ordinaryCopy, 'focus on unrelated row survives late snapshot');

    assert.equal(ownerBefore.querySelector(':scope > .message').textContent, 'Meeting at eleven');
    assert.equal(ownerBefore.dataset.rowAddress, `row:2@${ts(3)}#aead00000002`);
    assert.equal(spokenBefore.querySelector(':scope > .message').textContent, 'Spoken eleven');
    assert.equal(spokenBefore.dataset.rowAddress, `row:2@${ts(4)}#bebe00000002`);
    assert.equal(heardBefore.querySelector(':scope > .message').textContent, 'Heard in the call: eleven');
    assert.equal(heardBefore.dataset.rowAddress, `row:2@${ts(5)}#cece00000002`);

    ownerBefore.querySelector('.chat-message-copy').click();
    spokenBefore.querySelector('.chat-message-copy').click();
    heardBefore.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['Meeting at eleven', 'Spoken eleven', 'Heard in the call: eleven']);

    const spokenAction = messageRow(spokenBefore);
    assert.equal(spokenAction.address, `row:2@${ts(4)}#bebe00000002`);
    assert.equal(spokenAction.answerWords, 'Spoken eleven');
    assert.equal(spokenAction.text, 'Spoken eleven');

    spokenBefore.querySelector('.chat-message-reply').click();
    const replyEleven = JSON.parse(globalThis.sessionStorage.getItem(`ouro_chat_reply:${PREVIEW_SCOPE}:2`));
    assert.equal(replyEleven.address, `row:2@${ts(4)}#bebe00000002`);
    assert.equal(replyEleven.snippet, 'Spoken eleven');
    assert.deepEqual(
        r.messages.querySelector('.call-card').querySelectorAll('.call-line-text').map((n) => n.textContent),
        ['Meeting at eleven', 'Spoken eleven', 'Heard in the call: eleven'],
        'folded call card keeps the newer live WS captions after late HTTP resolves',
    );

    // Order 2 (reverse): a later live correction is missed during disconnect; a fresh HTTP newest begun after applies it.
    ordinaryCopy.focus();
    const readsBeforeSecond = r.reads.length;
    r.disconnect();
    r.rows[2] = { ...r.rows[2], text: 'Meeting at twelve', content: 'Meeting at twelve', row_address: `row:2@${ts(3)}#aead00000003` };
    r.rows[3] = { ...r.rows[3], text: 'Spoken twelve', content: 'Spoken twelve', row_address: `row:2@${ts(4)}#bebe00000003` };
    r.rows[4] = { ...r.rows[4], text: 'Heard in the call: twelve', content: 'Heard in the call: twelve', row_address: `row:2@${ts(5)}#cece00000003` };
    r.reconnect();
    await until(() => r.reads.length === readsBeforeSecond + 1);
    await quiet();

    assert.deepEqual(byHistory('chat:3'), [ownerBefore]);
    assert.deepEqual(byHistory('chat:4'), [spokenBefore]);
    assert.deepEqual(byHistory('chat:5'), [heardBefore]);
    assert.equal(ordinaryWrites, 0, 'unrelated row stays untouched during fresh HTTP revision');
    assert.equal(globalThis.document.activeElement, ordinaryCopy, 'focus on unrelated row stays intact');

    assert.equal(ownerBefore.querySelector(':scope > .message').textContent, 'Meeting at twelve');
    assert.equal(ownerBefore.dataset.rowAddress, `row:2@${ts(3)}#aead00000003`);
    assert.equal(spokenBefore.querySelector(':scope > .message').textContent, 'Spoken twelve');
    assert.equal(spokenBefore.dataset.rowAddress, `row:2@${ts(4)}#bebe00000003`);
    assert.equal(heardBefore.querySelector(':scope > .message').textContent, 'Heard in the call: twelve');
    assert.equal(heardBefore.dataset.rowAddress, `row:2@${ts(5)}#cece00000003`);

    copied.length = 0;
    ownerBefore.querySelector('.chat-message-copy').click();
    spokenBefore.querySelector('.chat-message-copy').click();
    heardBefore.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['Meeting at twelve', 'Spoken twelve', 'Heard in the call: twelve']);
    assert.equal(messageRow(spokenBefore).address, `row:2@${ts(4)}#bebe00000003`);
    assert.equal(messageRow(spokenBefore).answerWords, 'Spoken twelve');
    spokenBefore.querySelector('.chat-message-reply').click();
    const replyTwelve = JSON.parse(globalThis.sessionStorage.getItem(`ouro_chat_reply:${PREVIEW_SCOPE}:2`));
    assert.equal(replyTwelve.address, `row:2@${ts(4)}#bebe00000003`);
    assert.equal(replyTwelve.snippet, 'Spoken twelve');
    assert.deepEqual(
        r.messages.querySelector('.call-card').querySelectorAll('.call-line-text').map((n) => n.textContent),
        ['Meeting at twelve', 'Spoken twelve', 'Heard in the call: twelve'],
    );
});

test('late HTTP after arrival adopts canonical historyId on live-born captions without rolling back newer WS words or breaking Copy focus, and subsequent fresh HTTP still revises in place', async (t) => {
    const r = makeReconnectRoom(t);
    const { copied } = installCaptionTestEnv(t, r.messages);
    await until(() => r.reads.length === 1);
    await quiet();

    const callTask = 'call-w6';
    // Server has "ten" when the `after` read begins; hold it in flight.
    r.rows.push(
        { role: 'user', chat_id: 2, source: 'live_call', client_message_id: 'call:w6:u1', task_id: callTask,
            text: 'Draft at ten', content: 'Draft at ten', ts: ts(2),
            history_id: 'chat:2', history_position: { source: 'chat', offset: 2 }, row_address: `row:2@${ts(2)}#dada00000001` },
        { role: 'assistant', chat_id: 2, source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
            spoken: { call_id: 'w6', item_id: 'a1' }, task_id: callTask,
            text: 'Spoken at ten', content: 'Spoken at ten', ts: ts(3),
            history_id: 'chat:3', history_position: { source: 'chat', offset: 3 }, row_address: `row:2@${ts(3)}#dbdb00000001` },
    );
    const releaseAfter = r.hold('after');
    const receiptPromise = r.instance.refreshHistory({ revision: 2 });
    await until(() => r.reads.some((x) => x.mode === 'after'));

    // While `after` is in flight:
    // - user caption mounts live at "ten" and is revised live to "eleven"
    // - assistant caption mounts live directly at "eleven" (first client paint is already newer than the in-flight snapshot)
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'user', source: 'live_call', client_message_id: 'call:w6:u1', task_id: callTask,
        content: 'Draft at ten', ts: ts(2), row_address: `row:2@${ts(2)}#dada00000001`,
    });
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'user', source: 'live_call', client_message_id: 'call:w6:u1', task_id: callTask,
        content: 'Revised at eleven', ts: ts(2), row_address: `row:2@${ts(2)}#dada00000002`,
    });
    r.rawWs.emit('chat', {
        chat_id: 2, role: 'assistant', source: 'live_call', system_type: 'model_narration', narration: true, markdown: true,
        spoken: { call_id: 'w6', item_id: 'a1' }, task_id: callTask,
        content: 'Spoken at eleven', ts: ts(3), row_address: `row:2@${ts(3)}#dbdb00000002`,
    });
    await quiet();

    const bubblesOf = (pred) => r.messages.children.filter((n) => n.classList?.contains('chat-bubble') && pred(n));
    const liveOwner = bubblesOf((n) => n.dataset.clientMessageId === 'call:w6:u1')[0];
    const liveSpoken = bubblesOf((n) => n.classList.contains('assistant') && n.dataset.systemType === 'model_narration')[0];
    assert.ok(liveOwner && liveSpoken);

    const ownerCopy = liveOwner.querySelector('.chat-message-copy');
    ownerCopy.focus();
    assert.equal(globalThis.document.activeElement, ownerCopy);

    // Late `after` snapshot ("ten") resolves: adopts canonical historyIds with 0 duplicates and keeps "eleven".
    releaseAfter();
    assert.equal((await receiptPromise).read, true);
    await quiet();

    assert.deepEqual(bubblesOf((n) => n.dataset.clientMessageId === 'call:w6:u1'), [liveOwner], 'zero duplicate owner bubbles');
    assert.deepEqual(bubblesOf((n) => n.classList.contains('assistant') && n.dataset.systemType === 'model_narration'), [liveSpoken], 'zero duplicate spoken bubbles');
    assert.equal(liveOwner.dataset.historyId, 'chat:2');
    assert.equal(liveOwner.querySelector(':scope > .message').textContent, 'Revised at eleven');
    assert.equal(liveOwner.dataset.rowAddress, `row:2@${ts(2)}#dada00000002`);
    assert.equal(liveSpoken.dataset.historyId, 'chat:3');
    assert.equal(liveSpoken.querySelector(':scope > .message').textContent, 'Spoken at eleven');
    assert.equal(liveSpoken.dataset.rowAddress, `row:2@${ts(3)}#dbdb00000002`);
    assert.equal(messageRow(liveSpoken).answerWords, 'Spoken at eleven');
    assert.equal(globalThis.document.activeElement, ownerCopy, 'focus on Copy stays untouched when stale after is ignored');

    ownerCopy.click();
    liveSpoken.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['Revised at eleven', 'Spoken at eleven']);

    // Reverse order: disconnect gap -> server revises to "twelve" -> fresh reconnect newest applies in place and keeps Copy focus.
    ownerCopy.focus();
    const newestBefore = r.reads.filter((x) => x.mode === 'newest').length;
    r.disconnect();
    r.rows[1] = { ...r.rows[1], text: 'Final at twelve', content: 'Final at twelve', row_address: `row:2@${ts(2)}#dada00000003` };
    r.rows[2] = { ...r.rows[2], text: 'Spoken at twelve', content: 'Spoken at twelve', row_address: `row:2@${ts(3)}#dbdb00000003` };
    r.reconnect();
    await until(() => r.reads.filter((x) => x.mode === 'newest').length === newestBefore + 1);
    await quiet();

    assert.deepEqual(bubblesOf((n) => n.dataset.clientMessageId === 'call:w6:u1'), [liveOwner]);
    assert.deepEqual(bubblesOf((n) => n.classList.contains('assistant') && n.dataset.systemType === 'model_narration'), [liveSpoken]);
    assert.equal(liveOwner.querySelector(':scope > .message').textContent, 'Final at twelve');
    assert.equal(liveOwner.dataset.rowAddress, `row:2@${ts(2)}#dada00000003`);
    assert.equal(liveSpoken.querySelector(':scope > .message').textContent, 'Spoken at twelve');
    assert.equal(liveSpoken.dataset.rowAddress, `row:2@${ts(3)}#dbdb00000003`);
    assert.equal(messageRow(liveSpoken).answerWords, 'Spoken at twelve');
    assert.equal(globalThis.document.activeElement, liveOwner.querySelector('.chat-message-copy'),
        'focus stays on Copy button when fresh HTTP revises bubble in place');

    copied.length = 0;
    liveOwner.querySelector('.chat-message-copy').click();
    liveSpoken.querySelector('.chat-message-copy').click();
    await quiet();
    assert.deepEqual(copied, ['Final at twelve', 'Spoken at twelve']);
});
