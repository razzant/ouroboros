// W2 R1: disconnected send reconciliation through the real chat instance.
// Only a matching local owner row with positive dispatch evidence (or an existing
// authoritative receipt) in a successfully DRAWN, qualifying post-disconnect source
// response may settle Sending into Starting….
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createCatchUp, roomSocket } from '../modules/room_retire.js';
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

function makeRoom(t, { chatId = 2, asPanel = true, initialOnline = true, afterCursor = null } = {}) {
    const queue = [];
    const reads = [];
    let pageSeq = 0;
    let defaultBody = {
        profile_scope: PREVIEW_SCOPE,
        messages: [],
        progress: [],
        has_more: false,
        page_cursor: 'page:0',
        next_cursor: null,
        after_cursor: afterCursor,
        window: { complete: true, truncated_by: [] },
        coverage: null,
    };
    const { prior, mount } = installDom(async (url) => {
        const address = new URL(String(url), 'http://local');
        if (address.pathname === '/api/chat/history') {
            const cursor = address.searchParams.get('cursor');
            const mode = cursor ? (cursor.startsWith('after:') ? 'after' : 'older') : 'newest';
            reads.push({ mode, cursor, reason: address.searchParams.get('reason') || '',
                decision: address.searchParams.get('decision') || '' });
            const next = queue.length ? queue.shift() : defaultBody;
            const resolved = typeof next === 'function' ? await next(address, mode) : await next;
            if (resolved?.httpStatus) {
                return { ok: false, status: resolved.httpStatus, json: async () => resolved.body || {} };
            }
            return { ok: true, status: 200, json: async () => ({ profile_scope: PREVIEW_SCOPE, ...resolved }) };
        }
        if (address.pathname === '/api/state') {
            return { ok: true, status: 200, json: async () => ({ active_direct_turns: [] }) };
        }
        return { ok: true, status: 200, json: async () => ({}) };
    });
    const priorSocket = globalThis.WebSocket;
    const priorQuery = ElementStub.prototype.querySelector;
    // Allow nested lookups (status badge, bubble notes) in the flat DOM stub.
    ElementStub.prototype.querySelector = function (selector) {
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
        state: { activePage: 'chat', projectChatIds: new Set(asPanel ? [chatId] : []), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: {
            begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true,
            apply() {},
            fail() {},
            latest: () => (initialOnline ? { supervisor_ready: true } : null),
        },
        chatId,
        projectId: asPanel ? 'proj-r1' : '',
        idPrefix: 'chat',
        mountEl: mount,
        asPanel,
    });
    t.after(() => {
        instance.destroy();
        ElementStub.prototype.querySelector = priorQuery;
        globalThis.WebSocket = priorSocket;
        restoreDom(prior);
    });
    const status = () => mount.querySelector('.status-badge')?.textContent || '';
    const messages = globalThis.document.byId.get('chat-messages');
    const send = (text) => {
        const input = globalThis.document.byId.get('chat-input');
        input.value = text;
        globalThis.document.byId.get('chat-send').listeners.get('click')[0]();
        return `cm-${seq}`;
    };
    const bubble = (cmid) => messages.querySelectorAll('.chat-bubble.user').find((n) => n.dataset.clientMessageId === cmid) || null;
    const sessionIdOf = (cmid) => bubble(cmid)?.dataset.senderSessionId || '';
    const userRow = (cmid, text, extra = {}) => ({
        role: 'user',
        chat_id: chatId,
        source: 'web',
        text,
        content: text,
        client_message_id: cmid,
        ingress_accepted: true,
        ingress_dispatched: true,
        ts: isoAt(1),
        history_id: `chat:${cmid}`,
        history_position: { source: 'chat', offset: 10 + seq },
        ...extra,
    });
    const assistantRow = (id, text, offset = 1) => ({
        role: 'assistant',
        chat_id: chatId,
        text,
        content: text,
        ts: isoAt(10 + offset),
        history_id: id,
        history_position: { source: 'chat', offset: 100 + offset },
    });
    const page = (rows, extra = {}) => ({
        messages: rows,
        progress: [],
        has_more: false,
        page_cursor: `page:${++pageSeq}`,
        next_cursor: null,
        after_cursor: afterCursor ? `after:${pageSeq}` : null,
        window: { complete: true, truncated_by: [] },
        coverage: null,
        ...extra,
    });
    // Trigger a one-shot rendering failure inside applyHistoryMessages when painting `historyId`.
    const armDrawFault = (historyId) => {
        let hits = 0;
        const orig = messages.insertBefore.bind(messages);
        messages.insertBefore = function (node, before) {
            if (!hits && node?.dataset?.historyId === historyId) {
                hits += 1;
                messages.insertBefore = orig;
                throw new Error('transient render failure');
            }
            return orig(node, before);
        };
        return () => hits;
    };
    return {
        instance, ws, rawWs, reads, messages, status, send, bubble, sessionIdOf, userRow, assistantRow, page, armDrawFault,
        shown: () => messages.querySelectorAll('[data-history-id]').map((n) => n.dataset.historyId),
        push: (...items) => queue.push(...items),
        setDefault: (body) => { defaultBody = body; },
        disconnect() { connected = false; rawWs.ws.readyState = 3; rawWs.emit('close'); },
        reconnect() { connected = true; opened = true; rawWs.ws.readyState = 1; rawWs.emit('open', { previouslyConnected: true }); },
    };
}

test('R1 reproducer: pending saved echo pushed out of bounded 150-row newest never falsely settles Sending', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Stored but not yet dispatched');
    await quiet();
    assert.equal(r.status(), 'Sending');

    // Host echoes this local user row as saved while dispatch is still pending (no dispatch yet).
    const sessionId = r.sessionIdOf(cmid);
    r.rawWs.emit('chat', {
        role: 'user', chat_id: 2, source: 'web', content: 'Stored but not yet dispatched', text: 'Stored but not yet dispatched',
        client_message_id: cmid, sender_session_id: sessionId,
        row_address: 'row:2@2026-11-26T10:00:01Z#aaaaaaaaaaaa',
        ingress_accepted: true, ingress_pending: true, ts: isoAt(1),
    });
    await quiet();
    assert.equal(r.status(), 'Sending', 'saved-but-pending echo keeps Sending');

    r.disconnect();
    assert.equal(r.status(), 'Reconnecting…');

    // 150 later independent rows fill the bounded newest quota; our send is absent from the response.
    const filler = Array.from({ length: 150 }, (_, i) => r.assistantRow(`chat:fill-${i + 1}`, `Independent ${i + 1}`, i + 1));
    r.push(r.page(filler, {
        has_more: true, next_cursor: 'older:150', window: { complete: false, truncated_by: ['quota'] },
    }));
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();

    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.ok(r.shown().includes('chat:fill-150'));
    assert.equal(r.status(), 'Sending', 'absent send must remain Sending even after an earlier saved-pending echo');

    // Existing authoritative typing receipt still settles it normally.
    r.rawWs.emit('typing', { chat_id: 2, activity_id: 'turn-repro', kind: 'direct_chat', client_message_id: cmid });
    await quiet();
    assert.equal(r.status(), 'Starting…');
});

test('R1: still-pending and old-server rows without positive dispatch evidence do not guess; authoritative receipt settles', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();

    const cmid = r.send('Need positive dispatch');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    // 1. Matching row still pending dispatch -> stays Sending.
    // 2. Matching row with ingress_accepted only (ended process / old server lacking ingress_dispatched) -> stays Sending.
    // 3. Matching row lacking all ingress fields (older server) -> stays Sending.
    // 4. Same old-server row with durable chat_annotation (existing authoritative receipt) -> settles.
    r.push(
        r.page([r.userRow(cmid, 'Need positive dispatch', {
            sender_session_id: sessionId, ingress_accepted: true, ingress_dispatched: undefined, ingress_pending: true,
        })]),
        r.page([r.userRow(cmid, 'Need positive dispatch', {
            sender_session_id: sessionId, ingress_accepted: true, ingress_dispatched: undefined,
        })]),
        r.page([r.userRow(cmid, 'Need positive dispatch', {
            sender_session_id: sessionId, ingress_accepted: undefined, ingress_dispatched: undefined,
        })]),
        r.page([r.userRow(cmid, 'Need positive dispatch', {
            sender_session_id: sessionId, ingress_accepted: undefined, ingress_dispatched: undefined,
            chat_annotation: { annotation_type: 'routing_ack', action: 'new_task', status: 'routed' },
        })]),
    );

    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(r.status(), 'Sending', 'still-pending row never settles Sending');

    await r.instance.refreshHistory({ revision: 2 });
    await quiet();
    assert.equal(r.status(), 'Sending', 'ingress_accepted alone without ingress_dispatched must not guess dispatch');

    await r.instance.refreshHistory({ revision: 3 });
    await quiet();
    assert.equal(r.status(), 'Sending', 'old-server row lacking ingress fields must not guess dispatch');

    await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.equal(r.status(), 'Starting…', 'existing durable routing receipt still settles old-server row');
});

test('R1: same-id-but-not-ours rows never settle Sending; next qualifying read with our dispatched row does', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();

    const cmid = r.send('Mine only');
    const sessionId = r.sessionIdOf(cmid);
    const ownAddress = 'row:2@2026-11-26T10:00:01Z#aaaaaaaaaaaa';
    r.rawWs.emit('chat', {
        role: 'user', chat_id: 2, source: 'web', content: 'Mine only', text: 'Mine only',
        client_message_id: cmid, sender_session_id: sessionId, row_address: ownAddress,
        ingress_accepted: true, ingress_pending: true, ts: isoAt(1),
    });
    await quiet();

    r.disconnect();
    const impostors = [
        r.userRow(cmid, 'Projected copy', { history_id: 'chat:imp-1', origin_projected: true, sender_session_id: sessionId, row_address: ownAddress }),
        r.userRow(cmid, 'Other room', { history_id: 'chat:imp-2', chat_id: 99, sender_session_id: sessionId, row_address: 'row:99@2026-11-26T10:00:01Z#aaaaaaaaaaaa' }),
        r.userRow(cmid, 'Agent note', { history_id: 'chat:imp-3', declared_author: 'agent', sender_session_id: sessionId, row_address: ownAddress }),
        r.userRow(cmid, 'Telegram row', { history_id: 'chat:imp-4', source: 'telegram', sender_session_id: sessionId, row_address: ownAddress }),
        r.userRow(cmid, 'Other tab', { history_id: 'chat:imp-5', sender_session_id: 'other-tab-session', row_address: ownAddress }),
        r.userRow(cmid, 'Other physical row', { history_id: 'chat:imp-6', sender_session_id: sessionId, row_address: 'row:2@2026-11-26T10:00:02Z#bbbbbbbbbbbb' }),
        { ...r.assistantRow('chat:imp-7', 'Assistant with same id', 2), client_message_id: cmid, ingress_accepted: true, ingress_dispatched: true },
        r.userRow(cmid, 'Progress echo', { history_id: 'chat:imp-8', is_progress: true, sender_session_id: sessionId, row_address: ownAddress }),
        r.userRow(cmid, 'Pending despite dispatch flag', { history_id: 'chat:imp-9', ingress_accepted: true, ingress_dispatched: true, ingress_pending: true, sender_session_id: sessionId, row_address: ownAddress }),
    ];
    const ours = r.userRow(cmid, 'Mine only', { sender_session_id: sessionId, row_address: ownAddress });
    r.push(r.page(impostors), r.page([ours]));

    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(r.status(), 'Sending', 'same-client_message_id rows that are not ours must not settle Sending');

    await r.instance.refreshHistory({ revision: 2 });
    await quiet();
    assert.equal(r.status(), 'Starting…', 'subsequent qualifying read carrying our matching dispatched row settles Sending');
});

test('R1: failed, cut, and source_unavailable reads never settle; unrelated receipt Retry settles on clean newest', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Ship R1 safely');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    assert.equal(r.status(), 'Reconnecting…');
    assert.ok(r.bubble(cmid)?.querySelector('[data-ingress-unconfirmed]'), 'disconnect marks unsaved bubble with Send again doubt');

    const dispatched = r.userRow(cmid, 'Ship R1 safely', { sender_session_id: sessionId });
    r.push(
        { httpStatus: 503, body: { error: 'store unavailable' } },
        r.page([dispatched], { window: { complete: false, truncated_by: ['quota', 'chat_projection_failed'] } }),
        r.page([dispatched], { window: { complete: false, truncated_by: ['chat_source_unavailable'] } }),
        r.page([dispatched]),
    );
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(r.status(), 'Sending', 'HTTP 503 reconnect read does not settle Sending');

    const cut = await r.instance.refreshHistory({ revision: 2, why: { reason: 'retry' } });
    await quiet();
    assert.equal(cut.painted, false);
    assert.equal(r.status(), 'Sending', 'chat_projection_failed read must not settle Sending');

    const unavail = await r.instance.refreshHistory({ revision: 3, why: { reason: 'retry' } });
    await quiet();
    assert.equal(unavail.painted, false);
    assert.equal(r.status(), 'Sending', 'chat_source_unavailable read must not settle Sending');

    const recovered = await r.instance.refreshHistory({ revision: 4, why: { reason: 'retry' } });
    await quiet();
    assert.equal(recovered.painted, true);
    assert.equal(r.status(), 'Starting…', 'qualifying Retry settles Sending into Starting…');
    assert.equal(r.bubble(cmid)?.querySelector('[data-ingress-unconfirmed]'), null, 'confirmed row clears Send again doubt');
    assert.ok(r.bubble(cmid)?.querySelector('[data-ingress-saved]'), 'confirmed row marks Input saved');
    assert.deepEqual(r.reads.map((entry) => entry.mode), ['newest', 'newest', 'newest', 'newest', 'newest']);
});

test('R1 draw-before-confirm: transient render fault on syncHistory first-load branch (applyPage recent) keeps Sending until Retry paints', async (t) => {
    const r = makeRoom(t, { initialOnline: false });
    r.push({ httpStatus: 503, body: { error: 'booting' } });
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.instance.hasPaintedHistory(), false);

    const cmid = r.send('First load draw fault');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    const reply = r.assistantRow('chat:reply-first', 'Answer after boot', 1);
    const saved = r.userRow(cmid, 'First load draw fault', { sender_session_id: sessionId });
    r.push(r.page([reply, saved]), r.page([reply, saved]));
    const faults = r.armDrawFault('chat:reply-first');

    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(faults(), 1, 'transient draw fault fired inside applyHistoryMessages on first-load applyPage');
    assert.equal(r.instance.hasPaintedHistory(), false, 'draw-faulted first load never marks history painted');
    assert.equal(r.shown().includes('chat:reply-first'), false);
    assert.equal(r.status(), 'Sending', 'draw-faulted response bytes must not clear confirmation or settle Sending');

    const retried = await r.instance.refreshHistory({ revision: 2, why: { reason: 'retry' } });
    await quiet();
    assert.equal(retried.painted, true);
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.ok(r.shown().includes('chat:reply-first'));
    assert.equal(r.status(), 'Starting…', 'Retry after transient first-load draw fault paints and settles Sending');
});

test('R1 draw-before-confirm: transient render fault on syncHistory non-rechained branch keeps Sending until Retry paints', async (t) => {
    const r = makeRoom(t);
    const seed = r.assistantRow('chat:seed', 'Seed message', 0);
    r.push(r.page([seed]));
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Non-rechained draw fault');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    // Retain `seed` in the response so oldRecentIds stay present and syncHistory takes the non-rechained branch.
    const reply = r.assistantRow('chat:reply-nonrechain', 'Fresh reply', 2);
    const saved = r.userRow(cmid, 'Non-rechained draw fault', { sender_session_id: sessionId });
    r.push(r.page([seed, reply, saved]), r.page([seed, reply, saved]));
    const faults = r.armDrawFault('chat:reply-nonrechain');

    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(faults(), 1, 'transient draw fault fired inside syncHistory non-rechained branch');
    assert.equal(r.instance.hasPaintedHistory(), false);
    assert.equal(r.shown().includes('chat:reply-nonrechain'), false);
    assert.equal(r.status(), 'Sending', 'draw-faulted non-rechained syncHistory must not settle Sending');

    const retried = await r.instance.refreshHistory({ revision: 2, why: { reason: 'retry' } });
    await quiet();
    assert.equal(retried.painted, true);
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.ok(r.shown().includes('chat:reply-nonrechain'));
    assert.equal(r.status(), 'Starting…', 'Retry after transient non-rechained draw fault paints and settles Sending');
});

test('R1 draw-before-confirm: transient render fault on historyPager.applyPage (latest via showLatest) keeps Sending until Retry paints', async (t) => {
    const r = makeRoom(t);
    const seed = r.assistantRow('chat:seed-latest', 'Seed message', 0);
    r.push(r.page([seed]));
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Latest applyPage draw fault');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    // Reconnect read fails (503) so catchUp.owed stays true with no syncHistory in flight; ↓ showLatest takes historyPager.latest -> applyPage.
    const reply = r.assistantRow('chat:reply-latest', 'Reply via latest', 2);
    const saved = r.userRow(cmid, 'Latest applyPage draw fault', { sender_session_id: sessionId });
    r.push(
        { httpStatus: 503, body: { error: 'unavailable' } },
        r.page([seed, reply, saved]),
        r.page([seed, reply, saved]),
    );
    r.reconnect();
    await until(() => r.reads.length === 2);
    await quiet();
    assert.equal(r.status(), 'Sending');

    const faults = r.armDrawFault('chat:reply-latest');
    await r.instance.showLatest({ arrive: true });
    await quiet();
    assert.equal(r.reads.length, 3);
    assert.equal(r.reads[2].reason, 'navigate', 'showLatest drove historyPager.latest -> applyPage directly');
    assert.equal(faults(), 1, 'transient draw fault fired inside historyPager.applyPage');
    assert.equal(r.shown().includes('chat:reply-latest'), false);
    assert.equal(r.status(), 'Sending', 'draw-faulted applyPage(latest) must not clear confirmation or settle Sending');

    await r.instance.showLatest({ arrive: true });
    await quiet();
    assert.equal(r.reads.length, 4);
    assert.equal(r.reads[3].reason, 'navigate');
    assert.ok(r.shown().includes('chat:reply-latest'));
    assert.equal(r.instance.hasPaintedHistory(), true);
    assert.equal(r.status(), 'Starting…', 'Retry via showLatest paints and settles Sending into Starting…');
});

test('R1: same-socket cut-only Retry without a real reconnect must not settle a live send', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    const cmid = r.send('Still on live socket');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    const row = r.userRow(cmid, 'Still on live socket', { sender_session_id: sessionId });
    r.push(
        r.page([row], { window: { complete: false, truncated_by: ['chat_projection_failed'] } }),
        r.page([row]),
    );
    const cut = await r.instance.refreshHistory({ revision: 2 });
    await quiet();
    assert.equal(cut.painted, false);
    assert.equal(r.status(), 'Sending');

    const retried = await r.instance.refreshHistory({ revision: 3, why: { reason: 'retry' } });
    await quiet();
    assert.equal(retried.painted, true);
    assert.equal(r.status(), 'Sending', 'same-socket cut-only Retry must never settle a live send');

    r.rawWs.emit('typing', { chat_id: 2, activity_id: 'turn-live', kind: 'direct_chat', client_message_id: cmid });
    await quiet();
    assert.equal(r.status(), 'Online');
});

test('R1: pre-close late read, offline-queued send flushed on open, and new send born on new socket are never falsely settled', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();

    // Pre-disconnect send with an in-flight read started BEFORE close.
    const oldSend = r.send('Before close');
    const sessionId = r.sessionIdOf(oldSend);
    let releaseLate;
    r.push(new Promise((resolve) => { releaseLate = resolve; }));
    const lateRefresh = r.instance.refreshHistory({ revision: 2 });
    await until(() => r.reads.length === 2);

    r.disconnect();
    const queuedSend = r.send('Queued offline');
    assert.equal(r.status(), 'Reconnecting…');

    let releasePost;
    r.push(new Promise((resolve) => { releasePost = resolve; }));
    r.reconnect();
    // Offline queue flushes on new socket, and a brand-new send is born on the new socket before reconnect newest lands.
    r.rawWs.emit('outbound_sent', { clientMessageId: queuedSend });
    const freshSend = r.send('Born on new socket');

    const postRow = r.userRow(oldSend, 'Before close', { sender_session_id: sessionId });
    const queuedRow = r.userRow(queuedSend, 'Queued offline', { sender_session_id: sessionId });
    const freshRow = r.userRow(freshSend, 'Born on new socket', { sender_session_id: sessionId });

    // Late pre-disconnect response lands carrying dispatched oldSend -> must NOT settle.
    releaseLate(r.page([postRow]));
    await lateRefresh;
    await until(() => r.reads.length === 3);
    await quiet();
    assert.equal(r.status(), 'Sending', 'pre-close late read cannot confirm a disconnected send');

    // Post-reconnect read settles oldSend, but MUST NOT settle queuedSend or freshSend.
    releasePost(r.page([postRow, queuedRow, freshRow]));
    await quiet();
    assert.equal(r.status(), 'Sending', 'queued-then-flushed and new-socket sends stay Sending until authoritative receipt');

    r.rawWs.emit('typing', { chat_id: 2, activity_id: 'turn-q', kind: 'direct_chat', client_message_id: queuedSend });
    await quiet();
    assert.equal(r.status(), 'Sending', 'new-socket send still awaits its own authoritative receipt');

    r.rawWs.emit('typing', { chat_id: 2, activity_id: 'turn-f', kind: 'direct_chat', client_message_id: freshSend });
    await quiet();
    assert.equal(r.status(), 'Starting…', 'once flushed and new-socket sends get receipts, room transitions to Starting…');
});

test('R1: already-connected chat_source_unavailable still acks paint without settling an earlier disconnected send', async (t) => {
    const r = makeRoom(t);
    await until(() => r.reads.length === 1);
    await quiet();
    assert.equal(r.status(), 'Online');

    // Already-connected window with partial source availability still satisfies general read/paint ACK.
    r.push(r.page([], { window: { complete: false, truncated_by: ['chat_source_unavailable'] } }));
    const connectedPartial = await r.instance.refreshHistory({ revision: 2 });
    await quiet();
    assert.equal(connectedPartial.painted, true, 'normal non-disconnected chat_source_unavailable keeps painted ACK');
    assert.equal(r.status(), 'Online');

    // A disconnected send omitted from a bounded reconnect read stays pending even after the window is connected again;
    // a subsequent connected chat_source_unavailable read still acks paint but must not settle that send.
    const cmid = r.send('Needs clean newest');
    const sessionId = r.sessionIdOf(cmid);
    await quiet();
    assert.equal(r.status(), 'Sending');

    r.disconnect();
    r.push(r.page([r.assistantRow('chat:other', 'Other row', 1)], {
        has_more: true, next_cursor: 'older:1', window: { complete: false, truncated_by: ['quota'] },
    }));
    r.reconnect();
    await until(() => r.reads.length === 3);
    await quiet();
    assert.equal(r.status(), 'Sending', 'omitted send remains Sending after bounded reconnect');

    const dispatched = r.userRow(cmid, 'Needs clean newest', { sender_session_id: sessionId });
    r.push(
        r.page([dispatched], { window: { complete: false, truncated_by: ['chat_source_unavailable'] } }),
        r.page([dispatched]),
    );
    const partialWithRow = await r.instance.refreshHistory({ revision: 3 });
    await quiet();
    assert.equal(partialWithRow.painted, true, 'connected chat_source_unavailable still acks paint');
    assert.equal(r.status(), 'Sending', 'degraded connected read must not settle a disconnected send');

    const clean = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.equal(clean.painted, true);
    assert.equal(r.status(), 'Starting…', 'clean dispatched newest settles Sending into Starting…');
});

test('R1 unit: createCatchUp distinguishes connected partial reads from post-disconnect settlement', () => {
    const pending = new Map();
    const catchUp = createCatchUp(() => {}, 50, pending);
    assert.equal('needsWhole' in catchUp, false);
    assert.equal('noteSaved' in catchUp, false);
    assert.equal('confirmArrivals' in catchUp, false);

    catchUp.trackSend('live-1', false);
    // Already-connected chat_source_unavailable keeps owed false (general paint contract) and never settles live-1.
    assert.equal(catchUp.wouldConfirm({ recentVersion: 1, window: { truncated_by: ['chat_source_unavailable'] } }), true);
    assert.equal(catchUp.confirmWindow({ recentVersion: 1, window: { truncated_by: ['chat_source_unavailable'] } }, ['live-1']), true);
    assert.equal(catchUp.owed, false);
    assert.equal(pending.has('live-1'), true);

    // Cut read without disconnect owes repair, not send settlement.
    assert.equal(catchUp.wouldConfirm({ recentVersion: 2, window: { truncated_by: ['chat_projection_failed'] } }), false);
    assert.equal(catchUp.confirmWindow({ recentVersion: 2, window: { truncated_by: ['chat_projection_failed'] } }, ['live-1']), false);
    assert.equal(catchUp.owed, true);
    assert.equal(pending.has('live-1'), true);
    // Clean read without real disconnect repairs cut and keeps live-1.
    assert.equal(catchUp.wouldConfirm({ recentVersion: 3, window: { truncated_by: [] } }), true);
    assert.equal(catchUp.confirmWindow({ recentVersion: 3, window: { truncated_by: [] } }, ['live-1']), true);
    assert.equal(catchUp.owed, false);
    assert.equal(pending.has('live-1'), true);

    // Real disconnect marks live-1; pre-close version, degraded reads, and absent confirmation never settle it.
    catchUp.cancel(3, true);
    assert.equal(catchUp.confirmWindow({ recentVersion: 3, window: { truncated_by: [] } }, ['live-1']), true);
    assert.equal(catchUp.owed, true);
    assert.equal(pending.has('live-1'), true, 'pre-close version cannot settle');
    assert.equal(catchUp.wouldConfirm({ recentVersion: 4, window: { truncated_by: ['chat_source_unavailable'] } }), false);
    assert.equal(catchUp.confirmWindow({ recentVersion: 4, window: { truncated_by: ['chat_source_unavailable'] } }, ['live-1']), false);
    assert.equal(catchUp.owed, true);
    assert.equal(pending.has('live-1'), true, 'disconnected chat_source_unavailable cannot confirm or settle');
    assert.equal(catchUp.wouldConfirm({ recentVersion: 5, window: { truncated_by: ['progress_source_unavailable'] } }), false);
    assert.equal(catchUp.confirmWindow({ recentVersion: 5, window: { truncated_by: ['progress_source_unavailable'] } }, ['live-1']), false);
    assert.equal(catchUp.owed, true);
    assert.equal(pending.has('live-1'), true, 'disconnected progress_source_unavailable cannot confirm or settle');
    assert.equal(catchUp.confirmWindow({ recentVersion: 6, window: { truncated_by: ['quota'] } }, []), true);
    assert.equal(catchUp.owed, false);
    assert.equal(pending.has('live-1'), true, 'absent row in bounded newest cannot settle');

    // Once connected again, a degraded read still acks without settling live-1; a clean read settles it.
    assert.equal(catchUp.wouldConfirm({ recentVersion: 7, window: { truncated_by: ['chat_source_unavailable'] } }), true);
    assert.equal(catchUp.confirmWindow({ recentVersion: 7, window: { truncated_by: ['chat_source_unavailable'] } }, ['live-1']), true);
    assert.equal(catchUp.owed, false);
    assert.equal(pending.has('live-1'), true, 'connected degraded read cannot settle disconnected send');

    assert.equal(catchUp.confirmWindow({ recentVersion: 8, window: { truncated_by: [] } }, ['live-1']), true);
    assert.equal(pending.has('live-1'), false, 'next qualifying read with dispatched match settles live-1');
});
