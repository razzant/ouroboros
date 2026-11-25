// Shared real createChatInstance consumer and geometry for project-read tests.
import assert from 'node:assert/strict';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, PREVIEW_SCOPE, restoreDom } from './chat_dom_fixture.js';

export const TS = '2026-09-28T12:00:00.000Z';
export const row = (n) => ({
    role: 'assistant', text: `answer ${n}`, ts: TS, history_id: `chat:${n}`,
    history_position: { source: 'chat', offset: n },
});
// Arrived last (the highest offset) but keeps the older time it was written at.
export const late = (n) => ({ ...row(n), text: `late answer ${n}`, ts: '2026-09-28T11:00:00.000Z' });

export function room(t, { onReadingLatest, projectId, fetchHistory } = {}) {
    // `rows` is the durable state; a read answers with the state it STARTED on.
    // `older`, when set, is the one older page behind the recent window (`olderWindow`
    // and `olderCoverage` its facts); `pages` instead chains several ({rows, window,
    // coverage}), page `k` (1-based) held until `pageGates[k]` resolves; `pageFails`
    // makes every page read (a saved place's too) fail, or page `k` once. `pageReads` lists them.
    // A recent read's gate resolved with 'fail' fails that read.
    const server = { rows: [], older: null, pages: null, pageGates: {}, pageFails: false,
        window: { complete: true, truncated_by: [] }, gates: [], coverage: undefined, olderWindow: {}, olderCoverage: undefined };
    const reads = [], pageReads = [];
    const { prior, mount } = installDom(async (url) => {
        if (fetchHistory && String(url).startsWith('/api/chat/history')) return fetchHistory(url);
        if (String(url).includes('cursor=')) {
            const k = Number(/older%3A(\d+)/.exec(String(url))?.[1] || 1);
            pageReads.push(k);
            await server.pageGates[k];
            if (server.pageFails === true || server.pageFails === k) {
                if (server.pageFails === k) server.pageFails = false;
                return { ok: false, status: 503, json: async () => ({ error: 'unavailable' }) };
            }
            const page = server.pages?.[k - 1] || { rows: server.older, window: server.olderWindow, coverage: server.olderCoverage };
            const more = k < (server.pages?.length || 1);
            return { ok: true, json: async () => ({ messages: [...page.rows], profile_scope: PREVIEW_SCOPE, page_cursor: `page:older:${k}`,
                next_cursor: more ? `older:${k + 1}` : null, has_more: more, coverage: page.coverage,
                window: { complete: false, truncated_by: ['page'], ...page.window } }) };
        }
        if (String(url).startsWith('/api/chat/history')) {
            const older = Boolean(server.older || server.pages?.length);
            const data = { messages: [...server.rows], profile_scope: PREVIEW_SCOPE, page_cursor: 'page:recent',
                next_cursor: older ? 'older:1' : null, has_more: older, window: server.window,
                coverage: server.coverage };
            reads.push(data.messages.map((item) => item.history_id));
            const gate = server.gates.shift();
            if (gate && await gate === 'fail') return { ok: false, status: 503, json: async () => ({ error: 'unavailable' }) };
            return { ok: true, json: async () => data };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    // The fixture document drops listeners; this room needs its visibility return.
    const documentListeners = new Map();
    globalThis.document.addEventListener = (type, fn) => {
        documentListeners.set(type, [...(documentListeners.get(type) || []), fn]);
    };
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1 };
    const socket = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { socket.set(type, [...(socket.get(type) || []), fn]); return () => {}; },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true, onReadingLatest, projectId,
    });
    t.after(() => { instance.destroy(); restoreDom(prior); globalThis.WebSocket = priorSocket; });
    const messages = globalThis.document.byId.get('chat-messages');
    // The composer is drawn over the feed's bottom edge; by default it sits just
    // below the stub viewport ([0, 20]), so every stub row is clear of it.
    const composer = globalThis.document.byId.get('chat-input-area');
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 100, width: 100, height: bottom - top });
    const chrome = ({ viewport = [0, 20], below = [20, 60] } = {}) => {
        messages.getBoundingClientRect = () => box(...viewport);
        composer.getBoundingClientRect = () => box(...below);
    };
    chrome();
    const reconnect = () => { for (const fn of socket.get('open') || []) fn({ previouslyConnected: true }); };
    const emit = (type, frame) => { for (const fn of socket.get(type) || []) fn(frame); };
    const gate = () => { let open; server.gates.push(new Promise((resolve) => { open = resolve; })); return open; };
    const shown = () => messages.children.map((node) => node.dataset?.historyId).filter(Boolean);
    const scroll = (top) => {
        messages.scrollTop = top;
        for (const handler of messages.listeners.get('scroll') || []) handler({});
    };
    // A short viewport over the rows: scrollTop 0 is "reading further up". The
    // reader's own gesture (a wheel up) moved it there; only a gesture pages history.
    const readUp = () => {
        messages.clientHeight = 10; scroll(0);
        for (const handler of messages.listeners.get('wheel') || []) {
            handler({ type: 'wheel', deltaY: -1, target: messages, timeStamp: 0 });
        }
    };
    const readLatest = () => scroll(messages.scrollHeight);
    const setHidden = (hidden) => {
        globalThis.document.hidden = hidden;
        globalThis.document.visibilityState = hidden ? 'hidden' : 'visible';
        for (const fn of documentListeners.get('visibilitychange') || []) fn({ type: 'visibilitychange' });
    };
    return { instance, server, reads, pageReads, messages, gate, shown, readUp, readLatest, scroll, setHidden, chrome, box, reconnect, emit };
}

export function placeAtTop(t, r, id) {
    // That row sits at the top of the feed: on screen only while the reader is there.
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        if (this.dataset?.historyId !== id) return rect.call(this);
        const top = -r.messages.scrollTop;
        return { top, bottom: top + 20, left: 0, right: 100, width: 100, height: 20 };
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
}

export async function until(check) {
    for (let i = 0; i < 200 && !check(); i += 1) await new Promise((resolve) => setImmediate(resolve));
    assert.ok(check());
}

export const quiet = async () => { for (let i = 0; i < 50; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
