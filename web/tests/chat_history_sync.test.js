// Behavioural characterization of the feed/history owner, exercised where the
// code now lives. The factory is driven exactly like chat.js drives it: the
// bootstrap fires at construction, hydration rides the sticky single-flight,
// rows land through addMessage/insertMessageNode, and the socket-open resync
// paints the reconnect banner after the refetch — with the transport stubbed
// through global fetch and the collaborators through observable spies.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatHistorySync } from '../modules/chat_history_sync.js';

function makeNode(tag = 'div') {
    // utils.escapeHtmlText escapes through the textContent -> innerHTML round
    // trip of a scratch element, so the stub mirrors that link.
    let text = '';
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        get textContent() { return text; },
        set textContent(value) {
            text = String(value);
            el.innerHTML = text
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        },
        innerHTML: '',
        hidden: false,
        type: '',
        disabled: false,
        dataset: {},
        children: [],
        listeners: {},
        classNames: new Set(),
        parentNode: null,
        isConnected: false,
        scrollTop: 0,
        scrollHeight: 0,
        style: {},
        appendChild(child) {
            if (child.isFragment) {
                for (const nested of child.children.splice(0)) el.appendChild(nested);
                return child;
            }
            el.children.push(child);
            child.parentNode = el;
            child.isConnected = true;
            return child;
        },
        append(...nodes) { nodes.forEach((node) => el.appendChild(node)); },
        prepend(node) { el.children.unshift(node); node.parentNode = el; node.isConnected = true; },
        insertBefore(node, before) {
            if (node.isFragment) {
                for (const nested of node.children.splice(0)) el.insertBefore(nested, before);
                return node;
            }
            const at = el.children.indexOf(before);
            if (at === -1) el.children.push(node);
            else el.children.splice(at, 0, node);
            node.parentNode = el;
            node.isConnected = true;
            return node;
        },
        remove() {
            if (el.parentNode) {
                const at = el.parentNode.children.indexOf(el);
                if (at !== -1) el.parentNode.children.splice(at, 1);
            }
            el.parentNode = null;
            el.isConnected = false;
        },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        classList: {
            add(name) { el.classNames.add(name); },
            remove(name) { el.classNames.delete(name); },
            toggle() {},
            contains(name) { return el.classNames.has(name); },
        },
    };
    return el;
}

const tick = () => new Promise((resolve) => setImmediate(resolve));
async function settle(times = 6) {
    for (let i = 0; i < times; i += 1) await tick();
}

function historyFeed({ isMain = false, responses = [], saved = null } = {}) {
    const priorDocument = globalThis.document;
    const priorWindow = globalThis.window;
    const priorFetch = globalThis.fetch;
    const priorRaf = globalThis.requestAnimationFrame;
    const priorRic = globalThis.requestIdleCallback;
    const priorSession = globalThis.sessionStorage;
    const priorWebSocket = globalThis.WebSocket;

    globalThis.document = {
        createElement: (tag) => makeNode(tag),
        createDocumentFragment: () => {
            const fragment = makeNode('fragment');
            fragment.isFragment = true;
            return fragment;
        },
    };
    globalThis.window = { location: { href: 'http://localhost/' }, history: { replaceState() {} } };
    globalThis.requestAnimationFrame = (fn) => { fn(); return 0; };
    globalThis.requestIdleCallback = (fn) => { fn(); return 0; };
    globalThis.WebSocket = { OPEN: 1 };
    const stored = new Map();
    globalThis.sessionStorage = {
        getItem: (key) => (key === 'ouro_chat' && saved ? JSON.stringify(saved) : (stored.get(key) ?? null)),
        setItem: (key, value) => stored.set(key, value),
        removeItem: (key) => stored.delete(key),
    };
    const fetchCalls = [];
    globalThis.fetch = async (url) => {
        fetchCalls.push(String(url));
        const next = responses.length ? responses.shift() : { ok: true, messages: [] };
        if (next instanceof Error) throw next;
        return {
            ok: next.ok !== false,
            status: next.ok === false ? 500 : 200,
            json: async () => ({ messages: next.messages || [], window: next.window ?? null }),
        };
    };

    const page = makeNode('div');
    const messagesDiv = makeNode('div');
    const calls = [];
    const seen = new Set();
    const spy = (name, result) => (...args) => { calls.push({ name, args }); return result; };
    let keySeq = 0;

    const api = createChatHistorySync({
        ws: { ws: { readyState: 1 } },
        isMain,
        chatId: isMain ? 1 : 7,
        page,
        messagesDiv,
        storeKey: (base) => base,
        chatSessionId: 'session-me',
        initialScrollPending: false,
        isProjectOpening: null,
        persistedHistory: [],
        seenMessageKeys: seen,
        messageKeyOrder: [],
        pendingUserBubbles: new Map(),
        inputHistory: [],
        localEchoJournal: new Map(),
        pendingSubmissions: new Map(),
        retiredTaskIds: new Set(),
        liveCardRecords: new Map(),
        taskUiStates: new Map(),
        ephemeralDecisionTaskIds: new Set(),
        pendingSuggestedNames: new Map(),
        cancelableTaskIds: new Set(),
        subagentChildParents: new Map(),
        subagentTerminalChildren: new Set(),
        activeDirectActivities: new Map(),
        buildMessageKey: (role, text, ts) => `${role}|${text}|${ts || keySeq++}`,
        rememberMessageKey: (key) => { if (key) seen.add(key); calls.push({ name: 'rememberMessageKey', args: [key] }); },
        formatMsgTime: () => null,
        getSenderLabel: () => 'Sender',
        stampNodeTimestamp: (node, ts) => { node.dataset.ts = String(Date.parse(ts) || 0); return false; },
        renderRoutingAnnotation: spy('renderRoutingAnnotation'),
        appendDocumentBubble: spy('appendDocumentBubble', true),
        isNearBottom: () => true,
        captureVisibleTimelineAnchor: () => null,
        restoreVisibleTimelineAnchor: () => false,
        withStableViewport: (mutate) => mutate(),
        updateMessagesPadding: spy('updateMessagesPadding'),
        updateScrollButton: spy('updateScrollButton'),
        scrollToBottomAfterLayout: spy('scrollToBottomAfterLayout'),
        restoreScrollPosition: spy('restoreScrollPosition'),
        isViewportSticky: () => true,
        setStatus: spy('setStatus'),
        syncChatStatus: spy('syncChatStatus'),
        hideTypingIndicatorOnly: spy('hideTypingIndicatorOnly'),
        hasActiveLiveCard: () => false,
        loadUiPreferences: async () => {},
        refreshHeaderControlState: spy('refreshHeaderControlState'),
        setActiveLiveGroupId: spy('setActiveLiveGroupId'),
        setSyncPass1Active: spy('setSyncPass1Active'),
        finishLiveCard: spy('finishLiveCard'),
        ensureLiveCardVisible: spy('ensureLiveCardVisible'),
        getTaskUiState: () => null,
        markLiveCardFinalizing: spy('markLiveCardFinalizing'),
        updateLiveCardFromProgressMessage: spy('updateLiveCardFromProgressMessage'),
        appendTaskSummaryToLiveCard: spy('appendTaskSummaryToLiveCard'),
        setSubagentParent: spy('setSubagentParent'),
        routeSubagentFinalMessageToCard: spy('routeSubagentFinalMessageToCard'),
        routeSubagentTerminalToCard: spy('routeSubagentTerminalToCard'),
        renderLiveCardMeta: spy('renderLiveCardMeta'),
        updateLiveCardCount: spy('updateLiveCardCount'),
        syncLiveCardLayout: spy('syncLiveCardLayout'),
        saveInputHistory: spy('saveInputHistory'),
        setInputHistoryIndex: spy('setInputHistoryIndex'),
    });

    return {
        ...api,
        page,
        messagesDiv,
        calls,
        fetchCalls,
        stored,
        named: (name) => calls.filter((call) => call.name === name),
        bubbles: () => messagesDiv.children.filter((child) => child.className.includes('chat-bubble')),
        restore() {
            globalThis.document = priorDocument;
            globalThis.window = priorWindow;
            globalThis.fetch = priorFetch;
            globalThis.requestAnimationFrame = priorRaf;
            globalThis.requestIdleCallback = priorRic;
            globalThis.sessionStorage = priorSession;
            globalThis.WebSocket = priorWebSocket;
        },
    };
}

test('addMessage renders once, dedupes by key and snapshots durable rows', async () => {
    const f = historyFeed();
    await settle();
    const first = f.addMessage('hello', 'user', false, '2026-08-18T00:00:00Z');
    assert.ok(first, 'the first render returns the bubble');
    assert.ok(first.className.includes('chat-bubble user'));
    const dup = f.addMessage('hello', 'user', false, '2026-08-18T00:00:00Z');
    assert.equal(dup, null, 'the same message key renders once');
    assert.ok(f.stored.has('ouro_chat'), 'durable rows land in the session snapshot');
    const snapshotted = JSON.parse(f.stored.get('ouro_chat'));
    assert.equal(snapshotted.length, 1);
    // An ephemeral row renders but never persists.
    f.addMessage('transient', 'system', false, '2026-08-18T00:00:01Z', false, { ephemeral: true });
    assert.equal(JSON.parse(f.stored.get('ouro_chat')).length, 1);
    f.restore();
});

test('insertMessageNode keeps the feed chronological by timestamp', async () => {
    const f = historyFeed();
    await settle();
    const early = makeNode('div');
    early.dataset.ts = '100';
    const late = makeNode('div');
    late.dataset.ts = '300';
    const middle = makeNode('div');
    middle.dataset.ts = '200';
    f.insertMessageNode(early);
    f.insertMessageNode(late);
    f.insertMessageNode(middle);
    assert.deepEqual(f.messagesDiv.children.map((child) => child.dataset.ts), ['100', '200', '300']);
    f.restore();
});

test('hydration is sticky single-flight; only a failed sync unsticks it', async () => {
    const f = historyFeed({
        responses: [
            { ok: false },
            { ok: true, messages: [] },
            { ok: true, messages: [] },
        ],
    });
    await settle();
    assert.equal(f.fetchCalls.length, 1, 'the bootstrap fetched once and failed');
    await f.refreshHistory({ revision: 0 });
    assert.equal(f.fetchCalls.length, 2, 'the failed sync reset the sticky promise, so this refetches');
    await f.refreshHistory({ revision: 0 });
    assert.equal(f.fetchCalls.length, 2, 'a hydrated instance answers from the sticky promise');
    f.restore();
});

test('a rebuild paints the durable rows and reports painted history', async () => {
    const f = historyFeed({
        responses: [{
            ok: true,
            messages: [
                { role: 'user', text: 'question', ts: '2026-08-18T00:00:00Z', client_message_id: 'cmid-1' },
                { role: 'assistant', text: 'answer', ts: '2026-08-18T00:00:05Z', markdown: false },
            ],
        }],
    });
    await settle();
    assert.equal(f.hasPaintedHistory(), true);
    const texts = f.bubbles().map((bubble) => bubble.innerHTML);
    assert.equal(texts.length, 2, 'both durable rows painted');
    assert.match(texts[0], /question/);
    assert.match(texts[1], /answer/);
    f.restore();
});

test('the socket-open resync paints the reconnect banner after a real refetch', async () => {
    const f = historyFeed({
        responses: [
            { ok: true, messages: [] },
            { ok: true, messages: [] },
        ],
    });
    await settle();
    const before = f.fetchCalls.length;
    f.handleSocketOpen({ previouslyConnected: true });
    await settle();
    assert.equal(f.fetchCalls.length, before + 1, 'a reconnect always does a real fetch');
    const banner = f.bubbles().find((bubble) => bubble.innerHTML.includes('Reconnected'));
    assert.ok(banner, 'the reconnect banner lands as a bubble');
    assert.equal(banner.dataset.ephemeral, '1', 'the banner is ephemeral, never persisted');
    assert.ok(f.named('refreshHeaderControlState').length >= 1);
    f.restore();
});

test('an empty main feed greets exactly once', async () => {
    const f = historyFeed({ isMain: true, responses: [{ ok: true, messages: [] }] });
    await settle();
    const welcomes = f.bubbles().filter((bubble) => bubble.innerHTML.includes('awakened'));
    assert.equal(welcomes.length, 1, 'the welcome renders for an empty main feed');
    f.restore();
});

test('a project instance requests its own thread window', async () => {
    const f = historyFeed({ responses: [{ ok: true, messages: [] }] });
    await settle();
    assert.match(f.fetchCalls[0], /chat_id=7/, 'the non-main instance scopes history to its thread');
    f.restore();
});
