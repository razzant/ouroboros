// Issue #135: bounded live additions coexist with retained archive pages.
// Fresh history prunes only eligible old cards, preserving reading identity and
// cards created after the request began. No full DOM reset or quota ladder.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';

// DOM stub: the in-file harness of chat_instance_dom.test.js (house pattern:
// each instance-driving suite carries its own copy), plus a WebSocket constant
// so a failed sync is reported instead of being read as a socket drop.
class ClassList {
    constructor(node) { this.node = node; this.names = new Set(); }
    add(...names) { names.forEach((name) => this.names.add(name)); this.sync(); }
    remove(...names) { names.forEach((name) => this.names.delete(name)); this.sync(); }
    contains(name) { return this.names.has(name); }
    toggle(name, force) {
        const enabled = force === undefined ? !this.names.has(name) : Boolean(force);
        if (enabled) this.names.add(name); else this.names.delete(name);
        this.sync();
        return enabled;
    }
    sync() { this.node._className = [...this.names].join(' '); }
    from(value) { this.names = new Set(String(value || '').split(/\s+/).filter(Boolean)); this.sync(); }
}
class ElementStub {
    constructor(tag = 'div', doc = null) {
        this.tagName = tag.toUpperCase();
        this.ownerDocument = doc;
        this.dataset = {};
        const styleValues = new Map();
        this.style = { setProperty: (name, value) => styleValues.set(name, String(value)),
            getPropertyValue: (name) => styleValues.get(name) || '' };
        this.attributes = new Map();
        this.children = [];
        this.listeners = new Map();
        this.classList = new ClassList(this);
        this._className = '';
        this._innerHTML = '';
        this._textContent = '';
        this.value = '';
        this.hidden = false;
        this.disabled = false;
        this.isConnected = true;
        this.offsetParent = {};
        this.offsetHeight = 0;
        this.scrollTop = 0;
        this.scrollHeight = 0;
        this.clientHeight = 400;
    }
    set className(value) { this.classList.from(value); }
    get className() { return this._className; }
    set textContent(value) {
        this._textContent = String(value ?? '');
        this._innerHTML = this._textContent
            .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }
    get textContent() { return this._textContent; }
    set innerHTML(value) {
        this._innerHTML = String(value || '');
        if (!this.ownerDocument) return;
        this.children = [];
        for (const match of this._innerHTML.matchAll(/<([a-z0-9-]+)([^>]*)>/gi)) {
            const node = new ElementStub(match[1], this.ownerDocument);
            const attrs = match[2];
            const idMatch = attrs.match(/\sid="([^"]+)"/i);
            if (idMatch) node.id = idMatch[1];
            const classMatch = match[0].match(/\sclass="([^"]*)"/i);
            if (classMatch) node.className = classMatch[1];
            for (const data of attrs.matchAll(/\sdata-([a-z0-9-]+)(?:="([^"]*)")?/gi)) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                node.dataset[key] = data[2] ?? '';
            }
            node.parentNode = this;
            node.parentElement = this;
            this.children.push(node);
            if (node.id) this.ownerDocument.byId.set(node.id, node);
        }
    }
    get innerHTML() { return this._innerHTML; }
    addEventListener(type, fn) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(fn);
    }
    removeEventListener() {}
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) || ''; }
    removeAttribute(name) { this.attributes.delete(name); }
    appendChild(node) { return this.insertBefore(node, null); }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    prepend(...nodes) { const before = this.children[0] || null; nodes.forEach((node) => this.insertBefore(node, before)); }
    insertAdjacentElement(_position, node) { const list = this.parentNode?.children || []; return this.parentNode?.insertBefore(node, list[list.indexOf(this) + 1] || null); }
    insertBefore(node, before) {
        if (node?.isDocumentFragment) {
            for (const child of [...node.children]) this.insertBefore(child, before);
            return node;
        }
        node.parentNode?.removeChild?.(node);
        const index = before ? this.children.indexOf(before) : -1;
        if (index >= 0) this.children.splice(index, 0, node); else this.children.push(node);
        node.parentNode = this;
        node.parentElement = this;
        node.isConnected = true;
        this.scrollHeight = this.children.length * 80;
        return node;
    }
    removeChild(node) {
        const index = this.children.indexOf(node);
        if (index >= 0) this.children.splice(index, 1);
        node.parentNode = null;
        node.parentElement = null;
    }
    remove() { this.parentNode?.removeChild?.(this); this.isConnected = false; }
    replaceChildren(...nodes) { this.children = []; nodes.forEach((node) => this.appendChild(node)); }
    contains(node) {
        if (node === this) return true;
        return this.children.some((child) => child.contains(node));
    }
    querySelector(selector) {
        const id = selector.match(/^\[id="([^"]+)"\]$/)?.[1];
        if (id) return this.ownerDocument?.byId.get(id) || null;
        const data = selector.match(/^\[data-([a-z0-9-]+)\]$/i)?.[1];
        if (data) {
            const key = data.replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
            return this.children.find((child) => Object.hasOwn(child.dataset, key)) || null;
        }
        if (selector === '.typing-bubble') return this.children.find((child) => child.classList.contains('typing-bubble')) || null;
        if (selector.startsWith('.')) {
            const className = selector.slice(1).split(/[ :>\[]/)[0];
            return this.children.find((child) => child.classList.contains(className)) || null;
        }
        return null;
    }
    querySelectorAll(selector) {
        if (selector === '[id]') return this.children.filter((child) => child.id);
        const data = selector.match(/^\[data-([a-z0-9-]+)\]$/i)?.[1];
        if (data) {
            const key = data.replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
            return this.children.filter((child) => Object.hasOwn(child.dataset, key));
        }
        if (selector.startsWith('.')) {
            const className = selector.slice(1).split(/[ :>\[]/)[0];
            return this.children.filter((child) => child.classList.contains(className));
        }
        return [];
    }
    closest(selector) {
        if (selector === '.page.active' && this.classList.contains('page') && this.classList.contains('active')) return this;
        return this.parentElement?.closest?.(selector) || null;
    }
    getBoundingClientRect() {
        const feed = this.ownerDocument?.byId.get('chat-messages');
        if (this === feed) return { top: 0, bottom: this.clientHeight, left: 0, right: 100, width: 100, height: this.clientHeight };
        let item = this;
        while (item.parentNode && item.parentNode !== feed) item = item.parentNode;
        const index = feed?.children.indexOf(item) ?? -1;
        const top = index >= 0 ? index * 80 - feed.scrollTop : -1000;
        return { top, bottom: top + 20, left: 0, right: 100, width: 100, height: 20 };
    }
    getClientRects() { return [this.getBoundingClientRect()]; }
    focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; } click() {}
}
function installDom(fetchImpl = async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) })) {
    const prior = {
        document: globalThis.document, window: globalThis.window,
        sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch,
        ResizeObserver: globalThis.ResizeObserver,
        requestAnimationFrame: globalThis.requestAnimationFrame, WebSocket: globalThis.WebSocket,
        getSelection: globalThis.getSelection,
    };
    const document = {
        byId: new Map(), hidden: false, activeElement: null,
        createElement(tag) { return new ElementStub(tag, document); },
        createDocumentFragment() {
            const fragment = new ElementStub('#document-fragment', document);
            fragment.isDocumentFragment = true;
            return fragment;
        },
        getElementById(id) { return document.byId.get(id) || null; },
        addEventListener() {}, removeEventListener() {},
    };
    const mount = new ElementStub('div', document);
    document.byId.set('content', mount);
    const storage = new Map();
    globalThis.document = document;
    globalThis.window = {
        document, location: { href: 'http://local/' }, history: { replaceState() {} },
        addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
        getSelection: () => null, innerHeight: 800, CSS: { escape: (value) => value },
    };
    globalThis.getSelection = () => null;
    globalThis.sessionStorage = {
        getItem: (key) => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, String(value)),
        removeItem: (key) => storage.delete(key),
    };
    globalThis.fetch = fetchImpl;
    globalThis.ResizeObserver = class { observe() {} disconnect() {} };
    globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
    globalThis.WebSocket = { OPEN: 1 };
    return { prior, mount };
}
function restoreDom(prior) {
    Object.assign(globalThis, prior);
}

function makeInstance(mount) {
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
        ws: { readyState: 1 },
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
        isCurrent: () => true,
        apply() {},
    };
    const instance = createChatInstance({
        ws,
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
        asPanel: true,
    });
    return { instance, handlers, messages: globalThis.document.byId.get('chat-messages') };
}

// One task: a progress frame mints the card, the terminal frame seals it.
function sealCard(handlers, id, second) {
    const ts = `2026-09-02T01:${String(Math.floor(second / 60)).padStart(2, '0')}:${String(second % 60).padStart(2, '0')}Z`;
    handlers.get('chat')({ chat_id: 2, role: 'system', is_progress: true, task_id: id, content: 'working', ts });
    handlers.get('chat')({
        chat_id: 2, role: 'system', is_progress: true, task_id: id, content: 'done',
        task_terminal_status: 'completed', outcome_axes: { execution: 'ok' }, ts,
    });
}
const taskCards = (messages) => messages.children.filter((node) => node.dataset.taskId);
// Inspect live additions independently of the server rows also rendered by replay.
const liveTaskCards = (messages, historyRows) => {
    const historical = new Set(historyRows.map((row) => row.task_id));
    return taskCards(messages).filter((node) => !historical.has(node.dataset.taskId));
};

function historyPage(rows = [], { cursor = 'recent-page', next = null } = {}) {
    return { messages: rows, has_more: next !== null, next_cursor: next, page_cursor: cursor,
        window: { complete: next === null, truncated_by: next === null ? [] : ['quota'] } };
}
function historyFetch(historyCalls, response = () => historyPage()) {
    return async (url) => {
        const value = String(url);
        if (value.startsWith('/api/chat/history')) {
            historyCalls.push(value);
            return { ok: true, json: async () => response(value) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    };
}
// Cards have a 20px body and 60px gap. Park the viewport in a genuine gap when
// a case has no reader; pins below use the same sibling/scroll geometry.
function noReader(messages) { messages.clientHeight = 20; messages.scrollTop = 30; }
const cardById = (messages, id) => taskCards(messages).find(node => node.dataset.taskId === id);

// Durable rows that mint a card on replay: a finished task's chat summary (the
// n_human slice carries these) and a progress row (the n_progress slice).
const summaryRow = (id, second) => ({
    chat_id: 2, role: 'system', system_type: 'task_summary', task_id: id,
    history_id: `chat:${second}`, history_position: { source: 'chat', offset: second },
    content: 'done', tool_calls: 2, rounds: 2, task_terminal_status: 'completed',
    outcome_axes: { execution: 'ok' }, ts: historyTs(second),
});
const progressRow = (id, second) => ({
    chat_id: 2, role: 'system', is_progress: true, task_id: id,
    history_id: `progress:${second}`, history_position: { source: 'progress', offset: second },
    content: 'working', ts: historyTs(second),
});
const historyTs = (second) => `2026-09-01T00:${String(Math.floor(second / 60)).padStart(2, '0')}`
    + `:${String(second % 60).padStart(2, '0')}Z`;

async function sync(instance, revision) {
    await instance.refreshHistory({ revision });
    assert.equal(instance.hasPaintedHistory(), true, `sync ${revision} succeeded`);
}

test('a fresh sync bounds more than 200 live additions and later syncs remain routine', async () => {
    const calls = [];
    const { prior, mount } = installDom(historyFetch(calls));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        for (let i = 0; i < 201; i += 1) sealCard(handlers, `cap-t${i}`, i);
        assert.equal(taskCards(messages).length, 201, 'growth arms cleanup without eager eviction');
        noReader(messages);
        await sync(instance, 2);
        assert.equal(calls.length, 2);
        assert.equal(taskCards(messages).length, 0, 'fresh absent history releases every eligible completed addition');
        sealCard(handlers, 'after-prune', 300);
        const retained = cardById(messages, 'after-prune');
        noReader(messages);
        await sync(instance, 3);
        assert.equal(calls.length, 3);
        assert.equal(cardById(messages, 'after-prune'), retained, 'below-bound sync preserves exact DOM identity');
    } finally { instance.destroy(); restoreDom(prior); }
});

test('exactly 200 live additions do not trigger cleanup', async () => {
    const calls = [];
    const { prior, mount } = installDom(historyFetch(calls));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        for (let i = 0; i < 200; i += 1) sealCard(handlers, `at-cap-${i}`, i);
        const before = [...taskCards(messages)];
        noReader(messages);
        await sync(instance, 2);
        assert.equal(calls.length, 2);
        assert.deepEqual(taskCards(messages), before, 'the bound is exceeded, not merely reached');
    } finally { instance.destroy(); restoreDom(prior); }
});

test('cleanup preserves reading, focus and selection nodes until the reader releases them', async () => {
    const { prior, mount } = installDom(historyFetch([]));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        for (let i = 0; i < 201; i += 1) sealCard(handlers, `pinned-${i}`, i);
        const reading = cardById(messages, 'pinned-0');
        const focused = cardById(messages, 'pinned-80');
        const selected = cardById(messages, 'pinned-160');
        focused.focus();
        globalThis.getSelection = () => ({ isCollapsed: false, rangeCount: 1,
            getRangeAt: () => ({ intersectsNode: node => node === selected || selected.contains(node) }) });
        messages.clientHeight = 20;
        messages.scrollTop = messages.children.indexOf(reading) * 80;
        await sync(instance, 2);
        assert.equal(cardById(messages, 'pinned-0'), reading);
        assert.equal(cardById(messages, 'pinned-80'), focused);
        assert.equal(cardById(messages, 'pinned-160'), selected);
        assert.equal(document.activeElement, focused);
        assert.equal(taskCards(messages).length, 3, 'only the three explicitly protected cards survive');
        document.activeElement = null;
        globalThis.getSelection = () => null;
        noReader(messages);
        for (const listener of messages.listeners.get('scroll') || []) listener({ target: messages });
        assert.equal(taskCards(messages).length, 0, 'unpinning releases deferred memory without another full fetch');
    } finally { instance.destroy(); restoreDom(prior); }
});

test('live overflow cleanup preserves every retained archive card and its DOM node', async () => {
    const calls = [];
    const rows = Array.from({ length: 50 }, (_all, i) => summaryRow(`archive-${i}`, i));
    const { prior, mount } = installDom(historyFetch(calls, url => new URL(url, 'http://local').searchParams.get('cursor')
        ? historyPage(rows, { cursor: 'archive-page' })
        : historyPage([], { next: 'opaque-older' })));
    const created = [];
    const createElement = document.createElement;
    document.createElement = tag => { const node = createElement(tag); created.push(node); return node; };
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        const button = created.find(node => node.className === 'chat-load-older-btn');
        assert.ok(button && !button.hidden);
        await button.listeners.get('click')[0]();
        assert.equal(new URL(calls.at(-1), 'http://local').searchParams.get('cursor'), 'opaque-older');
        assert.ok(calls.every(url => !url.includes('n_human=')), 'navigation uses the opaque physical cursor');
        const archived = new Map(rows.map(row => [row.task_id, cardById(messages, row.task_id)]));
        assert.ok([...archived.values()].every(Boolean));
        for (let i = 0; i < 201; i += 1) sealCard(handlers, `extra-${i}`, i);
        noReader(messages);
        await sync(instance, 2);
        assert.equal(liveTaskCards(messages, rows).length, 0, 'archive ownership does not disable the live growth bound');
        assert.equal(taskCards(messages).length, rows.length);
        for (const [id, node] of archived) assert.equal(cardById(messages, id), node, 'archive node survives without a rebuild');
    } finally { instance.destroy(); restoreDom(prior); }
});

test('a default history window above 200 cards does not cause repeated cleanup', async () => {
    const calls = [];
    const rows = [
        ...Array.from({ length: 150 }, (_all, i) => summaryRow(`hist-s${i}`, i)),
        ...Array.from({ length: 60 }, (_all, i) => progressRow(`hist-p${i}`, 200 + i)),
    ];
    const { prior, mount } = installDom(historyFetch(calls, () => historyPage(rows)));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        const initial = cardById(messages, 'hist-s0');
        for (let revision = 2; revision <= 5; revision += 1) {
            sealCard(handlers, `live-t${revision}`, 300 + revision);
            noReader(messages);
            await sync(instance, revision);
            assert.equal(liveTaskCards(messages, rows).length, revision - 1);
            assert.equal(cardById(messages, 'hist-s0'), initial);
        }
        assert.equal(calls.length, 5, 'only the requested syncs execute');
    } finally { instance.destroy(); restoreDom(prior); }
});

test('an arm raised by a growing history response survives to the next fresh sync', async () => {
    let rows = [];
    const { prior, mount } = installDom(historyFetch([], () => historyPage(rows)));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        sealCard(handlers, 'live-before', 1);
        rows = Array.from({ length: 201 }, (_all, i) => summaryRow(`grown-${i}`, i));
        noReader(messages);
        await sync(instance, 2);
        assert.equal(liveTaskCards(messages, rows).length, 1, 'a response cannot consume an arm it raised');
        const historical = cardById(messages, 'grown-0');
        noReader(messages);
        await sync(instance, 3);
        assert.equal(liveTaskCards(messages, rows).length, 0);
        assert.equal(cardById(messages, 'grown-0'), historical, 'the fresh cleanup does not rebuild fetched cards');
        sealCard(handlers, 'after-prune', 400);
        noReader(messages);
        await sync(instance, 4);
        assert.equal(liveTaskCards(messages, rows).length, 1, 'the settled floor prevents a cleanup storm');
    } finally { instance.destroy(); restoreDom(prior); }
});

for (const alreadyArmed of [false, true]) {
    test(`a stale in-flight sync cannot evict newer cards (started armed=${alreadyArmed})`, async () => {
        const calls = [];
        let release;
        const gate = new Promise(resolve => { release = resolve; });
        const { prior, mount } = installDom(async url => {
            if (String(url).startsWith('/api/chat/history')) {
                calls.push(String(url));
                if (calls.length === 2) await gate;
                return { ok: true, json: async () => historyPage() };
            }
            return { ok: true, json: async () => ({ active_direct_turns: [] }) };
        });
        const { instance, handlers, messages } = makeInstance(mount);
        try {
            await sync(instance, 1);
            if (alreadyArmed) for (let i = 0; i < 201; i += 1) sealCard(handlers, `before-${i}`, i);
            noReader(messages);
            const inFlight = instance.refreshHistory({ revision: 2 });
            for (let i = 0; i < 201; i += 1) sealCard(handlers, `after-${i}`, 201 + i);
            const newest = cardById(messages, 'after-200');
            noReader(messages);
            release();
            await inFlight;
            assert.equal(calls.length, 2);
            assert.equal(taskCards(messages).length, 201);
            assert.equal(cardById(messages, 'after-200'), newest);
            if (!alreadyArmed) {
                noReader(messages);
                await sync(instance, 3);
                assert.equal(calls.length, 3);
                assert.equal(taskCards(messages).length, 0, 'the arm survives an older routine request');
            }
        } finally { instance.destroy(); restoreDom(prior); }
    });
}
