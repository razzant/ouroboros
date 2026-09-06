// Bounded Main-chat memory (issue #135). The Main instance never runs
// destroy(), so its live task cards used to accumulate for the whole session.
// Past LIVE_CARD_CAP the instance arms the existing full history rebuild — the
// transaction a reconnect already runs — instead of evicting cards one by one:
// the next sync (in practice the debounced post-completion resync; here a
// refreshHistory(), which awaits the same syncHistory) replaces every live card
// with durable history, once. The cap is RELATIVE to the population the last
// rebuild produced, because a history window mints cards itself (progress rows,
// task_summary rows, lineage rows) and can exceed the cap on its own; an
// absolute cap would rebuild on every later sync. A Load-older window is
// rebuilt with its own quota, not cut back to the default one.
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
    prepend(node) { return this.insertBefore(node, this.children[0] || null); }
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
        this.scrollHeight = this.children.length * 20;
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
    getBoundingClientRect() { return { top: 0, bottom: 20, left: 0, right: 100, width: 100, height: 20 }; }
    getClientRects() { return [this.getBoundingClientRect()]; }
    focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; } click() {}
}
function installDom(fetchImpl = async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) })) {
    const prior = {
        document: globalThis.document, window: globalThis.window,
        sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch,
        ResizeObserver: globalThis.ResizeObserver,
        requestAnimationFrame: globalThis.requestAnimationFrame, WebSocket: globalThis.WebSocket,
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
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
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

// The server reports a quota-truncated window so the Load-older control is
// offered; `rows` is the durable history it returns for that request.
function historyFetch(historyCalls, rows = () => []) {
    return async (url) => {
        const value = String(url);
        if (value.startsWith('/api/chat/history')) {
            historyCalls.push(value);
            return { ok: true, json: async () => ({
                messages: rows(value), window: { complete: false, truncated_by: ['quota'] },
            }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    };
}

// Durable rows that mint a card on replay: a finished task's chat summary (the
// n_human slice carries these) and a progress row (the n_progress slice).
const summaryRow = (id, second) => ({
    chat_id: 2, role: 'system', system_type: 'task_summary', task_id: id,
    content: 'done', tool_calls: 2, rounds: 2, task_terminal_status: 'completed',
    outcome_axes: { execution: 'ok' }, ts: historyTs(second),
});
const progressRow = (id, second) => ({
    chat_id: 2, role: 'system', is_progress: true, task_id: id,
    content: 'working', ts: historyTs(second),
});
const historyTs = (second) => `2026-09-01T00:${String(Math.floor(second / 60)).padStart(2, '0')}`
    + `:${String(second % 60).padStart(2, '0')}Z`;

async function sync(instance, revision) {
    await instance.refreshHistory({ revision });
    assert.equal(instance.hasPaintedHistory(), true, `sync ${revision} succeeded`);
}

test('past the cap the next sync is ONE full rebuild from durable history; syncs are routine again after it', async () => {
    const historyCalls = [];
    const { prior, mount } = installDom(historyFetch(historyCalls));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1); // first load: the ordinary bootstrap rebuild
        for (let i = 0; i < 201; i += 1) sealCard(handlers, `cap-t${i}`, i);
        assert.equal(taskCards(messages).length, 201, 'the cap arms a rebuild; nothing is evicted card by card');
        await sync(instance, 2);
        assert.equal(historyCalls.length, 2);
        assert.equal(taskCards(messages).length, 0,
            'the sync was a full rebuild: durable history (empty here) replaced every live card');
        // The rebuild consumed the flag: the next sync folds routinely.
        sealCard(handlers, 'after-rebuild', 300);
        await sync(instance, 3);
        assert.equal(historyCalls.length, 3);
        assert.deepEqual(taskCards(messages).map((node) => node.dataset.taskId), ['after-rebuild'],
            'a routine sync keeps live cards');
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

test('exactly at the cap a sync stays routine', async () => {
    const historyCalls = [];
    const { prior, mount } = installDom(historyFetch(historyCalls));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        for (let i = 0; i < 200; i += 1) sealCard(handlers, `cap-t${i}`, i);
        await sync(instance, 2);
        assert.equal(historyCalls.length, 2);
        assert.equal(taskCards(messages).length, 200, 'the bound is exceeded-by-one, not reached');
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

// A durable row mints a live-card RECORD on replay, but this DOM stub reports
// every fresh node as already connected, so the pass-2 sweep never inserts it:
// the cards visible below are the LIVE ones. That is the observable this suite
// needs — a rebuild drops every live card, a routine sync keeps it.
test('after Load older the cap is relative to the wider window and its rebuild keeps the owner quota', async () => {
    const historyCalls = [];
    // The default window is empty; the escalated one carries 50 durable cards.
    const { prior, mount } = installDom(historyFetch(historyCalls, (url) => (
        url.includes('n_human=400')
            ? Array.from({ length: 50 }, (_all, i) => summaryRow(`older-t${i}`, i))
            : []
    )));
    // The stub reports every fresh element as connected, so the Load-older
    // control is never (re)prepended; capture its button at creation instead.
    const created = [];
    const createElement = globalThis.document.createElement;
    globalThis.document.createElement = (tag) => { const el = createElement(tag); created.push(el); return el; };
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        const button = created.find((el) => el.className === 'chat-load-older-btn');
        assert.ok(button && !button.hidden, 'quota truncation offers Load older');
        button.listeners.get('click')[0]();
        await new Promise((resolve) => setTimeout(resolve, 10));
        assert.match(historyCalls.at(-1), /n_human=400/, 'the escalated window is in force');

        // 200 live cards on top of a 50-record window: over the absolute cap,
        // but not yet 200 PAST the population that rebuild produced.
        for (let i = 0; i < 200; i += 1) sealCard(handlers, `wide-t${i}`, i);
        await sync(instance, 2);
        assert.match(historyCalls.at(-1), /n_human=400/);
        assert.equal(taskCards(messages).length, 200, 'the sync folded in routinely');

        // One more crosses the relative bound. The window is wide, so the
        // rebuild replays the OWNER's quota instead of skipping the cap.
        sealCard(handlers, 'wide-t200', 200);
        await sync(instance, 3);
        assert.match(historyCalls.at(-1), /n_human=400/, 'the rebuild kept the owner quota');
        assert.equal(taskCards(messages).length, 0, 'the rebuild replaced every live card');
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

test('a window that itself exceeds the cap does not rebuild on every later sync', async () => {
    // The convergence case: 150 task_summary rows in the n_human slice plus 60
    // progress rows in the n_progress slice already mint 210 records, so an
    // ABSOLUTE cap is re-armed by the first live task after every rebuild — a
    // rebuild on every post-completion sync. Against the relative bound each
    // sync below stays routine and keeps the live cards it has folded in.
    const historyCalls = [];
    const rows = [
        ...Array.from({ length: 150 }, (_all, i) => summaryRow(`hist-s${i}`, i)),
        ...Array.from({ length: 60 }, (_all, i) => progressRow(`hist-p${i}`, 200 + i)),
    ];
    assert.equal(rows.length, 210, 'the default window alone exceeds LIVE_CARD_CAP');
    const { prior, mount } = installDom(historyFetch(historyCalls, () => rows));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1);
        for (let revision = 2; revision <= 5; revision += 1) {
            sealCard(handlers, `live-t${revision}`, 300 + revision);
            await sync(instance, revision);
            assert.equal(taskCards(messages).length, revision - 1,
                `sync ${revision} folded in routinely instead of rebuilding`);
        }
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

test('an arm raised while a routine replay is running survives to the next sync', async () => {
    // The window itself mints the cap-crossing cards, so the arm goes up AFTER
    // this sync already decided it is routine. Clearing it unconditionally at the
    // end of every successful sync would discard that request and let growth that
    // only ever arrives through history run unbounded.
    const historyCalls = [];
    let rows = [];
    const { prior, mount } = installDom(historyFetch(historyCalls, () => rows));
    const { instance, handlers, messages } = makeInstance(mount);
    try {
        await sync(instance, 1); // bootstrap rebuild over an empty window: floor 0
        sealCard(handlers, 'live-before', 1);
        rows = Array.from({ length: 201 }, (_all, i) => summaryRow(`grown-t${i}`, i));
        await sync(instance, 2);
        assert.deepEqual(taskCards(messages).map((node) => node.dataset.taskId), ['live-before'],
            'the cap-crossing sync itself stayed routine');
        await sync(instance, 3);
        assert.equal(taskCards(messages).length, 0, 'the armed rebuild ran on the next sync');
        sealCard(handlers, 'after-rebuild', 400);
        await sync(instance, 4);
        assert.deepEqual(taskCards(messages).map((node) => node.dataset.taskId), ['after-rebuild'],
            'that rebuild set the new floor: later syncs are routine again');
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

test('a sync already in flight when the cap arms does not rebuild from its older window', () => {
    // The reviewer scenario for the arm: the response was fetched before the cap
    // crossed, so rebuilding from it would clear cards that window never saw and
    // consume the arm with nothing left to replay them. The in-flight sync must
    // fold routinely; the arm belongs to the NEXT sync, which fetches again.
    const historyCalls = [];
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const { prior, mount } = installDom(async (url) => {
        const value = String(url);
        if (value.startsWith('/api/chat/history')) {
            historyCalls.push(value);
            if (historyCalls.length === 2) await gate;   // the deferred (stale) window
            return { ok: true, json: async () => ({
                messages: [], window: { complete: false, truncated_by: ['quota'] },
            }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const { instance, handlers, messages } = makeInstance(mount);
    return (async () => {
        try {
            await sync(instance, 1);
            const inFlight = instance.refreshHistory({ revision: 2 });
            for (let i = 0; i < 201; i += 1) sealCard(handlers, `race-t${i}`, i);
            release();
            await inFlight;
            assert.equal(historyCalls.length, 2);
            assert.equal(taskCards(messages).length, 201,
                'the older window folded in routinely and kept the newer cards');
            await sync(instance, 3);
            assert.equal(historyCalls.length, 3, 'the arm survived and fetched a fresh window');
            assert.equal(taskCards(messages).length, 0, 'that fresh sync rebuilt');
        } finally {
            instance.destroy();
            restoreDom(prior);
        }
    })();
});
