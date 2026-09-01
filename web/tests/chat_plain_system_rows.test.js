// Plain-text render contract for system rows without markdown:true (owner bug
// report 2026-08-31): a Project lifecycle card must render as escaped plain
// dashboard text — no markdown parsing, no enhancement (Mermaid/Chart/KaTeX
// live only behind enhanceChatMarkdown) — identically live and after reload,
// while assistant rows and markdown:true system rows keep the rich path and
// skill_review keeps its dedicated renderer.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { createChatInstance } from '../modules/chat.js';

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const styleSource = readFileSync(new URL('../style.css', import.meta.url), 'utf8');

// --- DOM harness (same stub family as chat_instance_dom.test.js) ---

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
        this.style = { setProperty() {} };
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
    focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; }
    click() {}
}

function installDom(fetchImpl = async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) })) {
    const prior = {
        document: globalThis.document, window: globalThis.window,
        sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch,
        WebSocket: globalThis.WebSocket,
        ResizeObserver: globalThis.ResizeObserver,
        requestAnimationFrame: globalThis.requestAnimationFrame,
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
    globalThis.WebSocket = { OPEN: 1 };
    globalThis.ResizeObserver = class { observe() {} disconnect() {} };
    globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
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
        updateUnreadBadge() {},
        stateSnapshots,
        chatId: 2,
        idPrefix: 'chat',
        mountEl: mount,
        asPanel: true,
    });
    return { instance, handlers };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

function findBubble(role) {
    const messages = globalThis.document.byId.get('chat-messages');
    return messages.children.find((node) => node.classList.contains('chat-bubble')
        && node.classList.contains(role) && !node.classList.contains('typing-bubble'));
}

const PLAIN_ROW = {
    chat_id: 2,
    role: 'system',
    system_type: 'project_completion_summary',
    markdown: false,
    content: 'Launch › Ship · Completed\nPlain excerpt line.',
    project_id: 'launch',
    project_name: 'Launch',
    ts: '2026-08-31T00:00:00Z',
};

test('plain project row renders escaped text with Open Project and no markdown machinery', async () => {
    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        made.handlers.get('chat')(PLAIN_ROW);
        const bubble = findBubble('system');
        assert.ok(bubble, 'the plain system row rendered a bubble');
        // Escaped plain text: the raw newline survives (pre-wrap), so the row
        // did NOT pass through the markdown renderer (whose no-parser fallback
        // rewrites \n to <br>) and produced no heading elements.
        assert.match(bubble.innerHTML, /Launch › Ship · Completed\nPlain excerpt line\./);
        assert.doesNotMatch(bubble.innerHTML, /<br>|<h1|<h2|md-h1|md-h2/);
        // Bug report #9: no enhancement pass — Mermaid/Chart/KaTeX/code-copy
        // only ever activate behind enhanceChatMarkdown's enhanced stamp.
        assert.equal(bubble.getAttribute('data-chat-markdown-enhanced'), '');
        // Bug report #4: the Open Project action still rides the message body.
        const message = bubble.querySelector('.message');
        const actions = message.children.find((node) => node.classList.contains('system-message-actions'));
        assert.ok(actions, 'system-message-actions container present');
        assert.equal(actions.children[0]?.textContent, 'Open Project ↗');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('plain system row renders identically live and after history reload', async () => {
    // Live pass.
    let liveHtml = '';
    {
        const { prior, mount } = installDom();
        let instance;
        try {
            const made = makeInstance(mount);
            instance = made.instance;
            made.handlers.get('chat')(PLAIN_ROW);
            liveHtml = findBubble('system').innerHTML;
        } finally {
            instance?.destroy();
            restoreDom(prior);
        }
    }
    // Reload pass: the same row replayed from /api/chat/history.
    const historyRow = {
        text: PLAIN_ROW.content,
        role: 'system',
        ts: PLAIN_ROW.ts,
        is_progress: false,
        system_type: PLAIN_ROW.system_type,
        markdown: false,
        project_id: PLAIN_ROW.project_id,
        project_name: PLAIN_ROW.project_name,
    };
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [historyRow] }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    let instance;
    try {
        ({ instance } = makeInstance(mount));
        await settle();
        await settle();
        const bubble = findBubble('system');
        assert.ok(bubble, 'history replay rendered the plain system row');
        assert.equal(bubble.innerHTML, liveHtml,
            'live DOM and reload DOM are byte-identical for the plain row');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('assistant rows and markdown:true system rows keep the rich markdown path', async () => {
    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        const handlers = made.handlers;
        handlers.get('chat')({
            chat_id: 2, role: 'assistant', markdown: true,
            content: 'Assistant line one\nline two',
            ts: '2026-08-31T00:00:01Z',
        });
        const assistant = findBubble('assistant');
        // The markdown renderer ran (no-parser fallback rewrites \n to <br>)
        // and the enhancement pass stamped the bubble.
        assert.match(assistant.innerHTML, /Assistant line one<br>line two/);
        assert.equal(assistant.getAttribute('data-chat-markdown-enhanced'), 'true');

        // Bug report #7: a system row that DOES carry markdown:true (e.g. a
        // markdown terminal_incident projection) still renders rich.
        handlers.get('chat')({
            chat_id: 2, role: 'system', system_type: 'terminal_incident',
            markdown: true, content: 'Incident line one\nline two',
            ts: '2026-08-31T00:00:02Z',
        });
        const incident = findBubble('system');
        assert.match(incident.innerHTML, /Incident line one<br>line two/);
        assert.equal(incident.getAttribute('data-chat-markdown-enhanced'), 'true');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('system row without a markdown flag renders plain (cancel_receipt class)', async () => {
    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        // Owner D14: the salvage text arrives VERBATIM (markers preserved) and
        // renders as escaped plain text — literal markers, no elements.
        made.handlers.get('chat')({
            chat_id: 2, role: 'system', system_type: 'cancel_receipt',
            content: 'Task cancelled. Preserved below.\n## Heading **bold** `code`',
            ts: '2026-08-31T00:00:03Z',
        });
        const bubble = findBubble('system');
        assert.match(bubble.innerHTML, /Task cancelled\. Preserved below\.\n## Heading \*\*bold\*\* `code`/);
        assert.doesNotMatch(bubble.innerHTML, /<br>|<h2|md-h2|<strong/);
        assert.equal(bubble.getAttribute('data-chat-markdown-enhanced'), '');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('render arm order and enhancement guard are pinned in source', () => {
    // The plain-system arm sits between the dedicated skill_review renderer
    // (bug report #8) and the byte-pinned final markdown arm
    // (tests/test_restart_reconnect.py pins ": renderChatMarkdown(text);").
    const ternary = chatSource.slice(
        chatSource.indexOf("const rendered = role === 'user'"),
        chatSource.indexOf(': renderChatMarkdown(text);'),
    );
    assert.match(ternary, /renderSkillReviewDisclosure\(text, opts\.skillReview \|\| null\)/);
    assert.match(ternary, /role === 'system' && systemType !== 'skill_review' && markdown !== true\n\s+\? escapeHtml\(text\)/);
    assert.match(chatSource, /: renderChatMarkdown\(text\);/);
    // The enhancement pass skips exactly the plain-system case.
    assert.match(
        chatSource,
        /if \(role !== 'user' && systemType !== 'skill_review' && \(role !== 'system' \|\| markdown === true\)\) enhanceMountedMarkdown\(bubble\);/,
    );
});

test('chat bubble heading clamp is scoped in style.css', () => {
    // Owner D5: inside chat bubbles H1 clamps to the section size and H2/H3 to
    // body size; the global md-h1 page-size rule stays for non-chat surfaces.
    assert.match(styleSource, /\.chat-bubble \.message \.md-h1 \{\n\s+font-size: var\(--type-section\);\n\}/);
    assert.match(styleSource, /\.chat-bubble \.message \.md-h2,\n\.chat-bubble \.message \.md-h3 \{\n\s+font-size: var\(--type-body\);\n\}/);
});
