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

// Source pins below match across line breaks; normalize CRLF so a Windows
// checkout (core.autocrlf) reads the same bytes the regexes were written for.
const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8')
    .replace(/\r\n?/g, '\n');
const styleSource = readFileSync(new URL('../style.css', import.meta.url), 'utf8')
    .replace(/\r\n?/g, '\n');

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
    prepend(...nodes) { const before = this.children[0] || null; nodes.forEach((node) => this.insertBefore(node, before)); }
    after(node) {
        const parent = this.parentNode;
        if (parent) parent.insertBefore(node, parent.children[parent.children.indexOf(this) + 1] || null);
    }
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
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
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
    content: 'Launch › Ship · Completed\nOpen the Project for details.',
    project_id: 'launch',
    project_name: 'Launch',
    ts: '2026-08-31T00:00:00Z',
};

// The stub DOM does not aggregate descendant text, so a reference is read by its own parts.
const referenceShape = (node) => ({
    intent: node?.dataset?.intent,
    pill: Boolean(node?.classList?.contains('chat-quiz-project')),
    parts: (node?.children || []).map((child) => child.textContent),
    spoken: node?.getAttribute?.('aria-label'),
});
const LAUNCH_REFERENCE = { intent: 'open-project', pill: true, parts: ['', 'Launch', '↗'], spoken: 'Open project Launch' };

test('plain project row renders escaped text with the Project reference and no markdown machinery', async () => {
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
        assert.match(bubble.innerHTML, /Launch › Ship · Completed\nOpen the Project for details\./);
        assert.doesNotMatch(bubble.innerHTML, /<br>|<h1|<h2|md-h1|md-h2/);
        // Bug report #9: no enhancement pass — Mermaid/Chart/KaTeX/code-copy
        // only ever activate behind enhanceChatMarkdown's enhanced stamp.
        assert.equal(bubble.getAttribute('data-chat-markdown-enhanced'), '');
        // The host action is beside prose, so Markdown margins cannot erase its gap.
        const message = bubble.querySelector('.message');
        const actions = bubble.children.find((node) => node.classList.contains('system-message-actions'));
        assert.equal(message.contains(actions), false);
        assert.equal(bubble.children.indexOf(actions), bubble.children.indexOf(message) + 1);
        assert.ok(actions, 'system-message-actions container present');
        // The row points at its Project with the one reference, never a button of its own.
        assert.deepEqual(referenceShape(actions.children[0]), LAUNCH_REFERENCE);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

// Project completion mirror (docs/DESIGN.md): a Project root that ended with
// Ouroboros's own final answer reaches Main as an ORDINARY Ouroboros message —
// the answer through the chat markdown path, folded by CSS, with the Project
// chip under it. The wire row is still role="system"; the typed key decides.
const MIRROR_ROW = {
    ...PLAIN_ROW,
    completion_answer: '**Done: PR #7 merged.**\n\nSecond paragraph.',
};

test('a completion row carrying the answer renders as an ordinary Ouroboros message with the Project chip', async () => {
    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        made.handlers.get('chat')(MIRROR_ROW);
        assert.equal(findBubble('system'), undefined, 'no yellow System bubble for an answered ending');
        const bubble = findBubble('assistant');
        assert.ok(bubble, 'the answer is an assistant bubble');
        assert.ok(bubble.classList.contains('project-answer'));
        assert.equal(bubble.dataset.systemType, 'project_completion_summary');
        assert.match(bubble.innerHTML, /<div class="sender">Ouroboros<\/div>/);
        // His words, through the markdown path — and none of the host's pointer text.
        assert.match(bubble.innerHTML, /Done: PR #7 merged\./);
        assert.doesNotMatch(bubble.innerHTML, /Open the Project for details|Completed/);
        const message = bubble.querySelector('.message');
        const actions = bubble.children.find((node) => node.classList.contains('system-message-actions'));
        assert.equal(bubble.children.indexOf(actions), bubble.children.indexOf(message) + 1);
        // One control, and the SAME one the System row carries: the voice of a row never
        // chooses how the UI points at its Project.
        assert.equal(actions.children.length, 1);
        const chip = actions.children[0];
        assert.deepEqual(referenceShape(chip), LAUNCH_REFERENCE);
        // The stub DOM has no event loop: run the chip's own click listener and
        // capture what it hands to the window.
        let opened = null;
        const priorDispatch = globalThis.window.dispatchEvent;
        const priorCustomEvent = globalThis.CustomEvent;
        globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
        globalThis.window.dispatchEvent = (event) => { opened = { type: event.type, detail: event.detail }; };
        try {
            [].concat(chip.listeners.get('click') || []).forEach((fn) => fn({}));
        } finally {
            globalThis.window.dispatchEvent = priorDispatch;
            globalThis.CustomEvent = priorCustomEvent;
        }
        assert.deepEqual(opened, {
            type: 'ouro:open-project', detail: { project: { id: 'launch', name: 'Launch' }, task_id: '', quiz_id: '' },
        });
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('the mirrored answer replays from history exactly as it arrived live', async () => {
    // The dominant path: Main hydrates from /api/chat/history, where the row is still
    // role="system" and carries the typed key.
    let liveHtml = '';
    {
        const { prior, mount } = installDom();
        let instance;
        try {
            const made = makeInstance(mount);
            instance = made.instance;
            made.handlers.get('chat')(MIRROR_ROW);
            liveHtml = findBubble('assistant').innerHTML;
        } finally {
            instance?.destroy();
            restoreDom(prior);
        }
    }
    const historyRow = {
        text: MIRROR_ROW.content, role: 'system', ts: MIRROR_ROW.ts, is_progress: false,
        system_type: MIRROR_ROW.system_type, markdown: false,
        project_id: MIRROR_ROW.project_id, project_name: MIRROR_ROW.project_name,
        completion_answer: MIRROR_ROW.completion_answer,
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
        assert.equal(findBubble('system'), undefined, 'history never falls back to the pointer when the key is present');
        const bubble = findBubble('assistant');
        assert.ok(bubble, 'history replay rendered the mirrored answer');
        assert.ok(bubble.classList.contains('project-answer'));
        assert.equal(bubble.innerHTML, liveHtml, 'live DOM and reload DOM are byte-identical for the mirror');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('the fold is CSS over the complete answer: clamp always, fade only when folded, tokens only', () => {
    const block = styleSource.slice(styleSource.indexOf('(chat: Project completion mirror)'));
    const rules = block.slice(0, block.indexOf('design-system:migrated-end'));
    assert.match(rules, /\.chat-bubble\.project-answer > \.message \{ max-height: var\(--project-answer-fold\); overflow: hidden; \}/);
    assert.match(rules, /\.chat-bubble\.project-answer\.is-folded > \.message \{[^}]*mask-image/);
    assert.doesNotMatch(rules, /user-select|font-size: \d|#[0-9a-fA-F]{3,6}\b/);
    // chat.js stays a caller: the decoration lives in its own module.
    assert.match(chatSource, /decorateProjectRow\(bubble, \{ role, projectId, projectName \}\)/);
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

// Host-refusal placement: an owner-initiated refusal is ONE typed
// system row bound to the target task's id, with plain `<title> · <cause>`
// text — `task_not_started` for a confirmed refusal («Not started: …») and
// `task_start_unconfirmed` when the host cannot tell («Not confirmed: …»).
// A typed keyed row is neither a terminal fact nor a plain untyped final, so it
// renders as an ordinary plain system bubble and mints/finishes no live card —
// live or on replay. Admission and steering get the same assertions.
const ORIGIN_ADDRESSED_NOTICE_ROWS = [
    {
        chat_id: 2,
        role: 'system',
        system_type: 'task_not_started',
        task_id: 'abc123',
        content: 'Аудит · Not started: the working folder can\'t be used',
        ts: '2026-09-16T00:00:04Z',
    },
    {
        chat_id: 2,
        role: 'system',
        system_type: 'task_start_unconfirmed',
        task_id: 'def456',
        content: 'Аудит · Not confirmed: the task may or may not have started',
        ts: '2026-09-16T00:00:05Z',
    },
    {
        chat_id: 2,
        role: 'system',
        system_type: 'steer_not_delivered',
        task_id: 'ghi789',
        content: 'Аудит · Not delivered: the task is in another chat',
        ts: '2026-09-18T00:00:06Z',
    },
];

function findCard(node, taskId) {
    if (node?.dataset?.taskId === taskId && node.classList?.contains('chat-live-card')) return node;
    for (const child of node?.children || []) {
        const hit = findCard(child, taskId);
        if (hit) return hit;
    }
    return null;
}

function liveCards() {
    const messages = globalThis.document.byId.get('chat-messages');
    return messages.children.filter((node) => node.classList.contains('chat-live-card'));
}

for (const noticeRow of ORIGIN_ADDRESSED_NOTICE_ROWS) {
    test(`${noticeRow.system_type} renders as a plain system bubble and mints no card, live and after reload`, async () => {
        const expectedText = new RegExp(noticeRow.content.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        let liveHtml = '';
        {
            const { prior, mount } = installDom();
            let instance;
            try {
                const made = makeInstance(mount);
                instance = made.instance;
                made.handlers.get('chat')(noticeRow);
                const bubble = findBubble('system');
                assert.ok(bubble, 'the notice row rendered a system bubble');
                assert.match(bubble.innerHTML, /📋 System/);
                assert.match(bubble.innerHTML, expectedText);
                assert.doesNotMatch(bubble.innerHTML, /<br>|<h1|<h2|md-h1|md-h2|<strong/);
                assert.equal(bubble.getAttribute('data-chat-markdown-enhanced'), '');
                assert.equal(bubble.dataset.taskId, noticeRow.task_id, 'the row keeps its target identity');
                const messages = globalThis.document.byId.get('chat-messages');
                assert.equal(findCard(messages, noticeRow.task_id), null, 'a refusal mints no live card');
                assert.equal(liveCards().length, 0);
                liveHtml = bubble.innerHTML;
            } finally {
                instance?.destroy();
                restoreDom(prior);
            }
        }
        const historyRow = {
            text: noticeRow.content,
            role: 'system',
            ts: noticeRow.ts,
            is_progress: false,
            system_type: noticeRow.system_type,
            task_id: noticeRow.task_id,
            markdown: false,
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
            assert.ok(bubble, 'history replay rendered the notice row');
            assert.equal(bubble.innerHTML, liveHtml,
                'live DOM and reload DOM are byte-identical for the notice row');
            const messages = globalThis.document.byId.get('chat-messages');
            assert.equal(findCard(messages, noticeRow.task_id), null,
                'replay does not mint a card for the refusal');
            assert.equal(liveCards().length, 0);
        } finally {
            instance?.destroy();
            restoreDom(prior);
        }
    });
}

test('a refused steer leaves the running target card open', () => {
    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        const notice = ORIGIN_ADDRESSED_NOTICE_ROWS.find((row) => row.system_type === 'steer_not_delivered');
        made.handlers.get('chat')({
            chat_id: 2, role: 'assistant', is_progress: true, content: 'Reviewing the change',
            ts: '2026-09-18T00:00:01Z', task_id: notice.task_id, cancelable: true,
        });
        const messages = globalThis.document.byId.get('chat-messages');
        const card = findCard(messages, notice.task_id);
        assert.ok(card, 'the target already has a live card');
        assert.equal(card.dataset.finished, '0');
        made.handlers.get('chat')(notice);
        assert.equal(card.dataset.finished, '0', 'a refused message is no terminal task fact');
        assert.equal(liveCards().length, 1);
        assert.match(findBubble('system').innerHTML, /📋 System/);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('render arm order and enhancement guard are pinned in source', () => {
    // The plain-system arm sits between the dedicated skill_review renderer
    // and the rich arm, whose template carries the content contract.
    const ternary = chatSource.slice(
        chatSource.indexOf("const rendered = role === 'user'"),
        chatSource.indexOf('const timeFmt =', chatSource.indexOf("const rendered = role === 'user'")),
    );
    assert.match(ternary, /renderSkillReviewDisclosure\(text, opts\.skillReview \|\| null\)/);
    assert.match(ternary, /role === 'system' && systemType !== 'skill_review' && markdown !== true\n\s+\? escapeHtml\(text\)/);
    assert.match(chatSource, /: renderChatMarkdown\(text\);/);
    // The enhancement pass skips exactly the plain-system case.
    assert.match(
        chatSource,
        /const richMarkdown = role !== 'user' && systemType !== 'skill_review' && \(role !== 'system' \|\| markdown === true\);/,
    );
});

test('chat bubble heading clamp is scoped in style.css', () => {
    // Inside chat bubbles every markdown heading is a subsection label at body
    // size (DESIGN.md §2); the global md-h1 page-size rule stays for non-chat
    // surfaces, and the live-card timeline carries its own inline clamp.
    assert.match(styleSource, /\.chat-bubble \.message \.md-h1,\n\.chat-bubble \.message \.md-h2,\n\.chat-bubble \.message \.md-h3 \{\n\s+font-size: var\(--type-body\);\n\s+font-weight: 600;\n\}/);
    // The timeline label follows its row's size: collapsed rows are meta size,
    // an expanded row is body size (DESIGN.md §5, "summary outranks details").
    // Unambiguous block scan (indent, then a non-space start): the `(\s+[^\n]+\n)*`
    // shape backtracks exponentially when an earlier same-named indented rule
    // (the @container copy of .chat-live-line-title) has no such declaration.
    const decl = (selector, declaration) => new RegExp(`\\n${selector} \\{\\n(?:[ \\t]+\\S[^\\n]*\\n)*?[ \\t]+${declaration}`);
    assert.match(styleSource, decl('\\.chat-live-line-body \\.md-h3', 'display: inline;'));
    assert.match(styleSource, decl('\\.chat-live-line-body \\.md-h3', 'font-size: inherit;'));
    assert.match(styleSource, decl('\\.chat-live-line-title', 'font-size: var\\(--type-meta\\);'));
    assert.match(styleSource, decl('\\.chat-live-line\\[data-expanded="1"\\] \\.chat-live-line-body', 'font-size: var\\(--type-body\\);'));
    assert.match(styleSource, decl('\\.chat-live-activity', 'font-size: var\\(--type-body\\);'));
    // The rich bubble renderer demotes h4-h6 to the smallest label so the clamp reaches them.
    const richSource = readFileSync(new URL('../modules/chat_markdown.js', import.meta.url), 'utf8');
    assert.match(richSource, /querySelectorAll\('h1, h2, h3, h4, h5, h6'\)[\s\S]{0,160}Math\.min\(Number\(heading\.tagName\.slice\(1\)\), 3\)/);
});

test('a host notice concludes a replayed card and keeps its markdown', async () => {
    // A terminal text the host wrote alone is projected as role=system with NO
    // system_type — exactly the shape replay needs to treat it as the task's
    // last word — and, unlike the salvage receipt, it keeps its markdown so the
    // host's own code spans do not render escaped.
    assert.match(chatSource, /const plainUntypedFinal = !msg\.system_type && !msg\.msg_type;/);
    assert.match(
        chatSource,
        /\(msg\.role === 'assistant' \|\| msg\.role === 'system'\)\n\s+&& \(positiveTaskTerminalFact\(msg\) \|\| plainUntypedFinal\)/,
    );

    const { prior, mount } = installDom();
    let instance;
    try {
        const made = makeInstance(mount);
        instance = made.instance;
        made.handlers.get('chat')({
            chat_id: 2, role: 'system', markdown: true, task_id: 'notice-task',
            content: 'Task rejected line one\nline two',
            ts: '2026-09-03T00:00:01Z',
        });
        const bubble = findBubble('system');
        assert.match(bubble.innerHTML, /Task rejected line one<br>line two/);
        assert.equal(bubble.getAttribute('data-chat-markdown-enhanced'), 'true');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
