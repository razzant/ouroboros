// Behavioural characterization of the delivered-document bubble owner,
// exercised where the code now lives. The builder is DOM-heavy but uses only
// createElement/innerHTML/querySelector/addEventListener, so a small element
// model plus a regex-free structural stub is enough to reach every branch:
// durable URL, live base64 blob, and neither.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createDocumentBubbles } from '../modules/chat_document_bubble.js';

// escapeHtmlText (utils.js) escapes by round-tripping through a detached div's
// textContent -> innerHTML, so the element model has to honour that pair.
const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;' };

function makeElement(tag) {
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        innerHTML: '',
        set textContent(value) { el.innerHTML = String(value ?? '').replace(/[&<>]/g, (c) => HTML_ESCAPES[c]); },
        get textContent() { return el.innerHTML; },
        dataset: {},
        children: [],
        listeners: {},
        removed: false,
        appendChild(child) { el.children.push(child); return child; },
        remove() { el.removed = true; },
        click() { (el.listeners.click || []).forEach((fn) => fn()); },
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        querySelector(selector) { return el.stubbedNodes?.[selector] || null; },
    };
    return el;
}

// The builder only ever asks for these two selectors; hand it back stubs so the
// wired handlers can be invoked without a real HTML parser.
function withStubbedButtons(bubble) {
    bubble.stubbedNodes = {
        '.chat-file[data-open]': makeElement('button'),
        '.chat-file-download[data-download]': makeElement('button'),
    };
    return bubble;
}

function harness({ seen = new Set() } = {}) {
    const created = [];
    const toasts = [];
    const opened = [];
    const downloaded = [];
    const inserted = [];
    const remembered = [];

    const priorDocument = globalThis.document;
    const priorAtob = globalThis.atob;
    const priorUrl = globalThis.URL;
    globalThis.document = {
        createElement: (tag) => {
            const el = withStubbedButtons(makeElement(tag));
            created.push(el);
            return el;
        },
        body: makeElement('body'),
    };

    const api = createDocumentBubbles({
        seenMessageKeys: seen,
        getSenderLabel: (role, isProgress, systemType, opts) => (
            role === 'user' ? `user:${opts?.source || ''}|${opts?.senderLabel || ''}` : 'Ouroboros'
        ),
        formatMsgTime: (raw) => (raw ? { short: `S(${raw})`, full: `F(${raw})` } : null),
        stampNodeTimestamp: (node, raw) => { node.dataset.ts = `stamped:${raw}`; return false; },
        rememberMessageKey: (key) => remembered.push(key),
        insertMessageNode: (node) => inserted.push(node),
    });

    return {
        ...api,
        seen,
        created,
        toasts,
        opened,
        downloaded,
        inserted,
        remembered,
        restore() {
            globalThis.document = priorDocument;
            globalThis.atob = priorAtob;
            globalThis.URL = priorUrl;
        },
    };
}

// ─────────────────────────────── dedup key ───────────────────────────────

test('the dedup key covers every field that distinguishes two deliveries', () => {
    const h = harness();
    try {
        assert.equal(
            h.documentMessageKey({ ts: 'T', download_url: '/api/files/download?id=1', filename: 'a.txt', caption: 'c' }),
            'document|T|/api/files/download?id=1|a.txt|c',
        );
        // Missing fields collapse to empty segments, never to undefined.
        assert.equal(h.documentMessageKey({}), 'document||||');
        // Same ts, different file: two distinct rows.
        assert.notEqual(
            h.documentMessageKey({ ts: 'T', filename: 'a.txt' }),
            h.documentMessageKey({ ts: 'T', filename: 'b.txt' }),
        );
    } finally {
        h.restore();
    }
});

test('the live frame and its history replay of one document insert exactly once', () => {
    const h = harness();
    try {
        const msg = { ts: 'T', download_url: '/api/files/download?id=7', filename: 'report.pdf' };
        assert.equal(h.appendDocumentBubble(msg), true);
        assert.equal(h.inserted.length, 1);
        assert.deepEqual(h.remembered, ['document|T|/api/files/download?id=7|report.pdf|']);

        // The dedup window is the instance's own Set; simulate what
        // rememberMessageKey does for the replay pass.
        h.seen.add(h.remembered[0]);
        assert.equal(h.appendDocumentBubble(msg), false, 'replay must not re-insert');
        assert.equal(h.inserted.length, 1);
    } finally {
        h.restore();
    }
});

// ──────────────────────────────── builder ────────────────────────────────

test('a user document is attributed through the instance sender label', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({
            role: 'user', ts: 'T', filename: 'a.txt', source: 'telegram', sender_label: 'Anton',
        });
        assert.equal(bubble.className, 'chat-bubble user');
        assert.ok(bubble.innerHTML.includes('user:telegram|Anton'));
        assert.equal(bubble.dataset.ts, 'stamped:T');
    } finally {
        h.restore();
    }
});

test('an assistant document is Ouroboros and carries the formatted time', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({ role: 'assistant', ts: 'T', filename: 'a.txt' });
        assert.equal(bubble.className, 'chat-bubble assistant');
        assert.ok(bubble.innerHTML.includes('Ouroboros'));
        assert.ok(bubble.innerHTML.includes('S(T)'));
        assert.ok(bubble.innerHTML.includes('F(T)'));
    } finally {
        h.restore();
    }
});

test('a payload-free document renders a disabled label with no controls', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({ role: 'assistant', filename: 'gone.bin' });
        assert.ok(bubble.innerHTML.includes('chat-file chat-file-empty'));
        assert.ok(!bubble.innerHTML.includes('data-open="1"'));
        assert.ok(!bubble.innerHTML.includes('data-download="1"'));
        // No handler is wired onto the stub buttons when nothing is downloadable.
        assert.deepEqual(bubble.stubbedNodes['.chat-file[data-open]'].listeners, {});
        assert.deepEqual(bubble.stubbedNodes['.chat-file-download[data-download]'].listeners, {});
    } finally {
        h.restore();
    }
});

test('only a server download path is trusted as a durable URL', () => {
    const h = harness();
    try {
        const good = h.buildDocumentBubble({ download_url: '/api/files/download?id=1', filename: 'a.txt' });
        assert.ok(good.innerHTML.includes('data-open="1"'), 'a server path is downloadable');

        const spoofed = h.buildDocumentBubble({ download_url: 'https://evil.example/x', filename: 'a.txt' });
        assert.ok(!spoofed.innerHTML.includes('data-open="1"'), 'an off-origin URL is dropped');
        assert.ok(spoofed.innerHTML.includes('chat-file-empty'));
    } finally {
        h.restore();
    }
});

test('a filename is flattened to one line and bounded', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({
            download_url: '/api/files/download?id=1',
            filename: `a\r\nb${'x'.repeat(400)}`,
        });
        assert.ok(!bubble.innerHTML.includes('\r'));
        const shown = bubble.innerHTML.match(/📎 ([^<]*)</)[1];
        assert.equal(shown.length, 200);
        assert.ok(shown.startsWith('a b'));
    } finally {
        h.restore();
    }
});

test('a downloadable document wires open and download exactly once each', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({ download_url: '/api/files/download?id=9', filename: 'a.txt' });
        const openBtn = bubble.stubbedNodes['.chat-file[data-open]'];
        const dlBtn = bubble.stubbedNodes['.chat-file-download[data-download]'];
        assert.equal((openBtn.listeners.click || []).length, 1, 'open is wired exactly once');
        assert.equal((dlBtn.listeners.click || []).length, 1, 'download is wired exactly once');
    } finally {
        h.restore();
    }
});

test('a live-only base64 payload still offers both controls', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({ file_base64: 'AAAA', filename: 'a.bin', mime: 'application/zip' });
        assert.ok(bubble.innerHTML.includes('data-open="1"'));
        assert.ok(bubble.innerHTML.includes('data-download="1"'));
        assert.equal((bubble.stubbedNodes['.chat-file[data-open]'].listeners.click || []).length, 1);
    } finally {
        h.restore();
    }
});

test('a malformed base64 payload is treated as no payload at all', () => {
    const h = harness();
    try {
        const bubble = h.buildDocumentBubble({ file_base64: 'not base64!!', filename: 'a.bin' });
        assert.ok(bubble.innerHTML.includes('chat-file-empty'));
    } finally {
        h.restore();
    }
});

test('two instances dedup independently', () => {
    const a = harness();
    const b = harness();
    try {
        const msg = { ts: 'T', filename: 'a.txt' };
        assert.equal(a.appendDocumentBubble(msg), true);
        a.seen.add(a.remembered[0]);
        assert.equal(a.appendDocumentBubble(msg), false);
        assert.equal(b.appendDocumentBubble(msg), true, 'the other thread has its own window');
    } finally {
        a.restore();
        b.restore();
    }
});
