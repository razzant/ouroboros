import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { createChatMedia, safeHttpUrl } from '../modules/chat_media.js';
import { createRebuildBatch } from '../modules/chat_render_batch.js';

const styleCss = await readFile(new URL('../style.css', import.meta.url), 'utf8');

class Classes {
    constructor(node) { this.node = node; this.values = new Set(); }
    set(value) { this.values = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
    add(...values) { values.forEach((value) => this.values.add(value)); }
    contains(value) { return this.values.has(value); }
    toggle(value, force) {
        const enabled = force === undefined ? !this.contains(value) : Boolean(force);
        if (enabled) this.add(value); else this.values.delete(value);
        return enabled;
    }
}

class NodeStub {
    constructor(tag = 'div', tracker = null) {
        this.tagName = tag.toUpperCase();
        this.tracker = tracker;
        this.children = [];
        this.parentNode = null;
        this.dataset = {};
        this.attributes = new Map();
        this.classList = new Classes(this);
        this.style = { setProperty() {} };
        this.value = '';
        this.disabled = false;
        this.paused = true;
        this.currentTime = 0;
        this.duration = 0;
        this.playbackRate = 1;
        this.loop = false;
        this.muted = false;
        this.pauseCalls = 0;
        this.selectCalls = 0;
        this.listeners = new Map();
    }
    set className(value) { this.classList.set(value); }
    get className() { return [...this.classList.values].join(' '); }
    set textContent(value) {
        this._text = String(value ?? '');
        this._html = this._text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    get textContent() { return this._text || ''; }
    set innerHTML(html) {
        this._html = String(html || '');
        this.children = [];
        const stack = [this];
        for (const token of String(html || '').matchAll(/<\/?[a-z0-9-]+(?:\s[^>]*)?>/gi)) {
            const source = token[0];
            if (source.startsWith('</')) {
                if (stack.length > 1) stack.pop();
                continue;
            }
            const tag = source.match(/^<([a-z0-9-]+)/i)?.[1] || 'div';
            const node = new NodeStub(tag, this.tracker);
            const classes = source.match(/\sclass="([^"]*)"/i)?.[1];
            if (classes) node.className = classes;
            for (const data of source.matchAll(/\sdata-([a-z0-9-]+)(?:="([^"]*)")?/gi)) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                node.dataset[key] = data[2] ?? '';
            }
            stack.at(-1).appendChild(node);
            if (!/\/$/.test(source) && !['IMG', 'INPUT', 'SOURCE'].includes(node.tagName)) stack.push(node);
        }
    }
    get innerHTML() { return this._html || ''; }
    appendChild(node) {
        node.parentNode?.removeChild(node);
        this.children.push(node);
        node.parentNode = this;
        return node;
    }
    removeChild(node) {
        const index = this.children.indexOf(node);
        if (index >= 0) this.children.splice(index, 1);
        node.parentNode = null;
    }
    remove() { this.parentNode?.removeChild(this); }
    before(node) {
        if (!this.parentNode) return;
        const index = this.parentNode.children.indexOf(this);
        this.parentNode.children.splice(index, 0, node);
        node.parentNode = this.parentNode;
    }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    removeAttribute(name) { this.attributes.delete(name); }
    addEventListener(type, fn) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(fn);
        this.tracker.adds += 1;
    }
    removeEventListener(type, fn) {
        this.listeners.get(type)?.delete(fn);
        this.tracker.removes += 1;
    }
    async click() {
        for (const listener of this.listeners.get('click') || []) await listener({ currentTarget: this });
    }
    querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
    }
    querySelectorAll(selector) {
        const matches = (node) => {
            if (selector.startsWith('.')) return node.classList.contains(selector.slice(1));
            const data = selector.match(/^\[data-([a-z0-9-]+)(?:="([^"]*)")?\]$/i);
            if (data) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                return Object.hasOwn(node.dataset, key) && (data[2] === undefined || node.dataset[key] === data[2]);
            }
            return node.tagName === selector.toUpperCase();
        };
        const found = [];
        const visit = (node) => node.children.forEach((child) => {
            if (matches(child)) found.push(child);
            visit(child);
        });
        visit(this);
        return found;
    }
    async play() { this.paused = false; }
    pause() { this.paused = true; this.pauseCalls += 1; }
    select() { this.selectCalls += 1; }
    load() {}
}

function fixture({ insertNode = null } = {}) {
    const tracker = { adds: 0, removes: 0, created: [] };
    const body = new NodeStub('body', tracker);
    const inserted = [];
    const prior = { document: globalThis.document, window: globalThis.window, navigator: globalThis.navigator };
    globalThis.document = {
        body,
        createElement: (tag) => {
            const node = new NodeStub(tag, tracker);
            tracker.created.push(node);
            return node;
        },
        createDocumentFragment: () => new NodeStub('#fragment', tracker),
        execCommand: () => true,
    };
    globalThis.window = { open() {} };
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: { clipboard: { writeText: async () => {} } },
    });
    const controller = createChatMedia({
        chatSessionId: 'session',
        durableChatMediaUrl: (value) => String(value || ''),
        formatMsgTime: () => null,
        insertMessageNode(node) {
            if (insertNode) insertNode(node);
            else body.appendChild(node);
            inserted.push(node);
        },
        senderLabel: () => 'Owner',
        stampNodeTimestamp(node, raw) { node.dataset.ts = String(raw || ''); },
    });
    return { controller, inserted, tracker, restore: () => {
        globalThis.document = prior.document;
        globalThis.window = prior.window;
        Object.defineProperty(globalThis, 'navigator', { configurable: true, value: prior.navigator });
    } };
}

test('safeHttpUrl accepts only absolute HTTP(S) URLs', () => {
    assert.equal(safeHttpUrl('https://example.com/a'), 'https://example.com/a');
    assert.equal(safeHttpUrl('http://example.com/'), 'http://example.com/');
    assert.equal(safeHttpUrl('/relative'), '');
    assert.equal(safeHttpUrl('javascript:alert(1)'), '');
    assert.equal(safeHttpUrl('data:text/plain,no'), '');
});

test('photo grouping is keyed by role and task without cross-task merging', () => {
    const fx = fixture();
    try {
        const photo = (task_id, role = 'assistant') => ({
            type: 'photo', role, task_id, image_base64: 'aGVsbG8=', mime: 'image/png',
            ts: `2026-08-30T00:00:0${fx.inserted.length}Z`,
        });
        for (const msg of [photo('task-a'), photo('task-a'), photo('task-b'), photo('task-a', 'user')]) {
            assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        }
        assert.equal(fx.inserted.length, 3, 'same role/task merged while other task and role stayed separate');
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 2);
        assert.equal(fx.inserted[0].classList.contains('is-multiple'), true);
        assert.equal(fx.inserted[0].querySelector('.chat-group-title').textContent, 'Multiple images');
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('photos with an empty task id remain separate bubbles', () => {
    const fx = fixture();
    try {
        const photo = {
            type: 'photo', role: 'assistant', task_id: '', image_base64: 'aGVsbG8=', mime: 'image/png',
        };
        assert.equal(fx.controller.buildGallery('photos', photo, fx.controller.buildMediaBubble(photo)), true);
        assert.equal(fx.controller.buildGallery('photos', photo, fx.controller.buildMediaBubble(photo)), true);
        assert.equal(fx.inserted.length, 2);
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 1);
        assert.equal(fx.inserted[1].querySelectorAll('.chat-gallery-item').length, 1);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('an intervening message breaks photo adjacency: same key starts a new gallery in feed order', () => {
    const fx = fixture();
    try {
        const photo = (ts) => ({
            type: 'photo', role: 'assistant', task_id: 'task-a',
            image_base64: 'aGVsbG8=', mime: 'image/png', ts,
        });
        let msg = photo('2026-08-30T00:00:00Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        // A plain text bubble lands below the gallery (inserted by chat.js,
        // outside chat_media) — the wrapper is no longer the feed tail.
        const text = new NodeStub('div', fx.tracker);
        text.className = 'chat-bubble';
        globalThis.document.body.appendChild(text);
        msg = photo('2026-08-30T00:00:02Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);

        assert.equal(fx.inserted.length, 2, 'a second wrapper starts instead of teleporting up');
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 1);
        assert.equal(fx.inserted[1].querySelectorAll('.chat-gallery-item').length, 1);
        assert.deepEqual(
            globalThis.document.body.children,
            [fx.inserted[0], text, fx.inserted[1]],
            'timeline order is preserved: gallery, text, gallery',
        );
        // The map keeps the LATEST wrapper: the next contiguous photo joins it.
        msg = photo('2026-08-30T00:00:03Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        assert.equal(fx.inserted.length, 2);
        assert.equal(fx.inserted[1].querySelectorAll('.chat-gallery-item').length, 2);
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 1);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('a trailing typing indicator does not break photo grouping', () => {
    const fx = fixture();
    try {
        const photo = (ts) => ({
            type: 'photo', role: 'assistant', task_id: 'task-a',
            image_base64: 'aGVsbG8=', mime: 'image/png', ts,
        });
        let msg = photo('2026-08-30T00:00:00Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        const typing = new NodeStub('div', fx.tracker);
        typing.className = 'chat-bubble typing-bubble';
        globalThis.document.body.appendChild(typing);
        msg = photo('2026-08-30T00:00:01Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        assert.equal(fx.inserted.length, 1, 'typing indicator does not split the gallery');
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 2);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('live media and quiz-state writes use the injected content boundary', () => {
    const fx = fixture();
    const handlers = new Map();
    const seen = new Set();
    let mutations = 0;
    let unread = 0;
    try {
        fx.controller.wireDeliveries({
            onWs(type, handler) { handlers.set(type, handler); },
            isMyThread: () => true,
            hideTypingIndicatorOnly() {},
            syncChatStatus() {},
            incrementUnreadIfNeeded() { unread += 1; },
            seenMessageKeys: seen,
            rememberMessageKey(key) { seen.add(key); },
            chatMediaMessageKey: (msg) => `media:${msg.task_id}:${msg.ts}`,
            documentMessageKey: (msg) => `doc:${msg.task_id}:${msg.ts}`,
            buildQuizCard: () => null,
            applyQuizStateFrame: (_root, msg) => msg.changed === true,
            messagesRoot: () => ({}),
            deliverContentMutation(mutate) { mutations += 1; return mutate(); },
        });
        handlers.get('photo')({
            type: 'photo', role: 'assistant', task_id: 'task-a', ts: 'one',
            image_base64: 'aGVsbG8=', mime: 'image/png',
        });
        handlers.get('photo')({
            type: 'photo', role: 'assistant', task_id: 'task-a', ts: 'two',
            image_base64: 'aGVsbG8=', mime: 'image/png',
        });
        handlers.get('quiz_state')({ quiz_id: 'q1', changed: false });
        handlers.get('quiz_state')({ quiz_id: 'q1', changed: true });
        assert.equal(mutations, 4);
        assert.equal(unread, 2);
        assert.equal(fx.inserted.length, 1);
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 2);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('an intervening message breaks file-card adjacency the same way', () => {
    const fx = fixture();
    try {
        const doc = (ts, filename) => ({
            type: 'document', role: 'assistant', task_id: 'task-a', filename,
            mime: 'application/pdf', file_base64: 'aGVsbG8=', size_bytes: 5, ts,
        });
        let msg = doc('2026-08-30T00:00:00Z', 'one.pdf');
        assert.equal(fx.controller.buildGallery('files', msg, fx.controller.buildDocumentBubble(msg)), true);
        const text = new NodeStub('div', fx.tracker);
        text.className = 'chat-bubble';
        globalThis.document.body.appendChild(text);
        msg = doc('2026-08-30T00:00:02Z', 'two.pdf');
        assert.equal(fx.controller.buildGallery('files', msg, fx.controller.buildDocumentBubble(msg)), true);

        assert.equal(fx.inserted.length, 2);
        assert.equal(fx.inserted[0].querySelectorAll('.chat-file-item').length, 1);
        assert.equal(fx.inserted[1].querySelectorAll('.chat-file-item').length, 1);
        assert.deepEqual(globalThis.document.body.children, [fx.inserted[0], text, fx.inserted[1]]);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('rebuild replay keeps gallery adjacency via the batch holding fragment', () => {
    let batch = null;
    const fx = fixture({ insertNode: (node) => batch.collect(node) });
    try {
        batch = createRebuildBatch(globalThis.document);
        const photo = (ts) => ({
            type: 'photo', role: 'assistant', task_id: 'task-a',
            image_base64: 'aGVsbG8=', mime: 'image/png', ts,
        });
        // Contiguous photos merge even while detached inside the batch.
        let msg = photo('2026-08-30T00:00:00Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        msg = photo('2026-08-30T00:00:01Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        assert.equal(fx.inserted.length, 1);
        assert.equal(fx.inserted[0].querySelectorAll('.chat-gallery-item').length, 2);
        // A text bubble collected between photos breaks adjacency during replay too.
        const text = new NodeStub('div', fx.tracker);
        text.className = 'chat-bubble';
        batch.collect(text);
        msg = photo('2026-08-30T00:00:03Z');
        assert.equal(fx.controller.buildGallery('photos', msg, fx.controller.buildMediaBubble(msg)), true);
        assert.equal(fx.inserted.length, 2, 'replay starts a new gallery after the text bubble');
        assert.equal(fx.inserted[1].querySelectorAll('.chat-gallery-item').length, 1);
        // The holding fragment preserves arrival (chronological) order.
        assert.deepEqual(text.parentNode.children, [fx.inserted[0], text, fx.inserted[1]]);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('media and document builders render upgraded DOM shapes with type-anchored MIME', () => {
    const fx = fixture();
    try {
        const photo = fx.controller.buildMediaBubble({
            type: 'photo', role: 'assistant', image_base64: 'aGVsbG8=', mime: 'text/html', caption: 'diagram',
        });
        assert.match(photo.innerHTML, /class="chat-photo"/);
        assert.match(photo.innerHTML, /data:image\/png;base64,aGVsbG8=/);
        assert.doesNotMatch(photo.innerHTML, /data:text\/html/);
        assert.match(photo.innerHTML, /aria-label="Photo actions"/);
        const video = fx.controller.buildMediaBubble({
            type: 'video', role: 'assistant', video_base64: 'aGVsbG8=', mime: 'video/mp4', caption: 'demo',
        });
        assert.match(video.innerHTML, /<video preload="metadata"/);
        assert.match(video.innerHTML, /aria-label="Playback speed"/);
        assert.match(video.innerHTML, /data-media-action="fullscreen"/);
        const audio = fx.controller.buildDocumentBubble({
            type: 'document', role: 'assistant', filename: 'briefing.mp3', mime: 'audio/mpeg',
            file_base64: 'aGVsbG8=', size_bytes: 5,
        });
        assert.ok(audio.querySelector('.chat-media-player').classList.contains('is-audio'));
        assert.ok(audio.querySelector('audio'));
        const document = fx.controller.buildDocumentBubble({
            type: 'document', role: 'assistant', filename: 'report.pdf', mime: 'application/pdf',
            file_base64: 'aGVsbG8=', size_bytes: 5,
        });
        assert.match(document.innerHTML, /class="chat-file-grid"/);
        assert.match(document.innerHTML, /class="chat-file-card"/);
        assert.match(document.innerHTML, /PDF · 5 B/);
        const actions = Array.from({ length: 13 }, (_value, index) => ({
            label: `Link ${index}`,
            url: index === 1 ? 'javascript:alert(1)' : `https://example.com/${index}`,
        }));
        const links = fx.controller.buildLinksMessage({ type: 'links', role: 'assistant', actions });
        assert.equal(links.querySelectorAll('.chat-link-button').length, 12,
            'unsafe actions are removed before the first twelve valid actions are selected');
        assert.match(links.innerHTML, /rel="noopener noreferrer"/);
        assert.doesNotMatch(links.innerHTML, /javascript:/);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('file dialog and photo menu expose no Share action (owner removal)', async () => {
    const fx = fixture();
    try {
        const bubble = fx.controller.buildDocumentBubble({
            type: 'document', role: 'assistant', filename: 'notes.txt', mime: 'text/plain',
            file_base64: 'aGVsbG8=', ts: '2026-08-30T00:00:00Z',
        });
        await bubble.querySelector('.chat-file-card').click();
        const dialog = globalThis.document.body.querySelector('.chat-file-dialog');
        assert.ok(dialog, 'card click creates the action dialog');
        assert.ok(dialog.querySelector('[data-file-action="download"]'));
        assert.ok(dialog.querySelector('[data-file-action="close"]'));
        assert.equal(dialog.querySelector('[data-file-action="share"]'), null,
            'Share was removed everywhere by owner decision');
        const photo = fx.controller.buildMediaBubble({
            type: 'photo', role: 'assistant', image_base64: 'aGVsbG8=', mime: 'image/png',
        });
        assert.ok(photo.querySelector('[data-photo-action="copy"]'));
        assert.equal(photo.querySelector('[data-photo-action="share"]'), null);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('video download filename follows the validated MIME subtype', async () => {
    const fx = fixture();
    const originalCreateObjectURL = URL.createObjectURL;
    const originalRevokeObjectURL = URL.revokeObjectURL;
    try {
        URL.createObjectURL = () => 'blob:test-video';
        URL.revokeObjectURL = () => {};
        const video = fx.controller.buildMediaBubble({
            type: 'video', role: 'assistant', video_base64: 'aGVsbG8=', mime: 'video/webm',
        });

        await video.querySelector('[data-media-action="download"]').click();

        const anchor = fx.tracker.created.filter((node) => node.tagName === 'A').at(-1);
        assert.equal(anchor.download, 'video.webm');
    } finally {
        URL.createObjectURL = originalCreateObjectURL;
        URL.revokeObjectURL = originalRevokeObjectURL;
        fx.controller.destroy();
        fx.restore();
    }
});

test('rejected clipboard writeText falls back to textarea copy', async () => {
    const fx = fixture();
    try {
        let execCalls = 0;
        globalThis.document.execCommand = (command) => {
            execCalls += 1;
            assert.equal(command, 'copy');
            return true;
        };
        Object.defineProperty(globalThis, 'navigator', {
            configurable: true,
            value: { clipboard: { writeText: async () => { throw new Error('denied'); } } },
        });
        const bubble = new NodeStub('div', fx.tracker);
        const button = fx.controller.attachCopyControl(bubble, 'raw message');

        await button.click();

        assert.equal(execCalls, 1);
        assert.equal(button.textContent, '✓');
        assert.equal(button.attributes.get('aria-label'), 'Message copied');
        assert.equal(fx.tracker.created.filter((node) => node.tagName === 'TEXTAREA').at(-1).selectCalls, 1);
        assert.equal(globalThis.document.body.querySelectorAll('.chat-copy-fallback').length, 0);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('throwing execCommand reports failure and removes the fallback textarea', async () => {
    const fx = fixture();
    try {
        globalThis.document.execCommand = () => { throw new Error('copy failed'); };
        Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {} });
        const bubble = new NodeStub('div', fx.tracker);
        const button = fx.controller.attachCopyControl(bubble, 'raw message');

        await button.click();

        assert.equal(button.textContent, '✗');
        assert.equal(button.title, 'Copy failed');
        assert.equal(button.attributes.get('aria-label'), 'Copy failed');
        assert.equal(globalThis.document.body.querySelectorAll('.chat-copy-fallback').length, 0);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('copy fallback reports failure when execCommand is unavailable', async () => {
    const fx = fixture();
    try {
        globalThis.document.execCommand = undefined;
        Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {} });
        const bubble = new NodeStub('div', fx.tracker);
        const button = fx.controller.attachCopyControl(bubble, 'raw message');
        await button.click();
        assert.equal(button.textContent, '✗');
        assert.equal(button.attributes.get('aria-label'), 'Copy failed');
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('copy control is an always-visible icon button that marks the bubble (D12)', async () => {
    const fx = fixture();
    try {
        const bubble = new NodeStub('div', fx.tracker);
        const button = fx.controller.attachCopyControl(bubble, 'raw message');

        assert.equal(button.type, 'button');
        assert.equal(button.title, 'Copy');
        assert.equal(button.attributes.get('aria-label'), 'Copy message');
        assert.ok(button.querySelector('svg'), 'inline copy-icon SVG is present');
        assert.match(button.innerHTML, /currentColor/);
        assert.equal(bubble.classList.contains('has-copy'), true,
            'bubble carries has-copy so CSS reserves the timestamp gutter');

        await button.click();
        assert.equal(button.textContent, '✓', 'success swaps the icon for a checkmark');
        assert.equal(button.title, 'Message copied', 'title swaps in step with aria-label');
        assert.equal(button.attributes.get('aria-label'), 'Message copied');
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('stylesheet pins the always-visible copy icon and the timestamp reserve (D12)', () => {
    // (a) The copy control is anchored bottom-right and PERMANENTLY visible:
    // a non-zero base opacity, never the old hover-reveal opacity 0.
    assert.match(styleCss, /\.chat-message-copy\s*\{[^}]*position:\s*absolute/);
    assert.match(styleCss, /\.chat-message-copy\s*\{[^}]*right:\s*\d+px/);
    assert.match(styleCss, /\.chat-message-copy\s*\{[^}]*bottom:\s*\d+px/);
    // The fractional part must carry a non-zero digit: 0.0 / .0 are invisible.
    assert.match(styleCss, /\.chat-message-copy\s*\{[^}]*opacity:\s*0?\.\d*[1-9]/);
    assert.doesNotMatch(styleCss, /\.chat-message-copy\s*\{[^}]*opacity:\s*0\s*;/);
    // (b) has-copy bubbles reserve a right gutter so the timestamp can never
    // sit under the icon (the structural overlap fix); 0px is no reserve.
    assert.match(styleCss, /\.chat-bubble\.has-copy\s+\.msg-time\s*\{[^}]*margin-right:\s*[1-9]\d*px/);
});

test('reset disposes listeners, stops players, clears groups, and destroy is final', () => {
    const fx = fixture();
    try {
        const msg = {
            type: 'video', role: 'assistant', task_id: 'task-video',
            video_base64: 'aGVsbG8=', mime: 'video/mp4', ts: '2026-08-30T00:00:00Z',
        };
        const bubble = fx.controller.buildMediaBubble(msg);
        const media = bubble.querySelector('video');
        fx.inserted.push(bubble);
        globalThis.document.body.appendChild(bubble);

        fx.controller.reset();

        assert.ok(fx.tracker.adds > 0);
        assert.equal(fx.tracker.removes, fx.tracker.adds);
        assert.ok(media.pauseCalls >= 1);
        assert.equal(globalThis.document.body.children.length, 1, 'the caller owns ordinary bubble removal');
        fx.controller.destroy();
        assert.equal(fx.controller.buildMediaBubble(msg), null);
        fx.controller.destroy();
    } finally {
        fx.restore();
    }
});

test('media host-bridge calls prefer the compat URL while the browser keeps the canonical one', async () => {
    const fx = fixture();
    const bridged = [];
    globalThis.window.pywebview = {
        api: {
            download_file_to_downloads: async (url, name, external) => {
                bridged.push([url, name, external]);
                return { ok: true };
            },
        },
    };
    try {
        const canonical = '/api/tasks/t-1/artifacts/chat-media-aa.png';
        const compat = '/api/files/download?path=tasks/t-1/chat-media-aa.png';
        const photo = fx.controller.buildMediaBubble({
            type: 'photo',
            role: 'assistant',
            mime: 'image/png',
            download_url: canonical,
            download_url_compat: compat,
        });
        // The rendered element addresses the canonical route: the browser has
        // no gate, and the compat form is only an alternative address.
        assert.ok(photo.innerHTML.includes(`src="${canonical}"`), photo.innerHTML);
        await photo.querySelector('[data-photo-action="download"]').click();
        assert.deepEqual(bridged, [[compat, 'image.png', false]]);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('a media frame without a compat URL still reaches the bridge on the canonical route', async () => {
    const fx = fixture();
    const bridged = [];
    globalThis.window.pywebview = {
        api: {
            download_file_to_downloads: async (url) => { bridged.push(url); return { ok: true }; },
        },
    };
    try {
        const canonical = '/api/tasks/t-1/artifacts/chat-media-bb.png';
        const photo = fx.controller.buildMediaBubble({
            type: 'photo', role: 'assistant', mime: 'image/png', download_url: canonical,
        });
        await photo.querySelector('[data-photo-action="download"]').click();
        assert.deepEqual(bridged, [canonical]);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('a compat URL that is not the files-download form is rejected, not trusted', async () => {
    const fx = fixture();
    const bridged = [];
    globalThis.window.pywebview = {
        api: {
            download_file_to_downloads: async (url) => { bridged.push(url); return { ok: true }; },
        },
    };
    try {
        const canonical = '/api/tasks/t-1/artifacts/chat-media-cc.png';
        const photo = fx.controller.buildMediaBubble({
            type: 'photo',
            role: 'assistant',
            mime: 'image/png',
            download_url: canonical,
            download_url_compat: 'https://evil.example/steal',
        });
        await photo.querySelector('[data-photo-action="download"]').click();
        assert.deepEqual(bridged, [canonical]);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('a live data: photo still hands the bridge the frame addresses', () => {
    const { controller, restore } = fixture();
    try {
        const msg = {
            msg_type: 'photo', task_id: 't9', ts: '2026-09-01T10:00:00+00:00',
            mime: 'image/png', image_base64: 'aGk=',
            download_url: '/api/tasks/t9/artifacts/chat-media-' + 'a'.repeat(64) + '.png',
            download_url_compat: '/api/files/download?path=x/chat-media.png',
        };
        const bubble = controller.buildMediaBubble(msg);
        assert.ok(bubble, 'bubble built from base64');
        const html = String(bubble.innerHTML || '');
        assert.ok(html.includes('data:image/png'), 'display stays base64');
        assert.ok(!html.includes('/api/files/download'), 'compat address is bridge-only, not the display');
    } finally { restore(); }
});
