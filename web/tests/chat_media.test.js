import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { bindComposerFileTargets, safeHttpUrl } from '../modules/chat_media.js';
import { createComposerAttachments } from '../modules/chat_attachments.js';
import { uploadView } from './helpers/attachment_views.js';
import { NodeStub, fixture } from './helpers/media_dom.js';

const styleCss = await readFile(new URL('../style.css', import.meta.url), 'utf8');

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
            messagesRoot: () => globalThis.document.body,
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
            type: 'document', role: 'assistant', filename: 'notes.pdf', mime: 'application/pdf',
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

test('only the delivered copy of a Markdown or text file offers Read; other files keep the file dialog', () => {
    const fx = fixture();
    try {
        const card = (extra) => fx.controller.buildDocumentBubble({
            type: 'document', role: 'assistant', ts: '2026-08-30T00:00:00Z', task_id: 'task-1', ...extra,
        }).querySelector('.chat-file-more');
        const readable = (more) => more.classList.contains('is-read');
        // Live inline bytes and the immutable task-artifact route are the delivered copy.
        assert.equal(readable(card({ filename: 'report.md', file_base64: 'IyBIaQ==' })), true);
        assert.equal(readable(card({ filename: 'notes.txt', download_url: '/api/tasks/task-1/artifacts/notes.txt' })), true);
        // A current file-browser path is not: Read is not offered, Open/Download stay.
        assert.equal(readable(card({ filename: 'report.md', download_url: '/api/files/download?path=report.md' })), false);
        assert.equal(readable(card({ filename: 'page.html', mime: 'text/plain', file_base64: 'PGI+' })), false);
        assert.equal(readable(card({ filename: 'report.pdf', file_base64: 'JVBERg==' })), false);
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

test('releasing one history subtree stops only its player and leaves another page interactive', async () => {
    const fx = fixture();
    try {
        const roots = ['older', 'newer'].map((id) => {
            const root = new NodeStub('section', fx.tracker);
            const bubble = fx.controller.buildMediaBubble({
                type: 'video', role: 'assistant', task_id: id, history_id: `history:${id}`,
                video_base64: 'aGVsbG8=', mime: 'video/mp4', ts: '2026-09-12T12:00:00Z',
            });
            root.appendChild(bubble);
            globalThis.document.body.appendChild(root);
            return { root, bubble, player: bubble.querySelector('video'),
                play: bubble.querySelector('[data-media-action="play"]') };
        });
        const [older, newer] = roots;
        await older.play.click();
        await newer.play.click();
        assert.equal(older.player.paused, false);
        assert.equal(newer.player.paused, false);
        const retainedListeners = [...newer.play.listeners.values()].reduce((sum, handlers) => sum + handlers.size, 0);
        fx.controller.release(older.root);
        assert.equal(older.player.paused, true);
        assert.ok(older.player.pauseCalls > 0);
        assert.equal(newer.player.paused, false);
        assert.equal(newer.player.pauseCalls, 0);
        assert.equal(newer.root.parentNode, globalThis.document.body);
        assert.equal([...newer.play.listeners.values()].reduce((sum, handlers) => sum + handlers.size, 0), retainedListeners);
        assert.equal(older.play.listeners.get('click').size, 0);
        await newer.play.click();
        assert.equal(newer.player.paused, true, 'retained controls still call the surviving player');
        assert.equal(newer.player.pauseCalls, 1);
        const removed = fx.tracker.removes;
        fx.controller.release(older.root);
        assert.equal(fx.tracker.removes, removed, 'release is idempotent for its own subtree');
        fx.controller.destroy();
        assert.equal(fx.tracker.removes, fx.tracker.adds, 'remaining resources close once at final teardown');
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('live gallery items adopt separate history identities and older photos do not join the newer tail', () => {
    const fx = fixture();
    try {
        const seen = new Set();
        const delivery = fx.controller.wireDeliveries({
            onWs() {}, isMyThread: () => true, hideTypingIndicatorOnly() {}, syncChatStatus() {},
            incrementUnreadIfNeeded() {}, seenMessageKeys: seen,
            rememberMessageKey: (key) => seen.add(key),
            chatMediaMessageKey: (msg) => `photo:${msg.task_id}:${msg.ts}`,
            documentMessageKey: (msg) => `file:${msg.task_id}:${msg.ts}`,
            messagesRoot: () => globalThis.document.body,
        });
        const photo = (seconds) => ({ type: 'photo', role: 'assistant', task_id: 'gallery',
            image_base64: 'aGVsbG8=', mime: 'image/png', ts: `2026-09-12T12:00:0${seconds}Z` });
        delivery.appendMediaBubble(photo(2));
        delivery.appendMediaBubble(photo(3));
        const newerWrapper = fx.inserted[0];
        const originalItems = newerWrapper.querySelectorAll('.chat-gallery-item');
        assert.equal(originalItems.length, 2);
        for (const seconds of [2, 3]) assert.equal(delivery.appendMediaBubble({
            ...photo(seconds), history_id: `chat:${seconds}`,
            history_position: { source: 'chat:rotation-1', offset: seconds * 100 },
        }), false);
        assert.deepEqual(newerWrapper.querySelectorAll('.chat-gallery-item'), originalItems);
        assert.deepEqual(originalItems.map((item) => item.dataset.historyId), ['chat:2', 'chat:3']);
        assert.equal(newerWrapper.dataset.historyId, undefined, 'physical IDs belong to items, not their shared gallery');
        seen.clear(); // The live-key FIFO can expire while this page stays mounted.
        for (const seconds of [2, 3]) assert.equal(delivery.appendMediaBubble({
            ...photo(seconds), history_id: `chat:${seconds}`,
        }), false);
        assert.deepEqual(newerWrapper.querySelectorAll('.chat-gallery-item'), originalItems);
        delivery.appendMediaBubble({ ...photo(1), history_id: 'chat:1' });
        assert.equal(fx.inserted.length, 2, 'older history obtains its own chronologically placed wrapper');
        assert.deepEqual(newerWrapper.querySelectorAll('.chat-gallery-item'), originalItems);
        assert.equal(fx.inserted[1].querySelector('.chat-gallery-item').dataset.historyId, 'chat:1');
        fx.controller.release(fx.inserted[1]);
        assert.equal(newerWrapper.parentNode, globalThis.document.body);
        assert.deepEqual(newerWrapper.querySelectorAll('.chat-gallery-item'), originalItems);
    } finally {
        fx.controller.destroy();
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

test('a task-incident toast carries the frame tone; absent tone keeps the alarm (#628)', async () => {
    const { showTaskIncidentToast } = await import('../modules/chat_media.js');
    const priorDocument = globalThis.document;
    const created = [];
    const tracker = { adds: 0, removes: 0 };
    const stack = new NodeStub('div', tracker);
    globalThis.document = {
        getElementById: () => stack,
        createElement: (tag) => { const node = new NodeStub(tag, tracker); created.push(node); return node; },
        body: { appendChild() {} },
    };
    const priorTimeout = globalThis.setTimeout;
    globalThis.setTimeout = () => 0;
    try {
        const recovered = showTaskIncidentToast({
            task_incident: 'network_wait', toast_once: 'eph:network_wait:recovered:1',
            toast_tone: 'ok', content: 'Provider connection restored — resuming.',
        });
        assert.ok(recovered, 'a new incident key renders a toast');
        assert.ok(recovered.classList.contains('toast-ok'), recovered.className);
        assert.equal(recovered.getAttribute('role'), 'status');
        const waiting = showTaskIncidentToast({
            task_incident: 'network_wait', toast_once: 'eph:network_wait:entered:1', toast_tone: 'warn',
            content: 'Could not establish a provider connection — waiting.',
        });
        assert.ok(waiting.classList.contains('toast-warn'));
        // No tone on the frame (older producers, cancellation_fault): the alarm.
        const fault = showTaskIncidentToast({
            task_incident: 'cancellation_fault', toast_once: 'root:cancellation_fault', content: 'did not settle',
        });
        assert.ok(fault.classList.contains('toast-danger'));
        assert.equal(fault.getAttribute('role'), 'alert');
        // An unknown tone spelling never crashes and never lands on a made-up class.
        const odd = showTaskIncidentToast({
            task_incident: 'network_wait', toast_once: 'eph:network_wait:ended:1', toast_tone: 'purple', content: 'x',
        });
        assert.ok(odd.classList.contains('toast-danger'));
        // Dedupe by toast_once is unchanged: the same key renders nothing again.
        assert.equal(showTaskIncidentToast({
            task_incident: 'network_wait', toast_once: 'eph:network_wait:recovered:1', toast_tone: 'ok', content: 'again',
        }), null);
        assert.equal(created.length, 4);
    } finally {
        globalThis.document = priorDocument;
        globalThis.setTimeout = priorTimeout;
    }
});

// The two implicit file routes into the composer. The Playwright smoke drives
// the real drop path in a browser; the paste path has no clipboard case there,
// so it is pinned here instead.
function composerTargetStub() {
    const listeners = new Map();
    const classes = new Set();
    return {
        classes,
        classList: {
            toggle(name, force) {
                if (force) classes.add(name); else classes.delete(name);
            },
        },
        addEventListener(type, fn) { listeners.set(type, fn); },
        fire(type, event) {
            const fn = listeners.get(type);
            assert.ok(fn, `no ${type} listener bound`);
            fn(event);
            return event;
        },
    };
}

test('bindComposerFileTargets stages pasted images and dropped files', () => {
    const page = composerTargetStub();
    const inputArea = composerTargetStub();
    const input = composerTargetStub();
    const staged = [];
    bindComposerFileTargets({ page, inputArea, input, stagePendingFiles: (files) => staged.push(...Array.from(files)) });

    // A clipboard image is staged under a generated name; the browser's own
    // paste is suppressed so the image never lands in the textarea as text.
    let prevented = 0;
    const pasteEvent = (items) => ({ preventDefault() { prevented += 1; }, clipboardData: { items } });
    input.fire('paste', pasteEvent([{
        kind: 'file', type: 'image/png', getAsFile: () => new File(['x'], 'pasted.png', { type: 'image/png' }),
    }]));
    assert.equal(prevented, 1);
    assert.equal(staged.length, 1);
    assert.match(staged[0].name, /^clipboard-\d+\.png$/);

    // Ordinary text keeps the native paste: nothing staged, nothing prevented.
    input.fire('paste', pasteEvent([{ kind: 'string', type: 'text/plain', getAsFile: () => null }]));
    assert.equal(prevented, 1);
    assert.equal(staged.length, 1);

    // A file drag arms the input-area affordance and disarms on leave.
    const fileDrag = (files = []) => ({
        preventDefault() {}, dataTransfer: { types: ['Files'], files, dropEffect: '' },
    });
    page.fire('dragenter', fileDrag());
    assert.ok(inputArea.classes.has('drag-active'));
    assert.equal(page.fire('dragover', fileDrag()).dataTransfer.dropEffect, 'copy');
    page.fire('dragleave', fileDrag());
    assert.ok(!inputArea.classes.has('drag-active'));

    // The drop stages its files and always clears the affordance.
    page.fire('dragenter', fileDrag());
    page.fire('drop', fileDrag([new File(['y'], 'dropped.txt', { type: 'text/plain' })]));
    assert.ok(!inputArea.classes.has('drag-active'));
    assert.equal(staged.length, 2);
    assert.equal(staged[1].name, 'dropped.txt');

    // A drag carrying no files is not ours: never captured, never armed.
    page.fire('dragenter', {
        preventDefault() { throw new Error('a text drag must keep its default'); },
        dataTransfer: { types: ['text/plain'] },
    });
    assert.ok(!inputArea.classes.has('drag-active'));
});

// --- Owner attachments (DESIGN "Chat attachments") ---------------------------------

test('owner attachments mount above the caption from the shared atoms and release with the instance', () => {
    const fx = fixture();
    try {
        const bubble = new NodeStub('div', fx.tracker);
        bubble.innerHTML = '<div class="sender">You</div><div class="message">Caption</div><div class="msg-time">now</div>';
        const views = [uploadView('one.png', 'image'), uploadView('two.png', 'image'), uploadView('note.wav', 'audio'),
            uploadView('plan.pdf', 'file'), { name: 'gone.zip', kind: 'file', mime: '', available: false }];
        const adds = fx.tracker.adds;
        assert.equal(fx.controller.mountAttachments(bubble, views, 'Caption'), true);
        const [block, message] = [bubble.children[1], bubble.children[2]];
        assert.equal(block.classList.contains('chat-attachments') && message.classList.contains('message'), true);
        assert.equal(bubble.classList.contains('has-attachments') && bubble.classList.contains('has-wide-media'), true);
        assert.equal(block.querySelector('.chat-gallery-grid').classList.contains('is-multiple'), true);
        assert.deepEqual(['.chat-photo', 'summary', 'audio', '.chat-file-card'].map((sel) => block.querySelectorAll(sel).length),
            [2, 2, 1, 2], 'two photos with always-present actions (no hover), one player, two cards');
        assert.ok(block.querySelectorAll('.chat-file-item')[1].classList.contains('is-unavailable'));
        assert.ok(block.innerHTML.includes('ZIP · Unavailable'),
            'an unavailable card says so in words, not only by its dimmed style');
        assert.equal(block.innerHTML.match(/aria-haspopup="dialog"/g)?.length, 1,
            'the live card announces the file dialog it opens, as a delivered card does; the inert one opens nothing');
        assert.ok(fx.tracker.adds > adds, 'listeners belong to the media controller');
        const single = new NodeStub('div', fx.tracker);
        single.innerHTML = '<div class="sender">You</div><div class="message"></div>';
        fx.controller.mountAttachments(single, [uploadView('only.png', 'image')], '');
        assert.equal(single.querySelector('.message').hidden, true, 'an empty caption leaves no empty text row');
        assert.equal(single.classList.contains('has-wide-media'), false, 'one photo shrinks to fit');
        const removes = fx.tracker.removes;
        fx.controller.release(bubble);
        assert.ok(fx.tracker.removes > removes, 'evicting the bubble disposes its attachment listeners');
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});

test('an undecodable preview becomes an honest card; a missing file becomes inert', async () => {
    const fx = fixture();
    const priorFetch = globalThis.fetch;
    try {
        for (const [status, inert, note] of [[200, false, 'Preview unavailable'], [404, true, 'Unavailable']]) {
            globalThis.fetch = async (_url, init) => ({ ok: status < 400, status, method: init?.method });
            const bubble = new NodeStub('div', fx.tracker);
            bubble.innerHTML = '<div class="message">x</div>';
            fx.controller.mountAttachments(bubble, [uploadView('clip.heic', 'image')], 'x');
            const block = bubble.children[0];
            for (const listener of block.querySelector('.chat-photo').listeners.get('error')) await listener({});
            await new Promise((resolve) => setTimeout(resolve, 0));
            assert.equal(block.querySelectorAll('.chat-photo').length, 0);
            assert.ok(block.querySelector('.chat-file-card'), 'the image became a card');
            assert.ok(fx.tracker.created.some((made) => made.innerHTML.includes(`· ${note}`)), note);
            assert.equal(block.querySelector('.chat-file-item').classList.contains('is-unavailable'), inert);
        }
    } finally {
        globalThis.fetch = priorFetch;
        fx.controller.destroy();
        fx.restore();
    }
});

test('a late HEAD answer for a retired message wires nothing: the fallback ends with its owner', async () => {
    const fx = fixture();
    const priorFetch = globalThis.fetch;
    try {
        for (const end of ['release', 'destroy']) {
            let answer = null;
            globalThis.fetch = (_url, init) => new Promise((resolve) => {
                answer = () => resolve({ ok: true, status: 200, method: init?.method });
            });
            const feed = new NodeStub('div', fx.tracker);
            const bubble = feed.appendChild(new NodeStub('div', fx.tracker));
            bubble.innerHTML = '<div class="message">x</div>';
            fx.controller.mountAttachments(bubble, [uploadView('clip.heic', 'image')], 'x');
            const block = bubble.children[0];
            for (const listener of block.querySelector('.chat-photo').listeners.get('error')) void listener({});
            // Retired while the HEAD is in flight: the feed releases and removes the whole message
            // (chat.js releaseMessageNode), so the block keeps its parent inside the detached bubble.
            if (end === 'release') {
                fx.controller.release(bubble);
                bubble.remove();
            } else fx.controller.destroy();
            const adds = fx.tracker.adds;
            answer();
            await new Promise((resolve) => setTimeout(resolve, 0));
            assert.equal(fx.tracker.adds, adds, `${end}: no listener comes back on the retired message`);
            assert.equal(block.querySelectorAll('.chat-photo').length, 1, `${end}: the retired subtree is left as it was`);
        }
    } finally {
        globalThis.fetch = priorFetch;
        fx.controller.destroy();
        fx.restore();
    }
});

test('the composer owns thumbnail URLs: once per image, revoked on remove/send/destroy, kept on failure', async () => {
    const fx = fixture();
    const prior = { create: URL.createObjectURL, revoke: URL.revokeObjectURL, fetch: globalThis.fetch };
    const created = [];
    const revoked = [];
    URL.createObjectURL = (file) => { created.push(file.name); return `blob:${file.name}`; };
    URL.revokeObjectURL = (url) => revoked.push(url);
    try {
        const parts = Object.fromEntries(['preview', 'attachBtn', 'fileInput', 'input'].map((key) => [key, new NodeStub('div', fx.tracker)]));
        const composer = createComposerAttachments({ ...parts, onLayout() {}, showToast() {} });
        const file = (name, type) => new File(['x'], name, { type });
        composer.stage([file('a.png', 'image/png'), file('b.pdf', 'application/pdf'), file('c.jpg', 'image/jpeg')]);
        assert.deepEqual(created, ['a.png', 'c.jpg'], 'one object URL per staged image, none for a PDF');
        assert.equal(parts.preview.querySelectorAll('.attach-thumb').length, 2);
        await parts.preview.querySelectorAll('[data-attachment-remove]')[2].click();
        assert.deepEqual(revoked, ['blob:c.jpg'], 'removing a staged image revokes its URL');
        let calls = 0;
        globalThis.fetch = async () => {
            calls += 1;
            if (calls === 2) return { ok: false, status: 500, json: async () => ({ ok: false, error: 'disk full' }) };
            return { ok: true, status: 200, json: async () => ({ ok: true, filename: `${'c'.repeat(32)}_a.png`,
                display_name: 'a.png', mime: 'image/png', view: uploadView('a.png', 'image') }) };
        };
        const readOnly = [];
        const upload = composer.upload(() => { readOnly.push(parts.input.readOnly); return true; });
        await assert.rejects(upload, (error) => error.message === 'disk full' && error.uploaded.length === 1);
        assert.deepEqual([...readOnly, parts.input.readOnly], [true, true, false], 'read-only, not disabled, while uploading');
        assert.deepEqual([composer.count, revoked], [2, ['blob:c.jpg']], 'a failed send keeps files and thumbnails');
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ ok: true, filename: 'f',
            display_name: 'x', mime: '', view: null }) });
        assert.equal((await composer.upload(() => true)).length, 2);
        composer.clear();
        assert.deepEqual(revoked, ['blob:c.jpg', 'blob:a.png'], 'a sent message releases its thumbnails');
        composer.stage([file('d.png', 'image/png')]);
        composer.destroy();
        assert.deepEqual(revoked.at(-1), 'blob:d.png', 'destroying the composer releases what is still staged');
    } finally {
        Object.assign(URL, { createObjectURL: prior.create, revokeObjectURL: prior.revoke });
        globalThis.fetch = prior.fetch;
        fx.controller.destroy();
        fx.restore();
    }
});

test('releasing a message, or the group buildGallery moved a file card into, closes the dialog that card opened', async () => {
    const fx = fixture();
    const doc = (name, task) => ({ type: 'document', role: 'assistant', task_id: task, filename: name, mime: 'application/pdf', file_base64: 'aGVsbG8=' });
    try {
        const bubble = Object.assign(new NodeStub('div', fx.tracker), { innerHTML: '<div class="message">x</div>' });
        fx.controller.mountAttachments(bubble, [uploadView('plan.pdf', 'file')], 'x');
        const [a, b] = ['a', 'b'].map((task) => ['one.pdf', 'two.pdf'].map((name) => {
            const built = fx.controller.buildDocumentBubble(doc(name, task));
            return fx.controller.buildGallery('files', doc(name, task), built) && built;
        })).map(([group, dropped]) => ({ group, dropped, item: group.querySelectorAll('.chat-file-item')[1] }));
        for (const [holder, other, owner] of [[bubble, new NodeStub('div', fx.tracker), bubble], [a.item, a.dropped, a.group], [b.item, b.dropped, b.item]]) {
            await holder.querySelector('.chat-file-card').click();
            const dialog = globalThis.document.body.querySelector('.chat-file-dialog'), created = fx.tracker.created.length;
            assert.equal(dialog.attributes.has('open'), true, 'the card opened the dialog');
            fx.controller.release(other);
            assert.equal(dialog.attributes.has('open'), true, 'releasing another message (or a dropped bubble) leaves it open');
            fx.controller.release(owner);
            assert.equal(dialog.attributes.has('open'), false, 'no actions remain on a released message\'s file');
            await dialog.querySelector('[data-file-action="download"]').click();
            assert.equal(fx.tracker.created.length, created, 'its pending Download acts on nothing');
        }
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});
