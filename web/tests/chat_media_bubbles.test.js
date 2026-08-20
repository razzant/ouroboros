// Behavioural characterization of the media-bubble owner, exercised where the
// code now lives: photo and video WS frames become bubbles with validated
// mime types, sanitized base64 payloads, optional captions and the unread
// bump — and a foreign-thread frame renders nothing at all.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createMediaBubbles } from '../modules/chat_media_bubbles.js';

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
        dataset: {},
        listeners: {},
        stubbedNodes: {},
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        querySelector(selector) { return el.stubbedNodes[selector] ?? null; },
    };
    return el;
}

function mediaBubbles({ mine = true } = {}) {
    const priorDocument = globalThis.document;
    const created = [];
    globalThis.document = {
        createElement: (tag) => {
            const el = makeNode(tag);
            created.push(el);
            return el;
        },
    };
    const calls = [];
    const spy = (name, result) => (...args) => { calls.push({ name, args }); return result; };
    const api = createMediaBubbles({
        isMyThread: () => mine,
        hideTypingIndicatorOnly: spy('hideTypingIndicatorOnly'),
        syncChatStatus: spy('syncChatStatus'),
        getSenderLabel: () => 'Owner',
        formatMsgTime: () => ({ full: 'full time', short: 'short' }),
        stampNodeTimestamp: spy('stampNodeTimestamp'),
        insertMessageNode: spy('insertMessageNode'),
        incrementUnreadIfNeeded: spy('incrementUnreadIfNeeded'),
    });
    return {
        ...api,
        created,
        // escapeHtml creates scratch elements too; bubbles are the ones styled.
        bubbles: () => created.filter((el) => el.className.startsWith('chat-bubble')),
        calls,
        named: (name) => calls.filter((call) => call.name === name),
        restore() { globalThis.document = priorDocument; },
    };
}

test('a photo frame renders a data-URL image bubble and bumps unread', () => {
    const m = mediaBubbles();
    m.handlePhotoFrame({
        role: 'assistant', mime: 'image/jpeg', image_base64: 'QUJD\nREVG',
        caption: 'a <caption>', ts: '2026-08-18T00:00:00Z',
    });
    const bubble = m.bubbles()[0];
    assert.equal(bubble.className, 'chat-bubble assistant');
    assert.match(bubble.innerHTML, /data:image\/jpeg;base64,QUJDREVG/, 'whitespace is stripped from the payload');
    assert.match(bubble.innerHTML, /a &lt;caption&gt;/, 'the caption is escaped');
    assert.equal(m.named('insertMessageNode').length, 1);
    assert.equal(m.named('incrementUnreadIfNeeded').length, 1);
    assert.equal(m.named('hideTypingIndicatorOnly').length, 1);
    m.restore();
});

test('an invalid mime or corrupt payload degrades safely', () => {
    const m = mediaBubbles();
    m.handlePhotoFrame({ role: 'assistant', mime: 'image/svg+xml;evil', image_base64: 'not*base64!' });
    const bubble = m.bubbles()[0];
    assert.doesNotMatch(bubble.innerHTML, /evil/, 'a malformed mime falls back to the default');
    assert.match(bubble.innerHTML, /src=""/, 'a corrupt payload renders no data URL at all');
    m.restore();
});

test('a video frame validates its own mime family', () => {
    const m = mediaBubbles();
    m.handleVideoFrame({ role: 'user', mime: 'video/webm', video_base64: 'QUJD' });
    const bubble = m.bubbles()[0];
    assert.equal(bubble.className, 'chat-bubble user');
    assert.match(bubble.innerHTML, /data:video\/webm;base64,QUJD/);
    assert.match(bubble.innerHTML, /<video class="chat-video"/);
    m.handleVideoFrame({ role: 'assistant', mime: 'image/png', video_base64: 'QUJD' });
    assert.match(m.bubbles()[1].innerHTML, /data:video\/mp4;base64/, 'a non-video mime falls back to mp4');
    m.restore();
});

test('a foreign-thread frame renders nothing', () => {
    const m = mediaBubbles({ mine: false });
    m.handlePhotoFrame({ role: 'assistant', image_base64: 'QUJD' });
    m.handleVideoFrame({ role: 'assistant', video_base64: 'QUJD' });
    assert.equal(m.bubbles().length, 0);
    assert.equal(m.calls.length, 0);
    m.restore();
});
