// The Main and Project composer is a message field (docs/DESIGN.md "Controls and editable
// choices"): Enter presses Send, so the key takes Send's own path (Swarm, pending upload,
// empty text); Shift+Enter is a line break, and a composing or held Enter sends nothing.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';

function composer({ chatId = 1, projectId = '' } = {}) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    const frames = [];
    let generation = 0;
    const instance = createChatInstance({
        ws: { on: () => () => {}, isConnected: () => true,
            send: (frame) => { frames.push(frame); return { status: 'sent', clientMessageId: `cm-${frames.length}` }; } },
        state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, projectId, idPrefix: 'chat', mountEl: env.mount, asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const byId = (suffix) => document.byId.get(`chat-${suffix}`);
    const input = byId('input');
    const press = (init = {}) => {
        const event = { key: 'Enter', target: input, preventDefault() { event.defaultPrevented = true; }, ...init };
        for (const listener of input.listeners.get('keydown')) listener(event);
        return event;
    };
    return { frames, input, byId, press, close() { instance.destroy(); restoreDom(env.prior); } };
}

for (const [room, options] of [['Main', {}], ['a Project', { chatId: 7, projectId: 'p7' }]]) {
    test(`${room} composer: Enter sends through Send, Shift+Enter and composition keep the draft`, () => {
        const c = composer(options);
        try {
            assert.equal(c.input.enterKeyHint, 'send');
            c.input.value = 'first line\nsecond line';
            assert.equal(c.press({ shiftKey: true }).defaultPrevented, undefined, 'Shift+Enter is the line break');
            assert.equal(c.press({ isComposing: true }).defaultPrevented, undefined);
            assert.equal(c.press({ keyCode: 229 }).defaultPrevented, undefined, 'WebKit commits IME this way');
            assert.deepEqual([c.frames.length, c.input.value], [0, 'first line\nsecond line']);
            assert.equal(c.press().defaultPrevented, true);
            assert.equal(c.frames.length, 1);
            assert.equal(c.frames[0].content, 'first line\nsecond line');
            assert.equal(c.frames[0].project_id, options.projectId, 'the room the button would send to');
            assert.equal(c.input.value, '');
            assert.equal(c.press({ repeat: true }).defaultPrevented, true, 'a held key adds no line breaks');
            c.press();
            assert.equal(c.frames.length, 1, 'an empty composer sends nothing');
        } finally { c.close(); }
    });
}

test('Enter follows Send: Swarm arms the next message, the existing chords send, a busy Send holds the draft', () => {
    const c = composer();
    try {
        c.byId('swarm').click();
        c.input.value = 'plan the migration';
        c.press({ metaKey: true });
        assert.equal(c.frames[0].force_plan, true);
        assert.equal(c.byId('swarm').dataset.armed, 'false', 'one-shot, exactly as a click on Send');
        c.input.value = 'and report back';
        c.press({ ctrlKey: true });
        assert.equal(c.frames[1].force_plan, false);
        c.byId('send').disabled = true;
        c.input.value = 'while uploading';
        assert.equal(c.press().defaultPrevented, true);
        assert.deepEqual([c.frames.length, c.input.value], [2, 'while uploading']);
    } finally { c.close(); }
});

// An effort change made just before Send is the setting the next root would start with, so
// Send and Enter wait for that save; a refused save keeps the draft in the field.
test('Send waits for a pending effort-range save; a refused save keeps the draft', async () => {
    const pending = [];
    const env = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        if (String(url) === '/api/owner/effort-range') return new Promise((resolve) => pending.push(resolve));
        return { ok: true, json: async () => ({ active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) };
    });
    const frames = [];
    let generation = 0;
    const instance = createChatInstance({
        ws: { on: () => () => {}, isConnected: () => true,
            send: (frame) => { frames.push(frame); return { status: 'sent', clientMessageId: `cm-${frames.length}` }; } },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    try {
        const page = env.mount.children[0];
        const row = page.children.find((node) => node.classList.contains('chat-toolbar-row'));
        const control = row.children.find((node) => node.classList.contains('chat-effort-range'));
        assert.ok(control, 'the effort range control mounts in the toolbar row');
        const rec = control.querySelector('.chat-effort-rec');
        const input = document.byId.get('chat-input');
        const send = document.byId.get('chat-send');
        const press = () => {
            const event = { key: 'Enter', target: input, preventDefault() { event.defaultPrevented = true; } };
            for (const listener of input.listeners.get('keydown')) listener(event);
            return event;
        };
        instance.hydrateStateSnapshot({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
        for (const listener of rec.listeners.get('keydown')) listener({ key: 'ArrowRight', currentTarget: rec, preventDefault() {} });
        assert.equal(pending.length, 1, 'the save left at the gesture end');
        input.value = 'use the new level';
        press();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(frames.length, 0, 'Send waits for the save');
        assert.equal(send.textContent, 'Saving');
        assert.equal(send.disabled, true, 'a second Enter holds the draft meanwhile');
        pending[0]({ ok: true, json: async () => ({ ok: true, effort_range: { min: 'low', recommended: 'high', max: 'high' } }) });
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(frames.length, 1, 'then sends');
        assert.equal(frames[0].content, 'use the new level');
        assert.equal(send.disabled, false);
        assert.equal(control.dataset.saving, 'false');
        // A refusal: the sentence goes to the toast, the field keeps the draft.
        document.body = new ElementStub('body', document);  // the toast stack's host
        for (const listener of rec.listeners.get('keydown')) listener({ key: 'ArrowLeft', currentTarget: rec, preventDefault() {} });
        input.value = 'still here';
        press();
        await new Promise((resolve) => setTimeout(resolve, 0));
        pending[1]({ ok: false, status: 400, json: async () => ({ error: 'Effort range refused.', saved: false, code: 'effort_range_invalid' }) });
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(frames.length, 1, 'a refused save sends nothing');
        assert.equal(input.value, 'still here');
        assert.equal(send.disabled, false);
        assert.equal(rec.getAttribute('aria-valuetext'), 'High', 'the control shows the server value again');
    } finally { instance.destroy(); restoreDom(env.prior); }
});
