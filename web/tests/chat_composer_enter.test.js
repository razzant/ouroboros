// The Main and Project composer is a message field (docs/DESIGN.md "Controls and editable
// choices"): Enter presses Send, so the key takes Send's own path (pending upload, empty text);
// Shift+Enter is a line break, and a composing or held Enter sends nothing.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { WS } from '../modules/ws.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';

function composer({ chatId = 1, projectId = '', transport = null, reply = null } = {}) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    env.prior.WebSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1, CONNECTING: 0 };  // constants only; no native socket can open
    const frames = [];
    if (reply) sessionStorage.setItem(chatId === 1 ? 'ouro_chat_reply' : `ouro_chat_reply:${chatId}`, JSON.stringify(reply));
    let generation = 0;
    const instance = createChatInstance({
        ws: transport || { on: () => () => {}, isConnected: () => true,
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
    return { frames, input, byId, press, instance,
        draft(text) { input.value = text; for (const fn of input.listeners.get('input')) fn(); },
        close() { instance.destroy(); restoreDom(env.prior); } };
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

test('Enter follows Send: the existing chords send an ordinary message, a busy Send holds the draft', () => {
    const c = composer();
    try {
        assert.equal(c.byId('swarm'), undefined, 'no one-shot Swarm pill: the room\'s Subagents capsule holds the mode');
        c.input.value = 'plan the migration';
        c.press({ metaKey: true });
        assert.equal('force_plan' in c.frames[0], false, 'the frame carries no one-shot field');
        c.input.value = 'and report back';
        c.press({ ctrlKey: true });
        assert.equal(c.frames[1].content, 'and report back');
        c.byId('send').disabled = true;
        c.input.value = 'while uploading';
        assert.equal(c.press().defaultPrevented, true);
        assert.deepEqual([c.frames.length, c.input.value], [2, 'while uploading']);
    } finally { c.close(); }
});

test('typed Panic bypasses a busy Send but still respects composition and held Enter', () => {
    const c = composer();
    try {
        c.byId('send').disabled = true;
        c.input.value = '/panic';
        c.press({ isComposing: true });
        c.press({ repeat: true });
        assert.equal(c.frames.length, 0);
        c.press();
        assert.equal(c.frames.length, 1);
        assert.equal(c.frames[0].content, '/panic');
        assert.equal(c.input.value, '/panic', 'unconfirmed emergency keeps the draft');
    } finally { c.close(); }
});

test('production WS interception reaches the composer without a chat bubble or submission', async (t) => {
    const ws = new WS('ws://unused');
    let stopped = 0;
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1, CONNECTING: 0 };
    t.after(() => {
        if (priorSocket === undefined) delete globalThis.WebSocket;
        else globalThis.WebSocket = priorSocket;
    });
    globalThis.pywebview = { api: { panic_stop: async () => { stopped++; return { status: 'stopping' }; } } };
    ws.connect = () => assert.fail('Panic must not reconnect');
    const c = composer({ transport: ws });
    try {
        c.draft('/panic');
        c.press();
        assert.equal(stopped, 1);
        assert.equal(c.input.value, '/panic');
        await Promise.resolve();
        assert.equal(c.input.value, '', 'bridge acceptance consumes only the sent command');
        assert.equal(sessionStorage.getItem('ouro_chat_draft'), null);
        assert.equal(c.byId('messages').querySelectorAll('.chat-bubble.user').length, 0);
        assert.deepEqual(ws._pendingMessages, []);
    } finally {
        c.close();
        delete globalThis.pywebview;
    }
    assert.equal(Object.values(ws.listeners).reduce((n, listeners) => n + listeners.size, 0), 0);
});

for (const door of ['socket', 'bridge']) for (const newer of [false, true]) {
    test(`typed Panic ${door} acceptance preserves files/reply and ${newer ? 'newer text' : 'clears sent text'}`, async (t) => {
        const ws = new WS('ws://unused');
        const socket = new EventTarget();
        const sent = [];
        Object.assign(socket, { readyState: 1, bufferedAmount: 0, send: text => sent.push(JSON.parse(text)) });
        ws.ws = socket;
        ws.connect = () => assert.fail('no reconnect');
        let accept;
        if (door === 'bridge') {
            globalThis.pywebview = { api: { panic_stop: () => new Promise(resolve => { accept = () => resolve({ status: 'stopping' }); }) } };
            t.after(() => delete globalThis.pywebview);
        } else accept = () => {
            const event = new Event('message');
            event.data = JSON.stringify({ type: 'panic_accepted' });
            socket.dispatchEvent(event);
        };
        const reply = { address: 'row:1@a#1', role: 'assistant', snippet: 'Kept reply', kind: 'message' };
        const c = composer({ transport: ws, chatId: 7, projectId: 'p7', reply });
        try {
            const files = c.byId('file-input');
            files.files = [{ name: 'kept.txt', type: 'text/plain' }];
            for (const fn of files.listeners.get('change')) fn({ target: files });
            const keptReply = sessionStorage.getItem('ouro_chat_reply:7');
            assert.ok(keptReply);
            c.draft(' /PaNiC  ');
            c.press();
            assert.equal(c.input.value, ' /PaNiC  ', 'unknown keeps the exact draft');
            assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), ' /PaNiC  ');
            if (newer) c.draft('my next message');
            accept();
            await Promise.resolve();
            assert.equal(c.input.value, newer ? 'my next message' : '');
            assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), newer ? 'my next message' : null);
            assert.equal(sessionStorage.getItem('ouro_chat_reply:7'), keptReply);
            assert.equal(c.instance.hasPendingWork(), true, 'the staged file remains');
            assert.equal(c.byId('messages').querySelectorAll('.chat-bubble.user').length, 0);
            assert.equal(sent.length, door === 'socket' ? 1 : 0);
            assert.deepEqual(ws._pendingMessages, []);
        } finally { c.close(); }
    });
}

test('unconfirmed typed Panic retains its saved draft after socket close', async () => {
    const ws = new WS('ws://unused');
    const socket = new EventTarget();
    Object.assign(socket, { readyState: 1, bufferedAmount: 0, send: () => {} });
    ws.ws = socket;
    const c = composer({ transport: ws });
    document.body = new ElementStub('body', document);
    try {
        c.draft('/panic');
        c.press();
        assert.equal(c.input.value, '/panic');
        socket.dispatchEvent(new Event('close'));
        await Promise.resolve();
        assert.equal(c.input.value, '/panic');
        assert.equal(sessionStorage.getItem('ouro_chat_draft'), '/panic');
        assert.deepEqual(ws._pendingMessages, []);
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
        input.value = '/panic';
        press();
        assert.equal(frames.length, 1, 'Panic is delivered before the settings save resolves');
        assert.equal(frames.pop().content, '/panic');
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
