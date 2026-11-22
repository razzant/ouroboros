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

// ouro:panic reaches the composer through the page's global event bus (chat_activity.js
// raises it on its Panic send); the DOM fixture has none, so a test installs one. This is
// synthetic in-memory cancellation — a real Panic is never sent here.
function panicBus() {
    const keys = ['addEventListener', 'removeEventListener', 'dispatchEvent'];
    const prior = keys.map(key => Object.getOwnPropertyDescriptor(globalThis, key));
    const bus = new EventTarget();
    globalThis.addEventListener = bus.addEventListener.bind(bus);
    globalThis.removeEventListener = bus.removeEventListener.bind(bus);
    globalThis.dispatchEvent = bus.dispatchEvent.bind(bus);
    return () => keys.forEach((key, i) => {
        if (prior[i]) Object.defineProperty(globalThis, key, prior[i]);
        else delete globalThis[key];
    });
}

// The deferred-save fixture: the room's effort-range save held on a deferred promise, and a ws
// stub that keeps its handlers, so a send is observed across the settings await and a host row
// can answer it mid-flight.
function deferredSaveRoom({ savedReply = null, uploads = null, status = 'sent', kept = null } = {}) {
    const pending = [];
    const deleted = [];
    const env = installDom(async (url, options) => {
        if (String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        if (String(url) === '/api/owner/effort-range') return new Promise((resolve) => pending.push(resolve));
        if (uploads && String(url).startsWith('/api/chat/upload')) {
            if (options?.method === 'DELETE') { deleted.push(JSON.parse(options.body).filename); return { ok: true, json: async () => ({}) }; }
            const data = await uploads();
            return { ok: data.ok !== false, json: async () => data };
        }
        return { ok: true, json: async () => ({ active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) };
    });
    globalThis.sessionStorage.setItem('ouro_chat_session_id', 'session-me');
    if (savedReply) globalThis.sessionStorage.setItem('ouro_chat_reply', JSON.stringify(savedReply));
    if (kept) globalThis.sessionStorage.setItem('ouro_chat_unconfirmed', kept);  // a reload: this tab's kept copies
    const handlers = new Map();
    const frames = [];
    let generation = 0;
    const instance = createChatInstance({
        ws: {
            ws: { readyState: 1 },
            on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
            isConnected: () => true,
            send: (frame) => { frames.push(frame); return { status, clientMessageId: `cm-${frames.length}` }; },
        },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const input = document.byId.get('chat-input');
    const press = () => {
        const event = { key: 'Enter', target: input, preventDefault() { event.defaultPrevented = true; } };
        for (const listener of input.listeners.get('keydown')) listener(event);
        return event;
    };
    const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };
    instance.hydrateStateSnapshot({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
    const rec = env.mount.children[0].children.find((node) => node.classList.contains('chat-toolbar-row'))
        .children.find((node) => node.classList.contains('chat-effort-range'))
        .querySelector('.chat-effort-rec');
    const pendingNotes = (node) => node?.querySelectorAll('.msg-pending') || [];
    return {
        frames, input, press, settle, deleted, instance,
        fire(type, payload) { for (const fn of handlers.get(type) || []) fn(payload); },
        kept: () => JSON.parse(globalThis.sessionStorage.getItem('ouro_chat_unconfirmed') || '[]'),
        bubble: (id) => document.byId.get('chat-messages').children
            .find((node) => node.classList.contains('chat-bubble') && node.dataset.clientMessageId === id),
        notes: (node) => pendingNotes(node).map((note) => note.textContent),
        actions: (node) => pendingNotes(node).flatMap((note) => note.children.filter((child) => child.tagName === 'BUTTON'))
            .map((button) => button.textContent),
        type(text) { input.value = text; for (const fn of input.listeners.get('input')) fn(); },
        set(text) { input.value = text; },  // no input event: the draft is untouched, its version stands
        stageFiles(list) { const files = document.byId.get('chat-file-input');
            files.files = list.map(([name, type]) => ({ name, type }));
            for (const fn of files.listeners.get('change')) fn({ target: files }); },
        removeBadge(index) { const preview = document.byId.get('chat-attachment-preview');
            const button = preview.querySelectorAll('[data-attachment-remove]')[index];
            for (const fn of button.listeners.get('click')) fn({ target: button }); },
        beginSave() { for (const listener of rec.listeners.get('keydown'))
            listener({ key: 'ArrowRight', currentTarget: rec, preventDefault() {} }); },
        resolveSave(ok = true) { pending.shift()({ ok, json: async () => (ok
            ? { ok: true, effort_range: { min: 'low', recommended: 'high', max: 'high' } }
            : { error: 'Effort range refused.', saved: false, code: 'effort_range_invalid' }) }); },
        echo(n, extra = {}) { for (const fn of handlers.get('chat') || []) fn({ type: 'chat', role: 'user', chat_id: 1,
            ts: '2026-10-16T09:40:00Z', source: 'web', sender_session_id: 'session-me', content: frames[n - 1].content,
            client_message_id: `cm-${n}`, ingress_accepted: true, ...extra }); },
        firePanic: () => globalThis.dispatchEvent(new CustomEvent('ouro:panic')),
        reopen() { for (const fn of handlers.get('open') || []) fn({ previouslyConnected: true }); },
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('an accepted Panic during the settings save stops the pending send before it sends', async () => {
    const restoreBus = panicBus();
    const f = deferredSaveRoom();
    try {
        f.beginSave();
        f.type('check the schedule');
        f.press();
        assert.equal(f.frames.length, 0, 'Send waits for the save');
        f.firePanic();
        f.resolveSave(true);
        await f.settle();
        assert.equal(f.frames.length, 0, 'no chat frame follows an accepted Panic');
        assert.equal(f.input.value, 'check the schedule', 'the newer draft stays untouched');
    } finally { f.close(); restoreBus(); }
});

test('an accepted Panic during the upload stops the pending send and cleans the completed upload', async () => {
    const restoreBus = panicBus();
    let release;
    const stored = `${'0'.repeat(32)}_note.txt`;
    const f = deferredSaveRoom({ uploads: () => new Promise((resolve) => { release = resolve; }).then(() => ({
        ok: true, filename: stored, display_name: 'note.txt', mime: 'text/plain',
        view: { name: 'note.txt', kind: 'file', available: true, size: 3, url: `/api/files/download?upload=${stored}` } })) });
    try {
        f.stageFiles([['note.txt', 'text/plain']]);
        f.type('see the file');
        f.press();
        assert.equal(f.frames.length, 0, 'the upload is in flight');
        f.firePanic();
        release();
        await f.settle();
        assert.equal(f.frames.length, 0, 'no chat frame follows an accepted Panic');
        assert.deepEqual(f.deleted, [stored], 'the completed upload is cleaned up, not orphaned');
    } finally { f.close(); restoreBus(); }
});

test('an accepted Panic during a batch upload starts no later file and cleans the completed one', async () => {
    const restoreBus = panicBus();
    const releases = [], stored = [];
    const f = deferredSaveRoom({ uploads: () => new Promise((resolve) => releases.push(resolve)).then(() => {
        const name = `${String(stored.length + 1).padStart(32, '0')}_${stored.length + 1}.txt`;
        stored.push(name);
        return { ok: true, filename: name, display_name: `${stored.length}.txt`, mime: 'text/plain',
            view: { name: `${stored.length}.txt`, kind: 'file', available: true, size: 3, url: `/api/files/download?upload=${name}` } };
    }) });
    try {
        document.body = new ElementStub('body', document);  // the toast stack's host
        f.stageFiles([['1.txt', 'text/plain'], ['2.txt', 'text/plain'], ['3.txt', 'text/plain']]);
        f.type('see the files');
        f.press();
        await f.settle();
        assert.equal(releases.length, 1, 'the first file is uploading');
        f.firePanic();
        releases[0]();
        await f.settle();
        assert.equal(releases.length, 1, 'no later file starts after the Panic attempt');
        assert.equal(f.frames.length, 0, 'no chat frame follows it');
        assert.deepEqual(f.deleted, [stored[0]], 'the completed upload is cleaned up, not orphaned');
        assert.equal(document.body.children.length, 0, 'a Panic is not reported as an upload error');
        assert.equal(f.input.value, 'see the files', 'the draft stays for a new submission');
    } finally { f.close(); restoreBus(); }
});

test('a message submitted after a Panic is sent; the send the Panic crossed stays abandoned', async () => {
    const restoreBus = panicBus();
    const f = deferredSaveRoom();
    try {
        f.beginSave();
        f.type('check the schedule');
        f.press();
        f.firePanic();
        f.resolveSave(true);
        await f.settle();
        assert.equal(f.frames.length, 0, 'the crossed send is abandoned');
        f.type('a new message');
        f.press();
        await f.settle();
        assert.deepEqual(f.frames.map((frame) => frame.content), ['a new message'], 'Send works again for a new submission');
        assert.equal(f.input.value, '');
        f.reopen();  // a reconnect is no resume: nothing the Panic crossed leaves afterwards
        await f.settle();
        assert.equal(f.frames.length, 1);
    } finally { f.close(); restoreBus(); }
});

test('a late edit during the settings save still sends the submitted words and keeps the newer draft', async () => {
    const f = deferredSaveRoom();
    try {
        f.beginSave();
        f.type('plan the migration');
        f.press();
        f.type('plan the migration and the rollback');
        f.resolveSave(true);
        await f.settle();
        assert.equal(f.frames.length, 1, 'the submitted message finishes');
        assert.equal(f.frames[0].content, 'plan the migration');
        assert.equal(f.input.value, 'plan the migration and the rollback', 'the newer draft is not consumed');
    } finally { f.close(); }
});

test('a sent selected reply frees the field: the next message starts empty and answers nothing', async () => {
    const reply = { address: 'row:1@a#1', role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const f = deferredSaveRoom({ savedReply: reply });
    try {
        f.type('First note.');
        f.press();
        await f.settle();
        assert.equal(f.frames.length, 1);
        assert.equal(f.frames[0].reply_to, reply.address);
        assert.deepEqual(f.frames[0].reply_excerpt, reply.excerpt);
        assert.equal(f.input.value, '', 'the socket took the frame: the field is free');
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_reply'), null, 'the quote left with its message');
        assert.equal(f.kept().at(-1).frame.client_message_id, 'cm-1', 'the words and quote wait in their bubble');
        assert.deepEqual(f.kept().at(-1).frame.reply_excerpt, reply.excerpt);
        f.press();
        await f.settle();
        assert.equal(f.frames.length, 1, 'an empty field sends nothing, the unconfirmed message included');
        f.echo(1, { ingress_dispatched: true, reply_to: { address: reply.address, excerpt: reply.excerpt } });
        assert.deepEqual(f.kept(), [], 'the saved row ends the doubt');
        f.type('Second note.');
        f.press();
        await f.settle();
        assert.equal(f.frames.length, 2);
        assert.equal(f.frames[1].reply_to, undefined, 'the next message answers nothing');
        assert.equal(f.frames[1].reply_excerpt, undefined);
    } finally { f.close(); }
});

test('a selected reply queued offline frees the field, says so on its own bubble and is queued once', async () => {
    const reply = { address: 'row:1@a#1', role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const f = deferredSaveRoom({ savedReply: reply, status: 'queued' });
    try {
        document.body = new ElementStub('body', document);  // the toast stack's host
        f.type('First note.');
        f.press();
        await f.settle();
        assert.equal(f.frames.length, 1, 'the Send went to the socket queue');
        assert.equal(f.input.value, '', 'the queue took the frame: the field is free');
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_reply'), null);
        const bubble = f.bubble('cm-1');
        assert.ok(bubble.classList.contains('pending'));
        assert.equal(f.notes(bubble).length, 1, 'the bubble itself says where the message is');
        assert.match(bubble.innerHTML, /<div class="msg-pending">Queued until reconnect<\/div>/);  // the stub keeps no text nodes
        assert.ok(bubble.children.some((child) => child.dataset.excerpt === '1'), 'with its quote');
        f.press();
        await f.settle();
        assert.equal(f.frames.length, 1, 'nothing is queued twice');
        f.fire('close');
        assert.deepEqual(f.actions(bubble), [], 'the queue will write it: nothing is offered twice');
        assert.deepEqual(document.body.children.flatMap((stack) => stack.children.map((toast) => toast.textContent)), [],
            'no toast stands in for the bubble');
        assert.deepEqual(f.kept().at(-1).frame.reply_excerpt, reply.excerpt, 'a reload still finds it kept');
        f.fire('outbound_sent', { clientMessageId: 'cm-1', queued: true });
        assert.equal(bubble.classList.contains('pending'), false, 'written: the queued note leaves');
        assert.deepEqual(f.notes(bubble), []);
        f.fire('close');
        assert.deepEqual(f.actions(bubble), ['Send again', 'Send as new', 'Discard'], 'written but not saved: now in doubt');
        assert.equal(f.frames.length, 1);
    } finally { f.close(); }
});

test('a late saved row of a sent selected reply leaves the newer draft alone, with or without its reply', async () => {
    const reply = { address: 'row:1@a#1', role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const f = deferredSaveRoom({ savedReply: reply });
    try {
        f.type('First note.');
        f.press();
        await f.settle();
        assert.equal(f.input.value, '');
        f.type('A newer thought.');
        // An older core's saved row without the reply (dispatch still owed), then the dispatched row with it.
        f.echo(1, { ingress_pending: true });
        assert.equal(f.input.value, 'A newer thought.');
        assert.equal(f.kept().length, 1, 'accepted, its dispatch still owed: the frame waits');
        f.echo(1, { ingress_dispatched: true, reply_to: { address: reply.address, excerpt: reply.excerpt } });
        assert.equal(f.input.value, 'A newer thought.', 'the newer draft is not consumed');
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_reply'), null, 'the sent quote does not come back');
        assert.deepEqual(f.kept(), []);
        assert.equal(f.frames.length, 1, 'nothing is sent again');
    } finally { f.close(); }
});

test('after a reload, Send again resends the same id, quote and files, and Send as new returns words and quote', async () => {
    const reply = { address: 'row:1@a#1', role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const stored = `${'0'.repeat(31)}1_plan.txt`;
    const first = deferredSaveRoom({ savedReply: reply, uploads: () => ({ ok: true, filename: stored, display_name: 'plan.txt',
        mime: 'text/plain', view: { name: 'plan.txt', kind: 'file', available: true, size: 3, url: `/api/files/download?upload=${stored}` } }) });
    let kept, sent;
    try {
        first.stageFiles([['plan.txt', 'text/plain']]);
        first.type('With the plan.');
        first.press();
        await first.settle();
        [sent] = first.frames;
        assert.equal(sent.attachments[0].filename, stored);
        assert.equal(first.input.value, '');
        kept = globalThis.sessionStorage.getItem('ouro_chat_unconfirmed');
    } finally { first.close(); }
    const f = deferredSaveRoom({ kept });  // the reload: the tab's kept copy, and a first history read without it saved
    try {
        for (let i = 0; i < 200 && f.actions(f.bubble('cm-1')).length === 0; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
        const bubble = f.bubble('cm-1');
        assert.ok(bubble, 'the kept message is shown again from its own words');
        assert.deepEqual(f.actions(bubble), ['Send again', 'Send as new', 'Discard']);
        bubble.querySelector('.message').textContent = 'With the plan.';  // the stub DOM keeps no bubble words
        bubble.querySelector('[data-unconfirmed-action="send_as_new"]').click();
        assert.equal(f.frames.length, 0, 'preparing sends nothing');
        assert.equal(f.input.value, 'With the plan.');
        assert.deepEqual(JSON.parse(globalThis.sessionStorage.getItem('ouro_chat_reply')).excerpt, reply.excerpt,
            'Send as new keeps the selected passage');
        bubble.querySelector('[data-unconfirmed-action="retry"]').click();
        const again = f.frames.at(-1);
        assert.equal(again.client_message_id, 'cm-1', 'the SAME id: the host rejoins a row it saved after all');
        assert.equal(again.content, sent.content);
        assert.equal(again.reply_to, reply.address);
        assert.deepEqual(again.reply_excerpt, reply.excerpt);
        assert.deepEqual(again.attachments, sent.attachments, 'and the same files');
        assert.equal(f.input.value, 'With the plan.', 'Send again leaves the prepared draft alone');
        f.press();
        await f.settle();
        const fresh = f.frames.at(-1);
        assert.notEqual(fresh, again);
        assert.equal(fresh.client_message_id, undefined, 'a new message: the socket mints its id');
        assert.deepEqual(fresh.reply_excerpt, reply.excerpt);
        assert.equal(fresh.attachments, undefined, 'the files stayed with the original message');
    } finally { f.close(); }
});

test('a badge kept in the newer draft keeps its own upload ref, not the removed badge\'s', async () => {
    const reply = { address: 'row:1@a#1', role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const names = [];
    const f = deferredSaveRoom({ savedReply: reply, uploads: () => {
        const stored = `${String(names.length + 1).padStart(32, '0')}_${names.length + 1}.txt`;
        names.push(stored);
        return { ok: true, filename: stored, display_name: `${names.length}.txt`, mime: 'text/plain',
            view: { name: `${names.length}.txt`, kind: 'file', available: true, size: 3, url: `/api/files/download?upload=${stored}` } };
    } });
    try {
        f.stageFiles([['1.txt', 'text/plain'], ['2.txt', 'text/plain']]);
        f.type('with both files');
        f.beginSave();
        f.press();
        // The owner drops the first badge and edits the words while the send waits on the room's
        // save; the submitted message still goes out with the files it snapshotted, and the kept
        // badge must take its own ref — never the removed one's by position.
        f.removeBadge(0);
        f.type('a newer draft');
        f.resolveSave(true);
        await f.settle();
        assert.equal(f.frames.length, 1);
        assert.equal(f.frames[0].attachments.length, 2, 'Send fixes the files before waiting');
        assert.equal(f.input.value, 'a newer draft', 'the newer draft is not consumed');
        const keptVersion = JSON.parse(globalThis.sessionStorage.getItem('ouro_chat_draft_version') || 'null');
        assert.deepEqual(keptVersion?.uploads?.map((item) => item.filename), [names[1]],
            'the kept badge keeps its own completed upload and so survives a reload');
    } finally { f.close(); }
});
