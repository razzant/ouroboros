import assert from 'node:assert/strict';
import test from 'node:test';
import { createRoomHeader, panicButtonHtml } from '../modules/chat_header.js';
import { createChatInstance } from '../modules/chat.js';
import { finishDesktopStartup, openDesktopRoomSession } from '../modules/desktop_room_session.js';
import { WS } from '../modules/ws.js';
import { panicWithoutSocket } from '../modules/panic.js';
import { installDomDouble } from './dom_double.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function liveSocket() {
    const socket = new EventTarget();
    Object.assign(socket, { readyState: 1, bufferedAmount: 0, sent: [] });
    socket.send = text => {
        socket.sent.push(JSON.parse(text));
        const ack = new Event('message');
        ack.data = JSON.stringify({ type: 'panic_accepted' });
        socket.dispatchEvent(ack);
    };
    return socket;
}

test('desktop button and typed Panic stop at the launcher even with an OPEN hung socket', async (t) => {
    let stopped = 0;
    globalThis.pywebview = { api: { panic_stop: async () => { stopped++; return { status: 'stopping' }; } } };
    globalThis.fetch = () => assert.fail('desktop Panic cannot depend on HTTP');
    t.after(() => { delete globalThis.pywebview; delete globalThis.fetch; });
    for (const frame of [{ type: 'command', cmd: '/panic' }, { type: 'chat', content: ' /PANIC' }]) {
        const ws = new WS('ws://unused');
        ws.ws = { readyState: 1, send: () => assert.fail('desktop stop needs no server') };
        ws.connect = () => assert.fail('Panic cannot reconnect');
        assert.deepEqual(ws.send(frame), { status: 'intercepted', clientMessageId: '' });
        assert.equal(ws._pendingMessages.length, 0);
    }
    await Promise.resolve();
    assert.equal(stopped, 2);
});

test('browser Panic uses its captured live transport regardless of clock skew', async (t) => {
    const original = Date.now;
    t.after(() => { Date.now = original; });
    for (const clock of [0, 999999999999999]) {
        Date.now = () => clock;
        const socket = liveSocket();
        await panicWithoutSocket({ input: 'typed_chat' }, socket);
        assert.deepEqual(socket.sent, [{ type: 'command', cmd: '/panic', client_surface: { input: 'typed_chat' } }]);
    }
});

test('offline or backlogged browser Panic reports failure and never waits for a successor', async (t) => {
    const env = installDomDouble();
    t.after(env.restore);
    globalThis.fetch = () => assert.fail('HTTP could connect to a successor');
    t.after(() => delete globalThis.fetch);
    for (const socket of [null, { readyState: 0 }, { readyState: 1, bufferedAmount: 1 }]) {
        await panicWithoutSocket({}, socket, () => assert.fail('unavailable transport accepted Panic'));
    }
    assert.equal(env.doc.body.querySelectorAll('.toast').length, 3);
    assert.match(env.doc.body.textContent, /Panic was not confirmed/);
    assert.equal(env.doc.body.querySelectorAll('.toast-dismiss').length, 3, 'unconfirmed Panic stays visible but can be closed');
    assert.doesNotMatch(env.doc.body.textContent, /tray|menu bar/);
});

test('legacy queued Panic is discarded at reconnect while ordinary chat survives', () => {
    const installed = Object.getOwnPropertyDescriptor(globalThis, 'WebSocket');
    globalThis.WebSocket = { OPEN: 1 };
    try {
        const ws = new WS('ws://unused');
        const sent = [];
        ws.ws = { readyState: 1, send: text => sent.push(JSON.parse(text)) };
        ws._pendingMessages = [{ type: 'chat', content: '/panic' }, { type: 'chat', content: 'hello' }];
        ws._flushPendingMessages();
        assert.deepEqual(sent, [{ type: 'chat', content: 'hello' }]);
    } finally {
        if (installed) Object.defineProperty(globalThis, 'WebSocket', installed);
        else delete globalThis.WebSocket;
    }
});

test('failed desktop bridge falls through only to the captured live socket', async (t) => {
    globalThis.pywebview = { api: { panic_stop: async () => { throw Error('bridge unavailable'); } } };
    t.after(() => delete globalThis.pywebview);
    const socket = liveSocket();
    await panicWithoutSocket({}, socket);
    assert.equal(socket.sent.length, 1);
});

test('acceptance, close and send failure dispose the captured socket listeners and timer', async (t) => {
    const env = installDomDouble();
    t.after(env.restore);
    for (const outcome of ['accepted', 'closed', 'throw']) {
        const socket = liveSocket();
        const listeners = new Map();
        const add = socket.addEventListener.bind(socket), remove = socket.removeEventListener.bind(socket);
        socket.addEventListener = (type, fn) => { listeners.set(type, fn); add(type, fn); };
        socket.removeEventListener = (type, fn) => { listeners.delete(type); remove(type, fn); };
        if (outcome !== 'accepted') socket.send = () => {
            if (outcome === 'throw') throw Error('fixture refusal');
            socket.dispatchEvent(new Event('close'));
        };
        let accepted = 0;
        await panicWithoutSocket({}, socket, () => { accepted++; });
        assert.equal(accepted, outcome === 'accepted' ? 1 : 0, outcome);
        assert.equal(listeners.size, 0, outcome);
    }
    assert.equal(env.doc.body.querySelectorAll('.toast').length, 2, 'accepted controls show no failure');
});

test('a resolved bridge refusal is not acceptance and does not send again', async (t) => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    for (const response of [undefined, false, { status: 'refused' }]) {
        globalThis.pywebview = { api: { panic_stop: async () => response } };
        let accepted = 0;
        await panicWithoutSocket({}, null, () => { accepted++; });
        assert.equal(accepted, 0);
        const socket = liveSocket();
        await panicWithoutSocket({}, socket, () => { accepted++; });
        assert.equal(accepted, 0);
        assert.equal(socket.sent.length, 0);
    }
});

test('the room button calls the fake Stop sink once on click, before its desktop open reply', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    const calls = [], roomPosts = [];
    const ticket = 'pressing-document';
    let finishOpen;
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: { ouroborosRoomSession: {
        postMessage(message) {
            roomPosts.push(message.op);
            return new Promise(done => { finishOpen = done; });
        },
    } } } });
    host.top = host;
    const doc = { documentElement: { dataset: {} }, querySelector: () => ({ content: ticket, remove() {} }) };
    const opening = openDesktopRoomSession(host, doc, assert.fail);
    const ws = new WS('unused');
    ws.ws = liveSocket();
    ws.connect = () => assert.fail('Panic must not reconnect');
    env.doc.parseHtml = true;
    const bar = env.el('header');
    env.doc.body.append(bar);
    bar.innerHTML = panicButtonHtml();
    const header = createRoomHeader({ bar, ws });
    t.after(() => header.destroy());
    const button = bar.querySelector('[data-chat-command="panic"]');
    let acceptStop;
    const pendingStop = new Promise(done => { acceptStop = done; });
    globalThis.pywebview = { api: { panic_stop(documentTicket) { calls.push(documentTicket); return pendingStop; } } };
    t.after(() => delete globalThis.pywebview);
    button.click();
    assert.deepEqual(calls, [ticket]);
    assert.deepEqual(roomPosts, ['open']);
    finishOpen({});
    await opening;
    acceptStop({ status: 'stopping' });
    await Promise.resolve();
    assert.deepEqual(calls, [ticket]);
    assert.deepEqual(ws.ws.sent, []);
    assert.equal(ws._pendingMessages.length, 0);

    const olderCalls = [];
    globalThis.pywebview.api.panic_stop = (...args) => { olderCalls.push(args); return { status: 'stopping' }; };
    button.click();
    await Promise.resolve();
    assert.deepEqual(olderCalls, [[]], 'an older zero-argument launcher still receives one call per press');
    assert.deepEqual(ws.ws.sent, []);
});

test('a stamped desktop button during open and typed Panic before socket use only the bound native channel', async t => {
    const env = installDom(), native = [], warnings = [];
    let room;
    t.after(() => { room?.destroy(); restoreDom(env.prior); });
    const ws = new WS('unused');
    ws.connect = () => assert.fail('Panic cannot reconnect');
    let resolveOpen, rejectNext = false;
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: {
        ouroborosRoomSession: { postMessage(message) {
            if (message.op === 'open') return new Promise(resolve => { resolveOpen = resolve; });
            return true;
        } },
        ouroborosDesktopPanic: { postMessage(ticket) {
            native.push(ticket);
            if (rejectNext) return Promise.reject(Error('reply missed'));
            return { status: 'stopping' };
        } },
    } } });
    host.top = host;
    const doc = { documentElement: { dataset: {} }, querySelector: () => ({ content: 'bound-document', remove() {} }) };
    openDesktopRoomSession(host, doc, text => warnings.push(text));
    room = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set() }, mountEl: env.mount,
        stateSnapshots: { begin: () => ({}), gate: async () => null, isCurrent: () => true, latest: () => null } });
    const started = finishDesktopStartup(() => room, assert.fail, () => {});
    // The chat fixture flattens template descendants. Mount a header with its actual button underneath it.
    const bar = document.createElement('header');
    env.mount.appendChild(bar);
    bar.innerHTML = panicButtonHtml();
    const header = createRoomHeader({ bar, ws });
    t.after(() => header.destroy());
    bar.querySelector('[data-chat-command="panic"]').click();
    assert.deepEqual(native, ['bound-document']);  // before any reply from room open
    assert.equal(ws.ws, null);
    resolveOpen({});
    await started;
    room.composer.setDraft('/panic');
    const key = { key: 'Enter', preventDefault() { this.defaultPrevented = true; } };
    for (const listener of room.composer.input.listeners.get('keydown')) listener(key);
    assert.deepEqual(native, ['bound-document', 'bound-document']);
    await Promise.resolve();
    assert.equal(room.composer.getDraft(), '');
    assert.deepEqual(ws._pendingMessages, []);
    assert.deepEqual(warnings, []);

    rejectNext = true;
    const socket = liveSocket();
    await panicWithoutSocket({}, socket, assert.fail);
    assert.deepEqual(native, Array(3).fill('bound-document'));
    assert.deepEqual(socket.sent, [], 'missing native reply must not replay the stop on another transport');
    assert.match(document.body.querySelector('.toast').children[0].textContent, /Panic was not confirmed/);
});

test('a lost reply from the stamped bridge fallback does not resend through the socket', async t => {
    const env = installDom(), calls = [];
    t.after(() => { delete globalThis.pywebview; restoreDom(env.prior); });
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: {
        ouroborosRoomSession: { postMessage: () => ({}) },
    } } });
    host.top = host;
    const doc = { documentElement: { dataset: {} }, querySelector: () => ({ content: 'this-document', remove() {} }) };
    await openDesktopRoomSession(host, doc, assert.fail);
    globalThis.pywebview = { api: { panic_stop(ticket) { calls.push(ticket); return Promise.reject(Error('reply missed')); } } };
    const socket = liveSocket();
    await panicWithoutSocket({}, socket, assert.fail);
    assert.deepEqual(calls, ['this-document']);
    assert.deepEqual(socket.sent, []);
});
