import assert from 'node:assert/strict';
import test from 'node:test';
import { createRoomHeader, panicButtonHtml } from '../modules/chat_header.js';
import { createChatInstance } from '../modules/chat.js';
import { desktopRoomStorage, finishDesktopStartup, openDesktopRoomSession } from '../modules/desktop_room_session.js';
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

// Sockets and helpers of the cases below that use inert receipts (no real stop, signal or server is involved).
function quietSocket(readyState = 1) {
    const socket = new EventTarget();
    const listeners = new Set();
    const add = socket.addEventListener.bind(socket), remove = socket.removeEventListener.bind(socket);
    socket.addEventListener = (type, fn) => { listeners.add(fn); add(type, fn); };
    socket.removeEventListener = (type, fn) => { listeners.delete(fn); remove(type, fn); };
    return Object.assign(socket, { readyState, bufferedAmount: 0, sent: [], listeners,
        send: text => socket.sent.push(JSON.parse(text)),
        ack() {
            const event = new Event('message');
            event.data = JSON.stringify({ type: 'panic_accepted' });
            socket.dispatchEvent(event);
        } });
}

const flush = async () => { for (let i = 0; i < 6; i += 1) await Promise.resolve(); };
const toasts = env => env.doc.body.querySelectorAll('.toast').length;

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

// An older launcher's bridge (a `panic_stop()` that takes no parameter): its one call is the stop. pywebview's promise for
// it has no timeout and a launcher may decline a call without answering, so the press waits only a bounded time.
test('an older launcher\'s bridge that never answers is waited for briefly, then the captured socket is asked; a late answer settles once', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [], receipts = [];
    globalThis.pywebview = { api: { panic_stop(...args) { calls.push(args); return new Promise((resolve, reject) => receipts.push({ resolve, reject })); } } };
    const socket = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
    assert.deepEqual(calls, [[]], 'the one call is made in the press itself, with no ticket');
    assert.deepEqual(socket.sent, []);
    t.mock.timers.tick(999);
    await flush();
    assert.deepEqual(socket.sent, [], 'an answer still owed within its bound is not a refusal');
    t.mock.timers.tick(1);
    await flush();
    assert.equal(socket.sent.length, 1, 'then the socket the press captured is asked, once');
    await press;
    receipts[0].resolve({ status: 'stopping' });
    await flush();
    assert.equal(accepted, 1, 'the late answer settles the press');
    assert.equal(socket.listeners.size, 0);
    socket.ack();
    t.mock.timers.tick(5000);
    await flush();
    assert.deepEqual([accepted, toasts(env), calls.length, socket.sent.length], [1, 0, 1, 1], 'nothing more settles or is asked');
});

test('an older launcher\'s refusal that comes after its bound does not end a press whose socket is still answering', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const receipts = [];
    globalThis.pywebview = { api: { panic_stop: () => new Promise(resolve => receipts.push(resolve)) } };
    const socket = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
    t.mock.timers.tick(1000);
    await flush();
    assert.equal(socket.sent.length, 1);
    receipts[0]({ status: 'refused' });   // too late to end the press: the socket has been asked and may still confirm
    await flush();
    assert.deepEqual([accepted, toasts(env)], [0, 0]);
    socket.ack();
    assert.deepEqual([accepted, toasts(env)], [1, 0]);
    await press;
});

test('an older launcher\'s bridge that throws or rejects asks the socket at once; one that refuses within its bound asks nothing', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    let shown = 0;
    for (const [panic_stop, asked] of [[() => { throw Error('no door'); }, 1], [async () => { throw Error('no answer'); }, 1],
        [async () => ({ status: 'refused' }), 0]]) {
        globalThis.pywebview = { api: { panic_stop } };
        const socket = liveSocket();
        let accepted = 0;
        await panicWithoutSocket({}, socket, () => { accepted += 1; });
        assert.deepEqual([socket.sent.length, accepted], [asked, asked]);
        if (!asked) shown += 1;   // a refusal says it was not confirmed, once
        assert.equal(toasts(env), shown);
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

test('a stamped desktop button during open and typed Panic use the bound native channel; its refusal falls through to the captured socket', async t => {
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
    let accepted = 0;
    const socket = liveSocket();
    await panicWithoutSocket({}, socket, () => { accepted += 1; });
    assert.deepEqual(native, Array(3).fill('bound-document'));
    // The refused door falls through to the captured live socket: /panic is idempotent, so a stop the
    // door never took still reaches the server, and its ack — not a toast — ends the gesture.
    assert.deepEqual(socket.sent, [{ type: 'command', cmd: '/panic', client_surface: {} }]);
    assert.equal(accepted, 1);
    assert.equal(document.body.querySelectorAll('.toast').length, 0, 'the second door confirmed the stop');
});

test('a lost reply from the stamped bridge falls through to the captured live socket', async t => {
    const env = installDom(), calls = [];
    t.after(() => { delete globalThis.pywebview; restoreDom(env.prior); });
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: {
        ouroborosRoomSession: { postMessage: () => ({}) },
    } } });
    host.top = host;
    const doc = { documentElement: { dataset: {} }, querySelector: () => ({ content: 'this-document', remove() {} }) };
    await openDesktopRoomSession(host, doc, assert.fail);
    globalThis.pywebview = { api: { panic_stop(ticket) { calls.push(ticket); return Promise.reject(Error('reply missed')); } } };
    let accepted = 0;
    const socket = liveSocket();
    await panicWithoutSocket({}, socket, () => { accepted += 1; });
    assert.deepEqual(calls, ['this-document']);
    // The bound door's unconfirmed stop is only asked for again through the socket; the idempotent
    // command cannot double a stop, and the captured socket never queues or reconnects.
    assert.deepEqual(socket.sent, [{ type: 'command', cmd: '/panic', client_surface: {} }]);
    assert.equal(accepted, 1);
});

// The cases below use inert receipts: a stamped document whose Panic door answers only when the test says so, and
// sockets that record what they are sent. No real stop, signal or server is involved.
async function stampedDocument({ ticket = 'stamped-document', door = true } = {}) {
    const asked = [], receipts = [];
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: {
        ouroborosRoomSession: { postMessage: () => ({}) },
        ...(door && { ouroborosDesktopPanic: { postMessage(sent) {
            asked.push(sent);
            return new Promise((resolve, reject) => receipts.push({ resolve, reject }));
        } } }),
    } } });
    host.top = host;
    const doc = { documentElement: { dataset: {} }, querySelector: () => ({ content: ticket, remove() {} }) };
    await openDesktopRoomSession(host, doc, assert.fail);
    return { asked, answer: reply => receipts.shift().resolve(reply), refuse: () => receipts.shift().reject(Error('refused')) };
}

test('a stamped reply that never comes is waited for briefly, then only the captured socket is asked; a late reply settles nothing twice', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const door = await stampedDocument();
    const socket = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
    assert.deepEqual(door.asked, ['stamped-document'], 'the stamped door is asked first, before the press returns');
    assert.deepEqual(socket.sent, [], 'and the socket is not');
    t.mock.timers.tick(999);
    await flush();
    assert.deepEqual(socket.sent, [], 'a reply still owed within its bound is not a refusal');
    t.mock.timers.tick(1);
    await flush();
    assert.deepEqual(socket.sent, [{ type: 'command', cmd: '/panic', client_surface: {} }],
        'with no reply the press asks the captured socket, once');
    await press;
    door.answer({ status: 'stopping' });   // the door answers late, while the socket's ack is still owed
    await flush();
    assert.equal(accepted, 1, 'the late reply settles the press');
    assert.equal(socket.listeners.size, 0, 'and its socket listeners are gone');
    socket.ack();
    t.mock.timers.tick(5000);
    await flush();
    assert.deepEqual([accepted, toasts(env)], [1, 0], 'the socket\'s own ack and timer after that settle nothing');
});

test('a refused stamped reply asks the socket the press captured at once; a successor connection is never asked', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    const door = await stampedDocument();
    const ws = new WS('unused');
    ws.connect = () => assert.fail('Panic cannot reconnect');
    const first = liveSocket(), successor = liveSocket();
    ws.ws = first;
    ws.send({ type: 'command', cmd: '/panic' });
    ws.ws = successor;   // the server restarted while the door was answering
    door.refuse();
    await flush();
    assert.equal(first.sent.length, 1, 'the connection the press held');
    assert.deepEqual(successor.sent, [], 'never the one that replaced it');
    assert.deepEqual([ws._pendingMessages.length, toasts(env)], [0, 0], 'no replay queue and no failure to report');
});

test('a captured socket that is gone when the door stays silent is stale: the press fails once and sends nothing', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const door = await stampedDocument();
    const socket = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
    socket.readyState = 3;
    socket.dispatchEvent(new Event('close'));
    t.mock.timers.tick(1000);
    await flush();
    assert.deepEqual([socket.sent, toasts(env)], [[], 1], 'a closed connection is not asked and the failure is said');
    await press;
    door.answer({ status: 'stopping' });   // after the failure was said, a reply cannot also accept
    await flush();
    assert.equal(accepted, 0);
    assert.equal(toasts(env), 1);
});

test('a socket that was not open and quiet at the press is never used later, even if it opens while the door is pending', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    await stampedDocument();
    const connecting = quietSocket(0), backlogged = Object.assign(quietSocket(), { bufferedAmount: 1 });
    let said = 0;
    for (const socket of [connecting, backlogged]) {
        const press = panicWithoutSocket({}, socket, () => assert.fail('nothing confirmed this press'));
        socket.readyState = 1;
        socket.bufferedAmount = 0;
        t.mock.timers.tick(1000);
        await flush();
        assert.deepEqual(socket.sent, []);
        assert.equal(toasts(env), said += 1, 'each press says it was not confirmed, once');
        await press;
    }
});

test('the door\'s reply racing its bound settles the press once, in either order', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const door = await stampedDocument();
    for (const replyFirst of [true, false]) {
        const socket = quietSocket();
        let accepted = 0;
        const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
        if (replyFirst) { door.answer({ status: 'stopping' }); t.mock.timers.tick(1000); }
        else { t.mock.timers.tick(1000); door.answer({ status: 'stopping' }); }
        await press;
        await flush();
        socket.ack();
        t.mock.timers.tick(5000);
        await flush();
        assert.equal(accepted, 1, `reply ${replyFirst ? 'before' : 'after'} the bound`);
        assert.ok(socket.sent.length <= (replyFirst ? 0 : 1), 'at most the one idempotent ask, and none when the reply came first');
        assert.equal(socket.listeners.size, 0);
    }
    assert.equal(toasts(env), 0);
});

test('a bound pywebview bridge that never replies falls through the same way, and its refusal does at once', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    await stampedDocument({ door: false });
    const receipts = [], calls = [];
    globalThis.pywebview = { api: { panic_stop(ticket) { calls.push(ticket); return new Promise((resolve, reject) => receipts.push({ resolve, reject })); } } };
    const silent = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, silent, () => { accepted += 1; });
    assert.deepEqual(calls, ['stamped-document']);
    assert.deepEqual(silent.sent, []);
    t.mock.timers.tick(1000);
    await flush();
    assert.equal(silent.sent.length, 1);
    await press;
    receipts[0].resolve({ status: 'stopping' });
    await flush();
    assert.equal(accepted, 1);

    const refused = quietSocket();
    const second = panicWithoutSocket({}, refused, () => { accepted += 1; });
    receipts[1].reject(Error('refused'));
    await second;
    assert.equal(refused.sent.length, 1, 'a refusal needs no bound');
    refused.ack();
    assert.equal(accepted, 2);
    assert.equal(toasts(env), 0);
});

test('a stamped document whose launcher bridge takes no ticket is handed none, and its silence falls through to the socket', async t => {
    const env = installDomDouble();
    t.after(env.restore);
    t.after(() => delete globalThis.pywebview);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    await stampedDocument({ door: false });
    const calls = [];
    globalThis.pywebview = { api: { panic_stop: (...args) => { calls.push(args); return new Promise(() => {}); } } };  // a call it declines
    const socket = quietSocket();
    let accepted = 0;
    const press = panicWithoutSocket({}, socket, () => { accepted += 1; });
    assert.deepEqual(calls, [[]], 'a function that publishes no parameter is never handed the document\'s ticket');
    t.mock.timers.tick(1000);
    await flush();
    assert.equal(socket.sent.length, 1, 'the unanswered call is no stop: the captured socket is asked');
    await press;
    socket.ack();
    assert.equal(accepted, 1);
});

test('typed Panic reaches the stamped door while a settings save is still in flight', async t => {
    const env = installDom(async url => {
        if (String(url) === '/api/owner/context-mode') return new Promise(() => {});  // the save never answers
        return { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } };
    });
    let room;
    t.after(() => { room?.destroy(); restoreDom(env.prior); });
    const door = await stampedDocument();
    const ws = new WS('unused');
    ws.connect = () => assert.fail('Panic cannot reconnect');
    room = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set() }, mountEl: env.mount,
        stateSnapshots: { begin: () => ({}), gate: async () => null, isCurrent: () => true, latest: () => null } });
    desktopRoomStorage().enableInput();
    room.composer.setDraft('plain words');
    const capsule = document.byId.get('chat-context-mode');
    const seg = capsule.querySelector('.chat-seg[data-mode="low"]');
    for (const listener of capsule.listeners.get('click')) listener({ target: { closest: selector => (selector === '.chat-seg' ? seg : null) } });
    room.composer.setDraft('/panic');
    for (const listener of room.composer.input.listeners.get('keydown')) listener({ key: 'Enter', preventDefault() {} });
    assert.deepEqual(door.asked, ['stamped-document'], 'the door was asked in the press itself, not after the save');
    assert.equal(room.composer.getDraft(), '/panic', 'the draft stays until the door confirms');
    door.answer({ status: 'stopping' });
    await flush();
    assert.equal(room.composer.getDraft(), '');
});
