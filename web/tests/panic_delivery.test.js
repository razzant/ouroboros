import assert from 'node:assert/strict';
import test from 'node:test';
import { WS } from '../modules/ws.js';
import { panicWithoutSocket } from '../modules/panic.js';
import { installDomDouble } from './dom_double.js';

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
    assert.doesNotMatch(env.doc.body.textContent, /tray|menu bar/);
});

test('legacy queued Panic is discarded at reconnect while ordinary chat survives', () => {
    globalThis.WebSocket = { OPEN: 1 };
    try {
        const ws = new WS('ws://unused');
        const sent = [];
        ws.ws = { readyState: 1, send: text => sent.push(JSON.parse(text)) };
        ws._pendingMessages = [{ type: 'chat', content: '/panic' }, { type: 'chat', content: 'hello' }];
        ws._flushPendingMessages();
        assert.deepEqual(sent, [{ type: 'chat', content: 'hello' }]);
    } finally { delete globalThis.WebSocket; }
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
