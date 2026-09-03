import test from 'node:test';
import assert from 'node:assert/strict';

import { bridgeChunkBuffer, moduleBridgeScript, moduleResizeScript } from '../modules/widget_frame.js';

// Runs the child bootstrap against a fake `window`; `deliver` plays a
// parent → child message, `posted` records child → parent messages.
function bridgeHarness() {
    const posted = [];
    const parent = { postMessage(message) { posted.push(message); } };
    const listeners = new Map();
    const window = {
        parent,
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (listeners.get(type) === listener) listeners.delete(type);
        },
    };
    Function('window', moduleBridgeScript('nonce-1'))(window);
    const deliver = (data, source = parent) => listeners.get('message')?.({ source, data: { nonce: 'nonce-1', ...data } });
    const chunk = (id, phase, extra = {}) => deliver({ type: 'ouro-widget-fetch-chunk', id, phase, ...extra });
    const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
    return { window, posted, listeners, deliver, chunk, flush };
}

const bytes = (...values) => new Uint8Array(values).buffer;

test('both bootstrap templates are valid scripts and the bridge opens no transport of its own', () => {
    assert.doesNotThrow(() => Function(moduleBridgeScript('n')));
    assert.doesNotThrow(() => Function(moduleResizeScript('n', 320, 8192, 2)));
    const child = moduleBridgeScript('n');
    for (const forbidden of ['EventSource(', 'WebSocket(', 'setTimeout(', 'XMLHttpRequest', 'ouro-widget-fetch-result']) {
        assert.equal(child.includes(forbidden), false, forbidden);
    }
});

test('bridgeChunkBuffer transfers exactly the bytes a view covers', () => {
    const whole = new Uint8Array([1, 2, 3]);
    assert.equal(bridgeChunkBuffer(whole), whole.buffer);
    const window = new Uint8Array(new Uint8Array([9, 1, 2, 3, 9]).buffer, 1, 3);
    const copy = bridgeChunkBuffer(window);
    assert.notEqual(copy, window.buffer);
    assert.deepEqual(Array.from(new Uint8Array(copy)), [1, 2, 3]);
    const raw = bytes(7, 8);
    assert.equal(bridgeChunkBuffer(raw), raw);
});

test('a bridged fetch streams a binary body into a real Response with every header', async () => {
    const { window, posted, chunk } = bridgeHarness();
    const pending = window.fetch('/api/extensions/s/blob', { headers: { 'X-Probe': '1' } });
    assert.deepEqual(posted, [{
        type: 'ouro-widget-fetch',
        nonce: 'nonce-1',
        id: 1,
        url: '/api/extensions/s/blob',
        init: { method: 'GET', headers: [['x-probe', '1']], body: null, timeoutMs: null },
    }]);
    chunk(1, 'headers', {
        status: 201,
        statusText: 'Created',
        headers: [['content-type', 'application/octet-stream'], ['x-blob', '7']],
    });
    const response = await pending;
    assert.equal(response.status, 201);
    assert.equal(response.statusText, 'Created');
    assert.equal(response.headers.get('content-type'), 'application/octet-stream');
    assert.equal(response.headers.get('x-blob'), '7');
    chunk(1, 'data', { chunk: bytes(1, 2, 3) });
    chunk(1, 'data', { chunk: bytes(4, 5) });
    chunk(1, 'end');
    assert.deepEqual(Array.from(new Uint8Array(await response.arrayBuffer())), [1, 2, 3, 4, 5]);
    // The abort signal and timeout knob travel in `init`.
    const controller = new AbortController();
    window.fetch('/api/extensions/s/x', { method: 'post', body: 'b', signal: controller.signal, timeoutMs: 250 });
    assert.equal(posted.at(-1).init.method, 'POST');
    assert.equal(posted.at(-1).init.body, 'b');
    assert.equal(posted.at(-1).init.timeoutMs, 250);
});

test('incremental body reads observe each data frame before end', async () => {
    const { window, chunk } = bridgeHarness();
    const pending = window.OuroborosWidget.fetch('/api/extensions/s/stream');
    chunk(1, 'headers', { status: 200, statusText: 'OK', headers: [['content-type', 'text/event-stream']] });
    const reader = (await pending).body.getReader();
    const first = reader.read();
    chunk(1, 'data', { chunk: bytes(97) });
    assert.deepEqual(Array.from((await first).value), [97]);
    chunk(1, 'data', { chunk: bytes(98) });
    assert.deepEqual(Array.from((await reader.read()).value), [98]);
    const last = reader.read();
    chunk(1, 'end');
    assert.equal((await last).done, true);
});

test('HEAD and 204/205/304 answers settle with a null body', async () => {
    const { window, chunk } = bridgeHarness();
    const head = window.fetch('/api/extensions/s/ping', { method: 'HEAD' });
    chunk(1, 'headers', { status: 200, statusText: 'OK', headers: [['content-length', '12']] });
    const headResponse = await head;
    assert.equal(headResponse.body, null);
    assert.equal(await headResponse.text(), '');
    chunk(1, 'end');
    const empty = window.fetch('/api/extensions/s/nobody');
    chunk(2, 'headers', { status: 204, statusText: 'No Content', headers: [] });
    assert.equal((await empty).body, null);
    for (const [id, status, statusText] of [[3, 205, 'Reset Content'], [4, 304, 'Not Modified']]) {
        const pending = window.fetch(`/api/extensions/s/nobody${id}`);
        chunk(id, 'headers', { status, statusText, headers: [] });
        const response = await pending;
        assert.equal(response.body, null);
        assert.equal(response.status, status);
    }
});

test('abort posts -fetch-abort and rejects before headers or errors the body after', async () => {
    const { window, posted, chunk } = bridgeHarness();
    const early = new AbortController();
    const rejected = window.fetch('/api/extensions/s/a', { signal: early.signal });
    early.abort();
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-fetch-abort', nonce: 'nonce-1', id: 1 });
    await assert.rejects(rejected, (err) => err.name === 'AbortError');
    const late = new AbortController();
    const streaming = window.fetch('/api/extensions/s/b', { signal: late.signal });
    chunk(2, 'headers', { status: 200, statusText: 'OK', headers: [] });
    const reader = (await streaming).body.getReader();
    late.abort();
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-fetch-abort', nonce: 'nonce-1', id: 2 });
    await assert.rejects(reader.read(), (err) => err.name === 'AbortError');
    // Cancelling the body stream is the same abort from the parent's side.
    const cancelled = window.fetch('/api/extensions/s/c');
    chunk(3, 'headers', { status: 200, statusText: 'OK', headers: [] });
    await (await cancelled).body.cancel();
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-fetch-abort', nonce: 'nonce-1', id: 3 });
    // An already-aborted signal never reaches the parent.
    const before = posted.length;
    const dead = new AbortController();
    dead.abort();
    await assert.rejects(window.fetch('/api/extensions/s/d', { signal: dead.signal }), (err) => err.name === 'AbortError');
    assert.equal(posted.length, before);
});

test('an error frame rejects an unsettled fetch and errors an open body', async () => {
    const { window, chunk } = bridgeHarness();
    const refused = window.fetch('/api/widgets');
    chunk(1, 'error', { error: 'module widget fetch outside extension route prefix' });
    await assert.rejects(refused, /outside extension route prefix/);
    const broken = window.fetch('/api/extensions/s/stream');
    chunk(2, 'headers', { status: 200, statusText: 'OK', headers: [] });
    const reader = (await broken).body.getReader();
    chunk(2, 'error', { error: 'connection lost' });
    await assert.rejects(reader.read(), /connection lost/);
});

test('events subscribe on the first listener, deliver {type, data}, unsubscribe on the last', () => {
    const { window, posted, listeners, deliver } = bridgeHarness();
    const seen = [];
    const offA = window.OuroborosWidget.onEvent((event) => seen.push(['a', event]));
    const offB = window.OuroborosWidget.onEvent((event) => seen.push(['b', event]));
    assert.deepEqual(posted, [{ type: 'ouro-widget-events', nonce: 'nonce-1', op: 'subscribe' }]);
    deliver({ type: 'ouro-widget-event', event: 'tick', data: { n: 1 } });
    assert.deepEqual(seen, [['a', { type: 'tick', data: { n: 1 } }], ['b', { type: 'tick', data: { n: 1 } }]]);
    offA();
    offA();
    assert.equal(posted.length, 1);
    offB();
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-events', nonce: 'nonce-1', op: 'unsubscribe' });
    deliver({ type: 'ouro-widget-event', event: 'tick', data: { n: 2 } });
    assert.equal(seen.length, 2);
    // Frames from a foreign source or with another nonce are ignored.
    window.OuroborosWidget.onEvent((event) => seen.push(['c', event]));
    deliver({ type: 'ouro-widget-event', event: 'tick', data: {} }, {});
    listeners.get('message')({ source: window.parent, data: { nonce: 'other', type: 'ouro-widget-event', event: 'tick', data: {} } });
    assert.equal(seen.length, 2);
});

test('dispose awaits hooks (bridge live), acks, then fails pending work and unlistens', async () => {
    const { window, posted, listeners, deliver, chunk, flush } = bridgeHarness();
    const hookSaw = [];
    window.__ouroWidgetOnDispose(async () => {
        const saved = await window.fetch('/api/extensions/s/flush', { method: 'POST', body: '{}' });
        hookSaw.push(saved.status);
    });
    // A stream left open across the dispose.
    const open = window.fetch('/api/extensions/s/slow');
    chunk(1, 'headers', { status: 200, statusText: 'OK', headers: [] });
    const reader = (await open).body.getReader();
    const off = window.OuroborosWidget.onEvent(() => {});
    deliver({ type: 'ouro-widget-dispose' });
    await flush();
    const flush_ = posted.find((message) => message.type === 'ouro-widget-fetch' && message.url.endsWith('/flush'));
    assert.ok(flush_, 'the hook fetch went through the live bridge');
    assert.equal(posted.some((message) => message.type === 'ouro-widget-disposed'), false);
    chunk(flush_.id, 'headers', { status: 200, statusText: 'OK', headers: [] });
    chunk(flush_.id, 'end');
    await flush();
    await flush();
    assert.deepEqual(hookSaw, [200]);
    const types = posted.map((message) => message.type);
    assert.ok(types.indexOf('ouro-widget-fetch') < types.indexOf('ouro-widget-disposed'));
    await assert.rejects(reader.read(), /widget disposed/);
    await assert.rejects(window.fetch('/api/extensions/s/late'), /widget disposed/);
    assert.equal(listeners.has('message'), false);
    // A late unsubscribe after dispose posts nothing.
    const count = posted.length;
    off();
    assert.equal(posted.length, count);
    assert.deepEqual(window.OuroborosWidget.onEvent(() => {})(), undefined);
});
