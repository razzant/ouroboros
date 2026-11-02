import test from 'node:test';
import assert from 'node:assert/strict';

import { bridgeChunkBuffer, moduleBridgeScript, moduleResizeScript } from '../modules/widget_frame.js';
import { FakeMessageChannel } from './helpers/fake_channel.js';

// Runs the child bootstrap against a fake `window`; `deliver` plays a parent → child message on the channel
// the bootstrap handed over in its hello, `posted` records child → parent messages on that channel, and
// `windowPosted` anything posted to the parent window itself (only the hello).
function bridgeHarness({ active = false, activationApi = true, initialTheme = '' } = {}) {
    const posted = [];
    const windowPosted = [];
    let hostPort = null;
    const parent = {
        postMessage(message, _origin, transfer = []) {
            windowPosted.push(message);
            if (message?.type === 'ouro-widget-hello') {
                hostPort = transfer[0];
                hostPort.onmessage = (event) => posted.push(event.data);
            }
        },
    };
    const listeners = new Map();
    const docListeners = new Map();
    const originalOpen = () => null;
    const window = {
        parent,
        MessageChannel: FakeMessageChannel,
        open: originalOpen,
        navigator: activationApi ? { userActivation: { isActive: active } } : {},
        document: {
            addEventListener(type, fn) { const list = docListeners.get(type) || []; list.push(fn); docListeners.set(type, list); },
            removeEventListener(type, fn) { docListeners.set(type, (docListeners.get(type) || []).filter((item) => item !== fn)); },
        },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (listeners.get(type) === listener) listeners.delete(type);
        },
    };
    Function('window', moduleBridgeScript('nonce-1', '', initialTheme))(window);
    // The parent speaks only on the channel; anything from another window would arrive as a window message,
    // which the bridge does not listen to.
    const deliver = (data, source = parent) => {
        const message = { nonce: 'nonce-1', ...data };
        if (source === parent) hostPort?.postMessage(message);
        else listeners.get('message')?.({ source, data: message });
    };
    const chunk = (id, phase, extra = {}) => deliver({ type: 'ouro-widget-fetch-chunk', id, phase, ...extra });
    const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
    return { window, posted, windowPosted, listeners, docListeners, originalOpen, deliver, chunk, flush, port: () => hostPort };
}

const bytes = (...values) => new Uint8Array(values).buffer;

test('both bootstrap templates are valid scripts and the bridge opens no transport of its own', () => {
    assert.doesNotThrow(() => Function(moduleBridgeScript('n')));
    assert.doesNotThrow(() => Function(moduleResizeScript(320, 8192, 2)));
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
    deliver({ nonce: 'other', type: 'ouro-widget-event', event: 'tick', data: {} });
    assert.equal(seen.length, 2);
});

test('theme is opt-in, starts from the injected resolved palette and follows live updates', () => {
    const { window, posted, deliver } = bridgeHarness({ initialTheme: 'light' });
    const seen = [];
    const off = window.OuroborosWidget.onTheme((theme) => seen.push(theme));
    assert.deepEqual(seen, ['light']);
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-theme', nonce: 'nonce-1', op: 'subscribe' });
    // The parent may answer the handshake with the same value; it must not
    // repaint a host widget twice at mount.
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, ['light']);
    deliver({ type: 'ouro-widget-theme', theme: 'system' });
    deliver({ type: 'ouro-widget-theme', theme: 'dark' });
    assert.deepEqual(seen, ['light', 'dark']);
    off();
    assert.deepEqual(posted.at(-1), { type: 'ouro-widget-theme', nonce: 'nonce-1', op: 'unsubscribe' });
});

test('theme callbacks reject foreign sources, invalid values and dispose cleanly', async () => {
    const { window, listeners, deliver, posted, flush } = bridgeHarness({ initialTheme: 'dark' });
    const seen = [];
    window.OuroborosWidget.onTheme((theme) => seen.push(theme));
    deliver({ type: 'ouro-widget-theme', theme: 'light' }, {});
    deliver({ nonce: 'wrong', type: 'ouro-widget-theme', theme: 'light' });
    deliver({ type: 'ouro-widget-theme', theme: 'system' });
    assert.deepEqual(seen, ['dark']);
    deliver({ type: 'ouro-widget-dispose' });
    await flush();
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, ['dark']);
    const count = posted.length;
    const lateOff = window.OuroborosWidget.onTheme(() => {});
    lateOff();
    assert.equal(posted.length, count);
});

test('a reentrant first theme callback still subscribes the parent once', () => {
    const { window, posted } = bridgeHarness({ initialTheme: 'dark' });
    let nestedOff = () => {};
    window.OuroborosWidget.onTheme(() => {
        nestedOff = window.OuroborosWidget.onTheme(() => {});
    });
    assert.equal(posted.filter((message) => message.type === 'ouro-widget-theme' && message.op === 'subscribe').length, 1);
    nestedOff();
});

test('a listener replaced during a live theme delivery never receives null', () => {
    const { window, deliver } = bridgeHarness({ initialTheme: 'dark' });
    const seen = [];
    let off = () => {};
    off = window.OuroborosWidget.onTheme((theme) => {
        seen.push(['first', theme]);
        if (theme === 'light') {
            off();
            window.OuroborosWidget.onTheme((next) => seen.push(['replacement', next]));
        }
    });
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, [['first', 'dark'], ['first', 'light']]);
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, [['first', 'dark'], ['first', 'light'], ['replacement', 'light']]);
});

test('unsubscribing a sibling during delivery suppresses its pending callback', () => {
    const { window, deliver } = bridgeHarness({ initialTheme: 'dark' });
    const seen = [];
    let live = false;
    let offSibling = () => {};
    window.OuroborosWidget.onTheme(() => {
        if (live) offSibling();
        seen.push('first');
    });
    offSibling = window.OuroborosWidget.onTheme(() => seen.push('sibling'));
    live = true;
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, ['first', 'sibling', 'first']);
});

test('re-registering the same sibling callback does not duplicate live delivery', () => {
    const { window, deliver } = bridgeHarness({ initialTheme: 'dark' });
    const seen = [];
    let live = false;
    let offSibling = () => {};
    const sibling = (theme) => {
        seen.push(['sibling', theme]);
    };
    window.OuroborosWidget.onTheme((theme) => {
        seen.push(['first', theme]);
        if (live && theme === 'light') {
            offSibling();
            offSibling = window.OuroborosWidget.onTheme(sibling);
        }
    });
    offSibling = window.OuroborosWidget.onTheme(sibling);
    live = true;
    deliver({ type: 'ouro-widget-theme', theme: 'light' });
    assert.deepEqual(seen, [
        ['first', 'dark'],
        ['sibling', 'dark'],
        ['first', 'light'],
        ['sibling', 'light'],
    ]);
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

test('an in-frame fault posts one bounded, deduplicated ouro-widget-error', () => {
    const { posted, listeners } = bridgeHarness();
    const long = 'x'.repeat(900);
    listeners.get('error')({ message: long, filename: 'y'.repeat(400), lineno: '17' });
    const faults = posted.filter((item) => item.type === 'ouro-widget-error');
    assert.equal(faults.length, 1);
    assert.equal(faults[0].nonce, 'nonce-1');
    assert.equal(faults[0].kind, 'error');
    assert.equal(faults[0].message.length, 500);
    assert.equal(faults[0].source.length, 200);
    assert.equal(faults[0].line, 17);

    // Same kind + message posts nothing further.
    listeners.get('error')({ message: long, filename: 'z', lineno: 1 });
    assert.equal(posted.filter((item) => item.type === 'ouro-widget-error').length, 1);

    // A rejection and a CSP refusal are their own kinds.
    listeners.get('unhandledrejection')({ reason: new Error('boom') });
    listeners.get('securitypolicyviolation')({ violatedDirective: 'script-src', blockedURI: 'https://cdn' });
    const kinds = posted.filter((item) => item.type === 'ouro-widget-error').map((item) => item.kind);
    assert.deepEqual(kinds, ['error', 'rejection', 'csp']);
});

test('the fault channel caps at 10 posts per frame and is removed on dispose', async () => {
    const { posted, listeners, deliver, flush } = bridgeHarness();
    for (let i = 0; i < 25; i += 1) listeners.get('error')({ message: `fault ${i}`, filename: '', lineno: i });
    assert.equal(posted.filter((item) => item.type === 'ouro-widget-error').length, 10);

    deliver({ type: 'ouro-widget-dispose' });
    await flush();
    for (const type of ['error', 'unhandledrejection', 'securitypolicyviolation', 'message']) {
        assert.equal(listeners.has(type), false, type);
    }
});


test('body pull grants one next chunk only when the consumer reads', async () => {
    const { window, posted, chunk, flush } = bridgeHarness();
    const pending = window.fetch('/api/extensions/s/stream');
    chunk(1, 'headers', { status: 200, headers: [] });
    const response = await pending;
    const pulls = () => posted.filter((message) => message.type === 'ouro-widget-fetch-pull');
    await flush();
    assert.equal(pulls().length, 0);
    const reader = response.body.getReader();
    const first = reader.read();
    await flush();
    assert.equal(pulls().length, 1);
    chunk(1, 'data', { chunk: bytes(1, 2) });
    assert.deepEqual(Array.from((await first).value), [1, 2]);
    await flush();
    assert.equal(pulls().length, 1);
    const last = reader.read();
    await flush();
    assert.equal(pulls().length, 2);
    chunk(1, 'end');
    assert.equal((await last).done, true);
});

test('download uses the nonce bridge and settles from the actual host outcome', async () => {
    const { window, posted, deliver, listeners, flush } = bridgeHarness();
    const blob = new Blob(['report'], { type: 'text/plain' });
    let settled = false;
    const pending = window.OuroborosWidget.download('report.txt', blob).then((result) => { settled = true; return result; });
    assert.equal(posted[0].type, 'ouro-widget-download');
    assert.equal(posted[0].source, blob);
    assert.equal(posted[0].name, 'report.txt');
    deliver({
        nonce: 'foreign', type: 'ouro-widget-download-result', id: 1, result: { ok: true },
    });
    await flush();
    assert.equal(settled, false);
    deliver({ type: 'ouro-widget-download-result', id: 1, result: { ok: true, native: true, filename: 'report.txt' } });
    assert.deepEqual(await pending, { ok: true, native: true, filename: 'report.txt' });
    const failure = window.OuroborosWidget.download('bad.txt', blob);
    deliver({ type: 'ouro-widget-download-result', id: 2, result: { ok: false, error: 'disk full' } });
    await assert.rejects(failure, /disk full/);
});


test('external-link request requires activation, checks parent and settles during disposal', async () => {
    const h = bridgeHarness({ active: true });
    const result = h.window.OuroborosWidget.openExternal('https://example.com');
    assert.deepEqual(h.posted.at(-1), { type: 'ouro-widget-open-external', nonce: 'nonce-1', id: 1, url: 'https://example.com/' });
    h.deliver({ type: 'ouro-widget-open-external-result', id: 1, result: { ok: true } }, {});
    h.deliver({ type: 'ouro-widget-open-external-result', id: 1, nonce: 'wrong', result: { ok: true } });
    h.deliver({ type: 'ouro-widget-open-external-result', id: 1, result: { ok: false, error: 'host unavailable' } });
    await assert.rejects(result, /host unavailable/);
    h.window.navigator.userActivation.isActive = false;
    await assert.rejects(h.window.OuroborosWidget.openExternal('https://example.com'), /user action/);
    h.window.navigator.userActivation.isActive = true;
    await assert.rejects(h.window.OuroborosWidget.openExternal('/relative'), /Unsupported/);
    const pending = h.window.OuroborosWidget.openExternal('https://example.com/again');
    const rejected = assert.rejects(pending, /disposed/);
    h.deliver({ type: 'ouro-widget-dispose' });
    await rejected;
    assert.equal(h.window.open, h.originalOpen);
    assert.equal(h.docListeners.get('click').length, 0);
});

test('trusted anchors work on old hosts; synthetic, handled, download and internal links are untouched', async () => {
    const h = bridgeHarness({ activationApi: false });
    await assert.rejects(h.window.OuroborosWidget.openExternal('https://example.com'), /user action/);
    const click = (href, changes = {}) => {
        const anchor = { getAttribute() { return href; }, hasAttribute(name) { return name === 'download' && changes.download; } };
        const event = { isTrusted: true, button: 0, defaultPrevented: false, target: { closest(selector) { return selector === 'a[href]' ? anchor : null; } },
            preventDefault() { this.defaultPrevented = true; }, ...changes };
        for (const fn of h.docListeners.get('click')) fn(event);
        return event;
    };
    for (const [href, changes] of [['#x', {}], ['/relative', {}], ['https://example.com', { isTrusted: false }], ['https://example.com', { defaultPrevented: true }], ['https://example.com', { download: true }]]) click(href, changes);
    assert.equal(h.posted.length, 0);
    assert.equal(click('https://example.com').defaultPrevented, true);
    assert.equal(h.posted.length, 1);
    h.deliver({ type: 'ouro-widget-open-external-result', id: 1, result: { ok: true } });
});

test('the bootstrap says hello once to its parent window with its nonce and a port, and nothing else', () => {
    const { windowPosted, posted, port, window } = bridgeHarness();
    assert.deepEqual(windowPosted, [{ type: 'ouro-widget-hello', nonce: 'nonce-1' }]);
    assert.ok(port(), 'the hello carries the channel');
    window.OuroborosWidget.onEvent(() => {});
    window.fetch('/api/extensions/s/x');
    assert.equal(windowPosted.length, 1, 'requests travel on the channel, never to the parent window');
    assert.deepEqual(posted.map((m) => m.type), ['ouro-widget-events', 'ouro-widget-fetch']);
});

test('a message from any window, even the parent window, is not heard: only the channel speaks', async () => {
    const { window, listeners, deliver, flush } = bridgeHarness();
    const seen = [];
    window.OuroborosWidget.onEvent((event) => seen.push(event));
    assert.equal(listeners.has('message'), false);
    deliver({ type: 'ouro-widget-event', event: 'tick', data: {} }, {});
    deliver({ type: 'ouro-widget-event', event: 'tick', data: { n: 1 } });
    await flush();
    assert.deepEqual(seen, [{ type: 'tick', data: { n: 1 } }]);
});

test('ready resolves when the parent accepts the channel', async () => {
    const { window, deliver } = bridgeHarness();
    let ready = false;
    window.OuroborosWidget.ready.then(() => { ready = true; });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(ready, false);
    deliver({ type: 'ouro-widget-ready', routeBase: 'https://h/api/extensions/s/' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(ready, true);
});
