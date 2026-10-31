// The document-bound frame channel (widget_module.js) and the route lease's renewal (widget_assets.js), with the
// repository DOM double and a synchronous channel double. A frame is heard only after its document's hello with the
// mount's proof; nothing is ever posted to the frame's window, so a document that later occupies the frame learns
// nothing. Real engines, real navigation and a real core restart are checked by the browser probe.
import assert from 'node:assert/strict';
import test from 'node:test';

import { maintainWidgetAssetLease } from '../modules/widget_assets.js';
import { mountModuleWidget, mountRouteIframeWidget } from '../modules/widget_module.js';
import { installCardDom } from './helpers/card_dom.js';
import { FakeMessageChannel, joinFrame, srcdocNonce } from './helpers/fake_channel.js';

const flush = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const LEASE = { token: 'L'.repeat(43), base_url: `/api/widget-resources/${'L'.repeat(43)}/`, expires_in: 120, renew_after: 40 };

function serve() {
    const calls = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        calls.push({ url: String(url), method: String(init.method || 'GET').toUpperCase(), body: init.body ?? null });
        if (url === '/api/widgets/asset-leases') return { ok: true, status: 200, json: async () => LEASE };
        if (String(url).startsWith('/api/widgets/asset-leases/')) return { ok: true, status: 200, json: async () => ({ ok: true }) };
        return { ok: true, status: 200, statusText: 'OK', headers: new Headers(), body: null,
            text: async () => '// reviewed module', json: async () => ({}) };
    };
    return { calls, restore: () => { globalThis.fetch = prior; } };
}

const TAB = { key: 'clock:face', skill: 'clock', tab_id: 'face', revision: '1', generation: 'g1', ws_prefix: 'ext:clock:' };

async function moduleFrame() {
    const card = document.createElement('div');
    card.setAttribute('data-widget-key', TAB.key);
    card.innerHTML = '<div data-widget-status hidden></div><div data-widget-mount></div>';
    const mount = card.querySelector('[data-widget-mount]');
    const handlers = new Set();
    const dispose = await mountModuleWidget(mount, TAB, { kind: 'module', entry: 'widget.js', height: 320 }, null, handlers);
    const frame = mount.querySelector('iframe');
    const windowPosts = [];
    frame.contentWindow = { postMessage: (message) => windowPosts.push(message) };
    return { card, mount, frame, handlers, dispose, windowPosts };
}

test('a module frame is heard only on the channel its own hello handed over, and never posted to', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { frame, handlers, dispose, windowPosts } = await moduleFrame();
        const emit = (event) => window.emit('message', event);
        // Another window, or the right window with another nonce, offers a port: neither is adopted.
        const foreign = new FakeMessageChannel();
        emit({ source: {}, data: { type: 'ouro-widget-hello', nonce: srcdocNonce(frame.srcdoc) }, ports: [foreign.port2] });
        emit({ source: frame.contentWindow, data: { type: 'ouro-widget-hello', nonce: 'guessed' }, ports: [foreign.port2] });
        const own = joinFrame(frame, emit);
        // A later hello, even with the right nonce (the widget's own code could repeat it), is refused.
        const second = new FakeMessageChannel();
        const secondSeen = [];
        second.port1.onmessage = (event) => secondSeen.push(event.data);
        emit({ source: frame.contentWindow, data: { type: 'ouro-widget-hello', nonce: own.nonce }, ports: [second.port2] });
        own.send({ type: 'ouro-widget-events', op: 'subscribe' });
        assert.equal(handlers.size, 1);
        [...handlers][0]({ type: 'ext:clock:tick', data: { n: 1 } });
        own.send({ type: 'ouro-widget-fetch', id: 7, url: '/api/extensions/clock/state', init: { method: 'GET', headers: [] } });
        await flush();
        assert.deepEqual(own.replies.map((m) => [m.type, m.phase || m.event]),
            [['ouro-widget-event', 'tick'], ['ouro-widget-fetch-chunk', 'headers'], ['ouro-widget-fetch-chunk', 'end']]);
        assert.deepEqual(secondSeen, []);
        // A window message from the frame's own window is not a request.
        emit({ source: frame.contentWindow, data: { type: 'ouro-widget-fetch', id: 8, nonce: own.nonce, url: '/api/extensions/clock/state', init: {} } });
        await flush();
        assert.equal(server.calls.filter((c) => c.url === '/api/extensions/clock/state').length, 1);
        const stopped = dispose();
        assert.equal(own.replies.at(-1).type, 'ouro-widget-dispose', 'the stop is asked on the channel too');
        own.send({ type: 'ouro-widget-disposed' });
        await stopped;
        assert.deepEqual(windowPosts, [], 'nothing was ever posted to the frame window');
        assert.equal(window.listeners('message'), 0);
    } finally {
        server.restore();
        dom.restore();
    }
});

test('a module frame that loads a second document is retired at once and revokes its lease', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { card, frame, handlers, windowPosts } = await moduleFrame();
        const own = joinFrame(frame, (event) => window.emit('message', event));
        own.send({ type: 'ouro-widget-events', op: 'subscribe' });
        frame.dispatch('load');            // its own srcdoc
        frame.dispatch('load');            // whatever it navigated to
        await flush();
        assert.equal(handlers.size, 0);
        assert.match(card.querySelector('[data-widget-status]').textContent, /left its own page/);
        assert.ok(!own.replies.some((m) => m.type === 'ouro-widget-dispose'), 'no handshake with the new document');
        assert.ok(server.calls.some((c) => c.url === '/api/widgets/asset-leases/revoke'));
        assert.deepEqual(windowPosts, []);
    } finally {
        server.restore();
        dom.restore();
    }
});

async function routeFrame() {
    const card = document.createElement('div');
    card.setAttribute('data-widget-key', TAB.key);
    card.innerHTML = '<div data-widget-status hidden></div><div data-widget-mount></div>';
    const mount = card.querySelector('[data-widget-mount]');
    const handlers = new Set();
    const dispose = await mountRouteIframeWidget(mount, TAB, { kind: 'iframe', route: 'page' }, null, null, handlers);
    const frame = mount.querySelector('iframe');
    frame.contentWindow = { postMessage() { throw new Error('nothing is posted to a route frame window'); } };
    const hello = (lease) => {
        const channel = new FakeMessageChannel();
        const replies = [];
        channel.port1.onmessage = (event) => replies.push(event.data);
        window.emit('message', { source: frame.contentWindow, data: { type: 'ouro-widget-hello', lease }, ports: [channel.port2] });
        return { replies, send: (data) => channel.port1.postMessage({ nonce: '', ...data }) };
    };
    return { mount, frame, handlers, dispose, hello };
}

test('a route page proves itself with the lease in its own URL and gets its skill\'s full bridge', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { frame, handlers, dispose, hello } = await routeFrame();
        assert.equal(frame.src, `http://local${LEASE.base_url}page`);
        assert.equal(frame.referrerPolicy, 'no-referrer');
        const guess = hello('G'.repeat(43));
        guess.send({ type: 'ouro-widget-events', op: 'subscribe' });
        assert.equal(handlers.size, 0, 'a document without the lease is not heard');
        assert.deepEqual(guess.replies, []);
        const page = hello(LEASE.token);
        assert.deepEqual(page.replies[0], { type: 'ouro-widget-ready', nonce: '', routeBase: 'http://local/api/extensions/clock/' });
        // A mutating call to its own route (resolved by the page against its lease URL) is relayed with the session.
        page.send({ type: 'ouro-widget-fetch', id: 1, url: `http://local${LEASE.base_url}act?x=1`, init: { method: 'POST', headers: [], body: '{}' } });
        page.send({ type: 'ouro-widget-events', op: 'subscribe' });
        await flush();
        const relayed = server.calls.filter((c) => c.url.startsWith('/api/extensions/clock/'));
        assert.deepEqual(relayed.map((c) => [c.method, c.url]), [['POST', '/api/extensions/clock/act?x=1']]);
        [...handlers][0]({ type: 'ext:clock:tick', data: { n: 2 } });
        assert.deepEqual(page.replies.filter((m) => m.type === 'ouro-widget-event').map((m) => m.data), [{ n: 2 }]);
        // Another skill or the core stays out of reach.
        page.send({ type: 'ouro-widget-fetch', id: 2, url: 'http://local/api/state', init: { method: 'GET', headers: [] } });
        await flush();
        assert.match(page.replies.find((m) => m.id === 2 && m.phase === 'error').error, /outside extension route prefix/);
        // Its next own document says hello again: the previous document's channel and subscriptions retire.
        frame.dispatch('load');
        const next = hello(LEASE.token);
        assert.equal(handlers.size, 0);
        page.send({ type: 'ouro-widget-events', op: 'subscribe' });
        assert.equal(handlers.size, 0, 'the retired channel is not heard');
        next.send({ type: 'ouro-widget-events', op: 'subscribe' });
        assert.equal(handlers.size, 1);
        // A load with no hello is a document that never joined: the channel retires with the page it belonged to.
        frame.dispatch('load');
        frame.dispatch('load');
        assert.equal(handlers.size, 0);
        await dispose();
    } finally {
        server.restore();
        dom.restore();
    }
});

// ---- the lease maintainer on a simulated clock ---------------------------------------------------------------

function clock() {
    const prior = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
    let now = 0;
    let seq = 0;
    const timers = new Map();
    globalThis.setTimeout = (fn, delay) => { const id = ++seq; timers.set(id, { fn, at: now + Number(delay || 0) }); return id; };
    globalThis.clearTimeout = (id) => timers.delete(id);
    const tick = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
    return {
        get now() { return now; },
        async advance(ms) {
            const end = now + ms;
            for (;;) {
                const due = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
                if (!due) break;
                now = due[1].at;
                timers.delete(due[0]);
                due[1].fn();
                await tick();
            }
            now = end;
            await tick();
        },
        restore() { Object.assign(globalThis, prior); },
    };
}

function renewals(script) {
    const calls = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        calls.push(String(url));
        const step = script.shift() || 'ok';
        if (step === 'hang') return new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(new Error('timed out'))));
        if (step === 'down') throw new TypeError('Failed to fetch');
        return { ok: step === 'ok', status: step === 'ok' ? 200 : Number(step), json: async () => ({ ...LEASE }) };
    };
    return { calls, restore: () => { globalThis.fetch = prior; } };
}

for (const [name, script, lostAt] of [
    ['a refused-then-healthy core (5xx, network down) keeps the lease', ['503', 'down', 'ok'], null],
    ['a renewal that hangs past its 25 s deadline is retried, not the end', ['hang', 'ok'], null],
    ['a signed-out session ends it', ['401'], 40000],
]) {
    test(`route lease: ${name}`, async () => {
        const dom = installCardDom();
        const time = clock();
        const net = renewals(script);
        let lost = null;
        const reopened = [];
        try {
            const release = maintainWidgetAssetLease({ ...LEASE }, () => { lost = time.now; },
                { reestablish: async () => { reopened.push(time.now); return { ...LEASE }; } });
            await time.advance(40000 + 25000 + 60000);
            assert.equal(lost, lostAt, net.calls.join(' '));
            assert.deepEqual(reopened, []);
            release();
        } finally { net.restore(); time.restore(); dom.restore(); }
    });
}

test('route lease: a lease the core no longer knows (restart) is re-established under the same URL', async () => {
    const dom = installCardDom();
    const time = clock();
    const net = renewals(['409', 'ok']);
    let lost = false;
    const reopened = [];
    try {
        const release = maintainWidgetAssetLease({ ...LEASE }, () => { lost = true; },
            { reestablish: async () => { reopened.push(time.now); return reopened.length > 1 ? { ...LEASE } : null; } });
        await time.advance(40000);
        assert.deepEqual(reopened, [40000], 'renewal 409 asks to re-establish');
        await time.advance(2000);              // not live yet (null): retried after the backoff
        assert.equal(reopened.length, 1);
        await time.advance(40000 * 3);
        assert.equal(lost, false);
        release();
    } finally { net.restore(); time.restore(); dom.restore(); }
});

test('route lease: an ended lease (410 on re-establishing) ends the mount; a file lease ends on any failure', async () => {
    const dom = installCardDom();
    const time = clock();
    const net = renewals(['409', 'down']);
    const ended = [];
    try {
        const definitive = Object.assign(new Error('ended'), { definitive: true });
        const route = maintainWidgetAssetLease({ ...LEASE }, () => ended.push(['route', time.now]),
            { reestablish: async () => { throw definitive; } });
        const files = maintainWidgetAssetLease({ ...LEASE }, () => ended.push(['files', time.now]));
        await time.advance(41000);
        assert.deepEqual(ended, [['route', 40000], ['files', 40000]]);
        route();
        files();
    } finally { net.restore(); time.restore(); dom.restore(); }
});
