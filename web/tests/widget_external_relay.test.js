import test from 'node:test';
import assert from 'node:assert/strict';

import { mountModuleWidget } from '../modules/widget_module.js';
import { joinFrame } from './helpers/fake_channel.js';

// Exercise the production parent mount and host opener. This DOM double does
// not establish browser user activation; click/Enter/touch need browser proof.
async function relayHarness(t, api = null, initialTheme = '') {
    const opened = [];
    const replies = [];
    const listeners = new Map();
    const attributes = new Map();
    const iframe = {
        isConnected: true,
        dataset: {},
        style: { setProperty() {} },
        setAttribute(name, value) { attributes.set(name, value); },
        contentWindow: { postMessage(message) { replies.push(message); } },  // never used: replies take the channel
        addEventListener() {},
        remove() { this.isConnected = false; this.parentNode = null; },
    };
    const mount = {
        replaceChildren(node) { node.parentNode = this; },
    };
    const win = {
        location: { origin: 'https://ouroboros.test' },
        navigator: { userActivation: { isActive: true } },
        open(...args) { opened.push(args); return null; },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (listeners.get(type) === listener) listeners.delete(type);
        },
        ...(api ? { pywebview: { api } } : {}),
    };
    t.mock.method(globalThis, 'fetch', async (url) => (url === '/api/widgets/asset-leases'
        ? Response.json({ token: 'route-lease', base_url: '/api/widget-resources/route-lease/', expires_in: 120, renew_after: 40 }) : new Response('// reviewed module')));
    const globals = new Map();
    let dispose;
    let send;
    t.after(async () => {
        try {
            if (dispose) {
                const stopped = dispose();
                send({ type: 'ouro-widget-disposed' });
                await stopped;
            }
        } finally {
            for (const [name, previous] of globals) {
                if (previous) Object.defineProperty(globalThis, name, previous);
                else delete globalThis[name];
            }
        }
    });
    for (const [name, value] of Object.entries({
        window: win,
        document: { createElement() { return iframe; }, documentElement: { dataset: initialTheme ? { theme: initialTheme } : {} } },
    })) {
        globals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
        Object.defineProperty(globalThis, name, { configurable: true, value });
    }
    dispose = await mountModuleWidget(mount, { skill: 'links' }, { entry: 'widget.js', height: 320 });
    const onMessage = listeners.get('message');
    const frame = joinFrame(iframe, onMessage);
    frame.channel.port1.onmessage = (event) => { if (event.data?.type !== 'ouro-widget-ready') replies.push(event.data); };
    const { nonce } = frame;
    send = (data = {}, source = iframe.contentWindow) => {
        const message = { type: 'ouro-widget-open-external', id: 1, nonce, url: 'https://example.test/target', ...data };
        if (source === iframe.contentWindow) frame.channel.port1.postMessage(message);
        else onMessage({ source, data: message });
    };
    const flush = () => new Promise((resolve) => setImmediate(resolve));
    return { win, iframe, attributes, opened, replies, listeners, send, dispose, flush };
}

test('module parent opens synchronously, acknowledges noopener null and keeps the sandbox', async (t) => {
    const h = await relayHarness(t);
    h.send();
    assert.deepEqual(h.opened, [['https://example.test/target', '_blank', 'noopener']]);
    await h.flush();
    assert.deepEqual(h.replies.at(-1).result, { ok: true, native: false, host: 'browser' });
    assert.equal(h.attributes.get('sandbox'), 'allow-scripts allow-pointer-lock allow-downloads');
    assert.equal(h.attributes.get('allow'), 'autoplay; fullscreen; clipboard-write');
});

test('module parent rejects foreign frames, stale nonces, inactive requests and unsafe URLs', async (t) => {
    const h = await relayHarness(t);
    h.send({}, {});
    h.send({ nonce: 'stale-nonce' });
    assert.deepEqual(h.replies, []);
    h.win.navigator.userActivation.isActive = false;
    h.send();
    await h.flush();
    assert.match(h.replies.at(-1).result.error, /requires a user action/);
    h.win.navigator.userActivation.isActive = true;
    h.send({ url: 'javascript:alert(1)' });
    await h.flush();
    assert.match(h.replies.at(-1).result.error, /Unsupported external link/);
    h.iframe.isConnected = false;
    h.send();
    await h.flush();
    assert.match(h.replies.at(-1).result.error, /disposed/);
    assert.deepEqual(h.opened, []);
});

test('module parent uses the native bridge and detaches the relay after disposal', async (t) => {
    const native = [];
    const h = await relayHarness(t, { open_external_url(url) { native.push(url); return { ok: true }; } });
    h.send();
    assert.deepEqual(native, ['https://example.test/target']);
    await h.flush();
    assert.deepEqual(h.replies.at(-1).result, { ok: true, native: true });
    assert.deepEqual(h.opened, []);
    const stopped = h.dispose();
    h.send();
    await h.flush();
    assert.match(h.replies.at(-1).result.error, /disposed/);
    h.send({ type: 'ouro-widget-disposed' });
    await stopped;
    const replyCount = h.replies.length;
    h.send();
    await h.flush();
    assert.equal(h.replies.length, replyCount);
    assert.equal(h.listeners.has('message'), false);
    assert.equal(h.iframe.isConnected, false);
    assert.equal(native.length, 1);
});

test('module parent embeds the resolved Light theme in the initial bridge script', async (t) => {
    const h = await relayHarness(t, null, 'light');
    assert.match(h.iframe.srcdoc, /"initialTheme":"light"/);
});

test('module parent releases a live theme relay when disposed', async (t) => {
    const h = await relayHarness(t);
    h.send({ type: 'ouro-widget-theme', op: 'subscribe' });
    assert.equal(h.listeners.has('ouro:theme-changed'), true);
    const stopped = h.dispose();
    h.send({ type: 'ouro-widget-disposed' });
    await stopped;
    assert.equal(h.listeners.has('ouro:theme-changed'), false);
});

test('module parent forwards the resolved theme only after child opt-in and releases it on dispose', async (t) => {
    const h = await relayHarness(t);
    document.documentElement.dataset.theme = 'light';
    h.send({ type: 'ouro-widget-theme', op: 'subscribe' });
    assert.equal(h.replies.at(-1).theme, 'light');
    h.win.ouroTheme = { theme: 'dark' };
    h.listeners.get('ouro:theme-changed')?.();
    assert.equal(h.replies.at(-1).theme, 'dark');
    const replyCount = h.replies.length;
    h.send({ type: 'ouro-widget-theme', op: 'unsubscribe' });
    assert.equal(h.listeners.has('ouro:theme-changed'), false);
    h.win.ouroTheme = { theme: 'light' };
    h.listeners.get('ouro:theme-changed')?.();
    assert.equal(h.replies.length, replyCount, 'unsubscribe itself is not a reply');
    const stopped = h.dispose();
    h.send({ type: 'ouro-widget-theme', op: 'subscribe' });
    h.send({ type: 'ouro-widget-disposed' });
    await stopped;
    assert.equal(h.listeners.has('ouro:theme-changed'), false);
});

test('module parent refreshes a repeated child subscription without duplicating its host listener', async (t) => {
    const h = await relayHarness(t);
    document.documentElement.dataset.theme = 'light';
    h.send({ type: 'ouro-widget-theme', op: 'subscribe' });
    const firstReplyCount = h.replies.length;
    h.win.ouroTheme = { theme: 'dark' };
    h.send({ type: 'ouro-widget-theme', op: 'subscribe' });
    assert.equal(h.replies.length, firstReplyCount + 1);
    assert.equal(h.replies.at(-1).theme, 'dark');
    assert.equal(h.listeners.get('ouro:theme-changed') !== undefined, true);
});
