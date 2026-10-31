import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { deferred } from './helpers/voice_env.js';

async function fixture() {
    const requests = [], states = [], controller = new AbortController();
    const win = new EventTarget(), doc = new EventTarget();
    doc.hidden = false;
    const previous = { window: globalThis.window, document: globalThis.document };
    Object.assign(globalThis, { window: win, document: doc });
    let provider, destroyed = 0, unregistered = 0;
    globalThis.__callConfig = {
        fetchJson() { const request = deferred(); requests.push(request); return request.promise; },
        mountCall: () => ({ state: value => states.push(value), destroy: () => destroyed++ }),
    };
    const source = readFileSync(new URL('../../skills/live_conversation/ui/index.js', import.meta.url), 'utf8')
        .replace(/import .*mountCall.*;\n/, 'const { mountCall } = globalThis.__callConfig;\n')
        .replace(/import .*mountCallSettings.*;\n/, 'const mountCallSettings = () => ({dispose() {}});\n')
        .replace(/import .*fetchJson.*;\n/, 'const { fetchJson } = globalThis.__callConfig;\n');
    const { mount } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}#${Math.random()}`);
    const handle = mount({ skill: 'live_conversation', signal: controller.signal, providers: {
        register(kind, value) { assert.equal(kind, 'conversation'); provider = value; return () => unregistered++; },
    } });
    provider.mountRoom({});
    return { requests, states, win, doc, controller, provider,
        cleanup() { handle.dispose(); Object.assign(globalThis, previous); delete globalThis.__callConfig; },
        counts: () => ({ destroyed, unregistered }) };
}
const settled = () => new Promise(resolve => setImmediate(resolve));

test('failed config reads keep the last good config; focus, visible return and Settings saves reread it', async () => {
    const f = await fixture();
    try {
        assert.equal(f.requests.length, 1, 'mount reads configuration');
        f.requests[0].resolve({ enabled: true, provider: 'test-engine' });
        await settled();
        assert.equal(f.states.at(-1).enabled, true);
        for (const trigger of ['focus', 'visibilitychange', 'ouro:settings-updated']) {
            const before = f.requests.length;
            (trigger === 'visibilitychange' ? f.doc : f.win).dispatchEvent(new Event(trigger));
            assert.equal(f.requests.length, before + 1, trigger);
            f.requests.at(-1).reject(new Error('temporary read failure'));
            await settled();
            assert.equal(f.states.at(-1).enabled, true, trigger);
        }
        const before = f.requests.length;
        f.doc.hidden = true; f.doc.dispatchEvent(new Event('visibilitychange'));
        assert.equal(f.requests.length, before, 'hiding does not read or disable calls');
        f.win.dispatchEvent(new Event('ouro:settings-updated'));
        f.requests.at(-1).resolve({ enabled: false });
        await settled();
        assert.equal(f.states.at(-1).enabled, false, 'only a successful off read disables calls');
    } finally { f.cleanup(); }
});

test('a later read wins and abort removes listeners, rooms and late refresh effects', async () => {
    const f = await fixture();
    try {
        f.win.dispatchEvent(new Event('ouro:settings-updated'));
        assert.equal(f.requests.length, 2);
        f.requests[1].resolve({ enabled: true }); await settled();
        f.requests[0].resolve({ enabled: false }); await settled();
        assert.equal(f.states.at(-1).enabled, true, 'stale off cannot end a newly enabled call');
        f.win.dispatchEvent(new Event('focus'));
        const count = f.states.length, reads = f.requests.length;
        f.controller.abort();
        f.requests.at(-1).resolve({ enabled: false }); await settled();
        for (const name of ['focus', 'ouro:settings-updated']) f.win.dispatchEvent(new Event(name));
        f.doc.dispatchEvent(new Event('visibilitychange'));
        assert.equal(f.requests.length, reads);
        assert.equal(f.states.length, count);
        assert.deepEqual(f.counts(), { destroyed: 1, unregistered: 1 });
    } finally { f.cleanup(); }
});
