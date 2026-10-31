/* The send-time surface fact carries the owner's Interface size only as the desktop window took it
   (interface_size.js): a size asked and not answered yet, refused, or whose call came back without a size the
   window took, is no fact of the window's and is sent as absence; nor does a read older than a size the window
   took bring an older size back. client_surface.js reads the page's one controller
   (`getInterfaceSize()`), which is built here, in this file's own process, around a recording desktop bridge. */
import test from 'node:test';
import assert from 'node:assert/strict';

const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((done) => setImmediate(done)); };

// As launcher_page_zoom.PageZoom does, the window answers each call when it is made; 'hold' delays only the delivery.
const bridge = { percent: 100, next: 'answer', holdReads: false, held: [], sets: [] };
const listeners = new Map();
globalThis.window = {
    navigator: { platform: 'MacIntel', userAgent: '' },
    performance: { timeOrigin: 1000 },
    innerWidth: 1280,
    innerHeight: 860,
    matchMedia: () => ({ matches: false }),
    addEventListener: (type, fn) => listeners.set(type, fn),
    removeEventListener: (type) => listeners.delete(type),
    pywebview: { api: {
        shell_info: () => {
            if (bridge.next === 'fail') return Promise.reject(new Error('controlled shell_info failure'));
            const answer = { interface_size: { supported: true, min: 50, max: 200, step: 5, percent: bridge.percent } };
            if (bridge.holdReads) return new Promise((resolve) => { bridge.held.push(() => resolve(answer)); });
            return Promise.resolve(answer);
        },
        set_interface_size: (percent) => {
            bridge.sets.push(percent);
            if (bridge.next === 'fail') return Promise.reject(new Error('controlled set_interface_size failure'));
            bridge.percent = percent;  // stored, as PageZoom.request stores it before scheduling the window's write
            // 'refuse': PageZoom's 'unavailable', the window's own write could not be scheduled.
            const answer = bridge.next === 'refuse' ? { ok: false, state: 'unavailable', percent }
                : { ok: true, state: 'scheduled', percent };
            if (bridge.next === 'hold') return new Promise((resolve) => { bridge.held.push(() => resolve(answer)); });
            return Promise.resolve(answer);
        },
    } },
};
globalThis.window.top = globalThis.window;
globalThis.document = { documentElement: { dataset: { layout: 'docked' } }, addEventListener() {}, removeEventListener() {} };
if (typeof navigator === 'undefined') globalThis.navigator = { userAgent: 'node-test/1.0' };

const { getInterfaceSize } = await import('../modules/interface_size.js');
const { clientSurfaceSnapshot } = await import('../modules/client_surface.js');

test('the surface fact carries the size the window reported, never one only asked of it', async () => {
    const size = getInterfaceSize();
    await settle();
    assert.equal(clientSurfaceSnapshot().interface_size, 100, 'the window\'s answer is a send-time fact');

    bridge.next = 'hold';
    const first = size.step(1);
    assert.equal(size.state().percent, 110, 'the control shows the choice on its way');
    assert.equal(clientSurfaceSnapshot().interface_size, undefined, 'asked, not reported yet: sent as absence');
    bridge.held.shift()();
    await first;
    assert.equal(clientSurfaceSnapshot().interface_size, 110, 'reported back: a fact again');

    bridge.next = 'fail';  // the call rejects, and the read after it too
    await size.set(150);
    assert.equal(size.state().unconfirmed, true);
    assert.equal(clientSurfaceSnapshot().interface_size, undefined, 'unknown is never sent as the 150 % only asked');

    bridge.next = 'answer';
    await size.reset();
    assert.deepEqual(bridge.sets, [110, 150, 100]);
    assert.equal(clientSurfaceSnapshot().interface_size, 100, 'an answered retry is the window\'s size');
});

test('neither a refusal\'s percent nor a read older than a size the window took is sent as its size', async () => {
    const size = getInterfaceSize();
    await size.set(100);
    assert.equal(clientSurfaceSnapshot().interface_size, 100);

    bridge.next = 'refuse';
    await size.set(120);
    assert.equal(size.state().unconfirmed, true);
    assert.equal(clientSurfaceSnapshot().interface_size, undefined, 'the refused 120 % is no size of the window\'s');
    bridge.next = 'answer';
    await size.set(120);
    assert.equal(clientSurfaceSnapshot().interface_size, 120, 'the retry it took is');

    bridge.holdReads = true;
    listeners.get('pywebviewready')();  // answered now, at 120 %, and delivered late
    bridge.holdReads = false;
    await size.set(130);
    bridge.held.shift()();
    await settle();
    assert.equal(clientSurfaceSnapshot().interface_size, 130, 'the older 120 % does not come back');
});
