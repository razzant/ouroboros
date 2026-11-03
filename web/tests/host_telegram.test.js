import test from 'node:test';
import assert from 'node:assert/strict';
import { createTelegramHost, hostCall, isTelegramHost, KEYBOARD_COLLAPSE_PX } from '../modules/host_telegram.js';

// The Telegram adapter (P3) against a fake SDK: present at boot, arriving late through the async tag, and an old
// client without the newer methods. Every host call goes through one gate; binding is idempotent; dispose releases.

function fakeSdk({ version = '8.0', scheme = 'light', methods = null } = {}) {
    const calls = [];
    const events = new Map();
    const all = ['ready', 'expand', 'disableVerticalSwipes', 'setHeaderColor', 'setBackgroundColor', 'setBottomBarColor', 'hideKeyboard'];
    const app = {
        version, initData: 'query_id=1', colorScheme: scheme,
        safeAreaInset: { top: 0, bottom: 34, left: 0, right: 0 },
        contentSafeAreaInset: { top: 46, bottom: 0, left: 0, right: 0 },
        // Segment-wise, like the real SDK: 7.10 is newer than 7.7.
        isVersionAtLeast: (v) => {
            const a = String(version).split('.').map(Number), b = String(v).split('.').map(Number);
            for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
                if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0);
            }
            return true;
        },
        onEvent(name, fn) { events.set(name, fn); calls.push(['onEvent', name]); },
        offEvent(name) { events.delete(name); calls.push(['offEvent', name]); },
        BackButton: {
            isVisible: false, handler: null,
            show() { this.isVisible = true; calls.push(['BackButton.show']); },
            hide() { this.isVisible = false; calls.push(['BackButton.hide']); },
            onClick(fn) { this.handler = fn; calls.push(['BackButton.onClick']); },
            offClick() { this.handler = null; calls.push(['BackButton.offClick']); },
        },
    };
    for (const name of methods || all) app[name] = (...args) => calls.push([name, ...args]);
    return { app, calls, events, fire: (name, detail) => events.get(name)?.(detail) };
}

function fakeWorld({ sdk = null, host = true, color = '#ffffff' } = {}) {
    const rootStyle = new Map();
    const timers = [];
    const listeners = new Map();
    const script = { handlers: new Map(), addEventListener(t, fn) { this.handlers.set(t, fn); }, removeEventListener(t) { this.handlers.delete(t); } };
    const doc = {
        documentElement: { dataset: host ? { ouroborosHost: 'telegram' } : {}, style: { setProperty: (k, v) => rootStyle.set(k, v) } },
        body: {}, activeElement: null,
        querySelector: (sel) => (sel.includes('telegram-web-app') ? script : null),
    };
    doc.activeElement = doc.body;
    const win = {
        Telegram: sdk ? { WebApp: sdk.app } : undefined,
        innerHeight: 800, visualViewport: { height: 800, listeners: new Map(), addEventListener(t, fn) { this.listeners.set(t, fn); }, removeEventListener(t) { this.listeners.delete(t); } },
        addEventListener(t, fn) { listeners.set(t, fn); }, removeEventListener(t) { listeners.delete(t); },
        setTimeout(fn, ms) { timers.push({ fn, ms }); return timers.length; }, clearTimeout() { timers.length = 0; },
        getComputedStyle: () => ({ getPropertyValue: () => color }),
    };
    const stack = { count: 0, popped: [], listeners: new Set(), size() { return this.count; },
        onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
        pop(reason) { this.popped.push(reason); this.count = Math.max(0, this.count - 1); this.emit(); },
        back(reason) { this.pop(reason); },
        emit() { this.listeners.forEach((fn) => fn({ size: this.count })); } };
    const theme = { source: null, refreshed: 0, setSystemSource(fn) { this.source = fn; return () => { this.source = null; }; }, refreshSystem() { this.refreshed += 1; } };
    return { doc, win, rootStyle, timers, listeners, script, stack, theme };
}

test('outside Telegram the adapter is a no-op; the host stamp or initData makes it active', () => {
    assert.equal(isTelegramHost({ win: {}, doc: { documentElement: { dataset: {} } } }), false);
    assert.equal(isTelegramHost({ win: { Telegram: { WebApp: { initData: '' } } }, doc: { documentElement: { dataset: {} } } }), false);
    assert.equal(isTelegramHost({ win: { Telegram: { WebApp: { initData: 'x' } } }, doc: { documentElement: { dataset: {} } } }), true);
    assert.equal(isTelegramHost({ win: {}, doc: { documentElement: { dataset: { ouroborosHost: 'telegram' } } } }), true);
    const sdk = fakeSdk();
    const w = fakeWorld({ sdk, host: false });
    sdk.app.initData = '';
    const adapter = createTelegramHost({ win: w.win, doc: w.doc, layers: w.stack, theme: w.theme });
    assert.equal(adapter.bind(), false);
    assert.equal(sdk.calls.length, 0, 'a browser tab never calls the SDK');
    adapter.dispose();
});

test('the SDK present at boot: ready, expand, swipes off, chrome painted, insets mirrored, Back follows the stack', () => {
    const sdk = fakeSdk({ version: '7.7' });
    const w = fakeWorld({ sdk, color: 'rgb(13, 11, 15)' });
    const adapter = createTelegramHost({ win: w.win, doc: w.doc, layers: w.stack, theme: w.theme });
    assert.equal(adapter.bind(), true);
    assert.equal(adapter.bind(), true, 'binding twice binds once');
    const names = sdk.calls.map((c) => c[0]);
    assert.deepEqual(names.slice(0, 3), ['ready', 'expand', 'disableVerticalSwipes']);
    assert.equal(names.filter((n) => n === 'ready').length, 1);
    assert.deepEqual(sdk.calls.find((c) => c[0] === 'setHeaderColor'), ['setHeaderColor', '#0d0b0f']);
    assert.ok(sdk.calls.find((c) => c[0] === 'setBackgroundColor'));
    assert.equal(sdk.calls.some((c) => c[0] === 'setBottomBarColor'), false, '7.10 is newer than this client');
    assert.equal(w.rootStyle.get('--tg-safe-area-inset-bottom'), '34px');
    assert.equal(w.rootStyle.get('--tg-content-safe-area-inset-top'), '46px');
    assert.equal(w.rootStyle.has('--tg-safe-area-inset-left'), false, 'the shell reads vertical insets only');
    // The insets move: the tokens follow.
    sdk.app.safeAreaInset.bottom = 0;
    sdk.fire('safeAreaChanged');
    assert.equal(w.rootStyle.get('--tg-safe-area-inset-bottom'), '0px');
    // Back is hidden with an empty stack, shown while a layer is open, and its click is the stack's Back.
    assert.equal(sdk.app.BackButton.isVisible, false);
    w.stack.count = 2; w.stack.emit();
    assert.equal(sdk.app.BackButton.isVisible, true);
    sdk.app.BackButton.handler();
    assert.deepEqual(w.stack.popped, ['host-back']);
    assert.equal(sdk.app.BackButton.isVisible, true, 'one layer left');
    w.stack.pop('scrim');
    assert.equal(sdk.app.BackButton.isVisible, false);
    // Telegram's scheme is the System source; its change re-settles the theme and repaints the chrome.
    assert.equal(w.theme.source(), 'light');
    sdk.app.colorScheme = 'dark';
    const before = sdk.calls.length;
    sdk.fire('themeChanged');
    assert.equal(w.theme.refreshed, 1);
    assert.equal(w.theme.source(), 'dark');
    assert.ok(sdk.calls.slice(before).some((c) => c[0] === 'setHeaderColor'));
    // The page's own theme change (Light/Dark picked) repaints the chrome too.
    w.listeners.get('ouro:theme-changed')();
    assert.ok(sdk.calls.length > before + 2);
    adapter.dispose();
    assert.equal(w.theme.source, null, 'dispose releases the System source');
    assert.equal(sdk.app.BackButton.handler, null);
    assert.equal(sdk.app.BackButton.isVisible, false);
    assert.equal(sdk.events.size, 0, 'every SDK event is released');
    assert.equal(w.listeners.size, 0);
    assert.equal(w.win.visualViewport.listeners.size, 0);
    adapter.dispose();
});

test('an SDK arriving late binds on the tag load or the poll, once', () => {
    const sdk = fakeSdk();
    const w = fakeWorld({ sdk: null });
    const adapter = createTelegramHost({ win: w.win, doc: w.doc, layers: w.stack, theme: w.theme });
    assert.equal(adapter.bind(), false, 'not yet: the async tag is still loading');
    assert.equal(adapter.bound(), false);
    assert.equal(w.timers.length, 1, 'one poll armed');
    // The global appears before the load event fires (loaded first): the poll catches it.
    w.win.Telegram = { WebApp: sdk.app };
    w.timers[0].fn();
    assert.equal(adapter.bound(), true);
    assert.equal(sdk.calls.filter((c) => c[0] === 'ready').length, 1);
    w.script.handlers.get('load')?.();
    assert.equal(sdk.calls.filter((c) => c[0] === 'ready').length, 1, 'the load event binds nothing twice');
    adapter.dispose();
    // The other order: the load event first.
    const sdk2 = fakeSdk();
    const w2 = fakeWorld({ sdk: null });
    const adapter2 = createTelegramHost({ win: w2.win, doc: w2.doc, layers: w2.stack, theme: w2.theme });
    adapter2.bind();
    w2.win.Telegram = { WebApp: sdk2.app };
    w2.script.handlers.get('load')();
    assert.equal(adapter2.bound(), true);
    adapter2.dispose();
    // Disposed while waiting: the poll is cancelled and nothing binds later.
    const w3 = fakeWorld({ sdk: null });
    const adapter3 = createTelegramHost({ win: w3.win, doc: w3.doc, layers: w3.stack, theme: w3.theme });
    adapter3.bind();
    adapter3.dispose();
    assert.equal(w3.timers.length, 0);
    w3.win.Telegram = { WebApp: fakeSdk().app };
    assert.equal(adapter3.bind(), false);
});

test('an old client without the newer methods gets only what it has, and never a throw', () => {
    const sdk = fakeSdk({ version: '6.0', methods: ['ready', 'expand'] });
    delete sdk.app.BackButton;
    const w = fakeWorld({ sdk });
    const adapter = createTelegramHost({ win: w.win, doc: w.doc, layers: w.stack, theme: w.theme });
    assert.equal(adapter.bind(), true);
    assert.deepEqual(sdk.calls.filter((c) => !String(c[0]).includes('Event')).map((c) => c[0]), ['ready', 'expand']);
    w.stack.count = 1; w.stack.emit();  // no BackButton: nothing to show, nothing thrown
    adapter.dispose();
    // The gate itself: presence, version, and a throwing method.
    const target = { isVersionAtLeast: (v) => Number(v) <= 7, works: () => true, breaks: () => { throw new Error('x'); } };
    assert.equal(hostCall(target, 'missing'), false);
    assert.equal(hostCall(target, 'works', '7.0'), true);
    assert.equal(hostCall(target, 'works', '7.7'), false);
    assert.equal(hostCall(target, 'breaks'), false);
});

test('a collapsed visual viewport with a focused field blurs it and asks the host to hide the keyboard', () => {
    const sdk = fakeSdk({ version: '9.1' });
    const w = fakeWorld({ sdk });
    const adapter = createTelegramHost({ win: w.win, doc: w.doc, layers: w.stack, theme: w.theme });
    adapter.bind();
    let blurred = 0;
    const field = { matches: () => true, blur: () => { blurred += 1; w.doc.activeElement = w.doc.body; } };
    w.doc.activeElement = field;
    w.win.visualViewport.height = KEYBOARD_COLLAPSE_PX + 200;
    w.win.visualViewport.listeners.get('resize')();
    assert.equal(blurred, 0, 'a normal viewport is left alone');
    w.win.visualViewport.height = 1;
    sdk.fire('viewportChanged', { isStateStable: false });
    assert.equal(blurred, 1);
    assert.ok(sdk.calls.some((c) => c[0] === 'hideKeyboard'));
    // Nothing focused: no recovery, however small the viewport.
    w.win.visualViewport.listeners.get('resize')();
    assert.equal(blurred, 1);
    adapter.dispose();
});
