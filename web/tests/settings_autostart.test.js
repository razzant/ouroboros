import assert from 'node:assert/strict';
import test from 'node:test';

import { apiClient } from '../modules/api_client.js';
import { bindAutostartControl } from '../modules/settings_autostart.js';

function fakeWindow() {
    const listeners = new Map();
    return {
        addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type, fn) => { if (listeners.get(type) === fn) listeners.delete(type); },
        fire: (type, event) => listeners.get(type)?.(event),
        has: (type) => listeners.has(type),
    };
}

function fakeBlock() {
    let onChange = null;
    const box = {
        checked: false,
        disabled: false,
        addEventListener: (type, fn) => { if (type === 'change') onChange = fn; },
        removeEventListener: (type, fn) => { if (onChange === fn) onChange = null; },
    };
    const status = { textContent: '', dataset: {} };
    const section = {
        hidden: true,
        querySelector: (selector) => ({ '[data-autostart-toggle]': box, '[data-autostart-status]': status })[selector] || null,
    };
    const page = { querySelector: (selector) => (selector === '[data-autostart-settings]' ? section : null) };
    const click = (checked) => { box.checked = checked; return onChange?.(); };
    return { page, section, box, status, click, bound: () => onChange !== null };
}

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    return { promise, resolve, reject };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

function withApi(t, { read, write }) {
    const original = { read: apiClient.desktopAutostart, write: apiClient.setDesktopAutostart };
    apiClient.desktopAutostart = read;
    apiClient.setDesktopAutostart = write || (async () => { throw new Error('unexpected write'); });
    const win = fakeWindow();
    const previousWindow = globalThis.window;
    globalThis.window = win;
    t.after(() => {
        apiClient.desktopAutostart = original.read;
        apiClient.setDesktopAutostart = original.write;
        globalThis.window = previousWindow;
    });
    return win;
}

test('unknown states stay hidden; unavailable hosts show a disabled explanation', async (t) => {
    for (const state of ['something_newer', undefined]) {
        withApi(t, { read: async () => ({ state }) });
        const block = fakeBlock();
        bindAutostartControl(block.page);
        await settle();
        assert.equal(block.section.hidden, true, `state ${state} keeps the block hidden`);
        assert.equal(block.box.checked, false);
    }
});

test('the OS state paints the toggle and explains entries it does not own', async (t) => {
    let state = 'on';
    const win = withApi(t, { read: async () => ({ state }) });
    const block = fakeBlock();
    bindAutostartControl(block.page);
    await settle();
    assert.equal(block.section.hidden, false);
    assert.equal(block.box.checked, true);
    assert.equal(block.status.textContent, '');

    state = 'disabled_by_os'; // switched off in Windows Startup apps meanwhile
    win.fire('ouro:page-shown', { detail: { page: 'settings' } });
    await settle();
    assert.equal(block.box.checked, false);
    assert.match(block.status.textContent, /Turned off in the host operating system/);
    assert.equal(block.status.dataset.tone, 'warn');

    state = 'other_copy';
    win.fire('ouro:page-shown', { detail: { page: 'chat' } });
    await settle();
    assert.match(block.status.textContent, /host operating system/, 'another page does not trigger a read');
    win.fire('ouro:page-shown', { detail: { page: 'settings' } });
    await settle();
    assert.match(block.status.textContent, /from a different entry \(another copy/);
});

test('a click applies at once and a refused change shows what the OS now holds', async (t) => {
    const writes = [];
    let server = 'off';
    let refuse = false;
    let partial = false;
    withApi(t, {
        read: async () => ({ state: server }),
        write: async (enabled) => {
            writes.push(enabled);
            if (partial) server = enabled ? 'on' : 'off'; // the Run value landed, the switch clear did not
            if (refuse || partial) throw new Error('access denied');
            server = enabled ? 'on' : 'off';
            return { state: server };
        },
    });
    const block = fakeBlock();
    bindAutostartControl(block.page);
    await settle();
    await block.click(true);
    assert.deepEqual(writes, [true]);
    assert.equal(block.box.checked, true);
    assert.equal(block.box.disabled, false);

    refuse = true;
    await block.click(false);
    assert.deepEqual(writes, [true, false]);
    assert.equal(block.box.checked, true, 'the entry is still there, so the toggle says so');
    assert.match(block.status.textContent, /Could not change the host startup entry: access denied/);
    assert.equal(block.status.dataset.tone, 'danger');

    refuse = false;
    partial = true;
    await block.click(false);
    assert.equal(block.box.checked, false, 'a refusal after the first write is read back, not guessed');
    assert.equal(block.box.disabled, false);
    assert.equal(block.status.dataset.tone, 'danger');
});

test('a read that started before a click cannot repaint over the owner choice', async (t) => {
    const slowRead = deferred();
    withApi(t, {
        read: () => slowRead.promise,
        write: async () => ({ state: 'on' }),
    });
    const block = fakeBlock();
    block.section.hidden = false;
    bindAutostartControl(block.page);
    await block.click(true);
    slowRead.resolve({ state: 'off' });
    await settle();
    assert.equal(block.box.checked, true);
});

test('a failed write and failed read never assert a guessed host state', async (t) => {
    let unreadable = false;
    const win = withApi(t, {
        read: async () => {
            if (unreadable) throw new Error('read failed');
            return { state: 'off' };
        },
        write: async () => { unreadable = true; throw new Error('write failed'); },
    });
    const block = fakeBlock();
    bindAutostartControl(block.page);
    await settle();
    await block.click(true);
    assert.equal(block.box.disabled, true);
    assert.match(block.status.textContent, /Current host state could not be read/);
    assert.equal(block.status.dataset.tone, 'danger');
    unreadable = false;
    win.fire('ouro:page-shown', { detail: { page: 'settings' } });
    await settle();
    assert.equal(block.box.checked, false);
    assert.equal(block.box.disabled, false);
});

test('a failed first read shows the block with the error and retries when Settings opens', async (t) => {
    let failing = true;
    const win = withApi(t, {
        read: async () => {
            if (failing) throw new Error('systemctl did not answer');
            return { state: 'off' };
        },
    });
    const block = fakeBlock();
    bindAutostartControl(block.page);
    await settle();
    assert.equal(block.section.hidden, false, 'a lasting read error is explained, not hidden');
    assert.equal(block.box.disabled, true);
    assert.equal(block.status.textContent, 'Could not read the host startup entry: systemctl did not answer');
    assert.equal(block.status.dataset.tone, 'danger');
    failing = false;
    win.fire('ouro:page-shown', { detail: { page: 'settings' } });
    await settle();
    assert.equal(block.box.disabled, false);
    assert.equal(block.status.textContent, '');
});

test('pagehide releases every listener the block took', async (t) => {
    const win = withApi(t, { read: async () => ({ state: 'off' }) });
    const block = fakeBlock();
    bindAutostartControl(block.page);
    await settle();
    win.fire('pagehide', { persisted: true });
    assert.equal(block.bound(), true, 'a bfcache hide keeps the page alive');
    win.fire('pagehide', { persisted: false });
    assert.equal(block.bound(), false);
    assert.equal(win.has('ouro:page-shown'), false);
    assert.equal(win.has('pagehide'), false);
});

for (const reason of ['Sign-in startup needs a newer app build (7.2.0 or later).', 'Move Ouroboros to Applications before enabling sign-in startup.']) {
    test(reason, async (t) => {
        withApi(t, { read: async () => ({ state: 'unavailable', reason }) });
        const block = fakeBlock();
        bindAutostartControl(block.page);
        await settle();
        assert.equal(block.section.hidden, false);
        assert.equal(block.box.disabled, true);
        assert.equal(block.status.textContent, reason);
    });
}

function fakeSection() {
    const handlers = new Map();
    const control = (name) => ({
        checked: false,
        disabled: false,
        addEventListener: (type, fn) => { if (type === 'change') handlers.set(name, fn); },
        removeEventListener: (type, fn) => { if (handlers.get(name) === fn) handlers.delete(name); },
    });
    const startup = control('startup');
    const background = control('background');
    const startupStatus = { textContent: '', dataset: {} };
    const backgroundStatus = { textContent: '', dataset: {} };
    const row = { hidden: true };
    const section = {
        hidden: true,
        querySelector: (selector) => ({
            '[data-autostart-toggle]': startup, '[data-autostart-status]': startupStatus,
            '[data-background-row]': row, '[data-background-toggle]': background, '[data-background-status]': backgroundStatus,
        })[selector] || null,
    };
    const page = { querySelector: (selector) => (selector === '[data-autostart-settings]' ? section : null) };
    const click = (box, name, checked) => { box.checked = checked; return handlers.get(name)?.(); };
    return { page, section, row, startup, background, backgroundStatus, handlers, click };
}

function withBothApis(t, { background, setBackground }) {
    const saved = { read: apiClient.desktopBackground, write: apiClient.setDesktopBackground };
    apiClient.desktopBackground = background;
    apiClient.setDesktopBackground = setBackground || (async () => { throw new Error('unexpected write'); });
    t.after(() => {
        apiClient.desktopBackground = saved.read;
        apiClient.setDesktopBackground = saved.write;
    });
    return withApi(t, { read: async () => ({ state: 'off' }) });
}

test('the keep-running control applies at once and reads back what the host holds', async (t) => {
    let server = 'off';
    let refuse = false;
    const writes = [];
    withBothApis(t, {
        background: async () => ({ state: server }),
        setBackground: async (enabled) => {
            writes.push(enabled);
            if (refuse) throw new Error('settings locked');
            server = enabled ? 'on' : 'off';
            return { state: server };
        },
    });
    const block = fakeSection();
    bindAutostartControl(block.page);
    await settle();
    assert.equal(block.row.hidden, false);
    assert.equal(block.background.checked, false);
    await block.click(block.background, 'background', true);
    assert.deepEqual(writes, [true]);
    assert.equal(block.background.checked, true);
    refuse = true;
    await block.click(block.background, 'background', false);
    assert.equal(block.background.checked, true, 'a refused change shows the choice the host still holds');
    assert.match(block.backgroundStatus.textContent, /Could not change the background setting: settings locked/);
});

test('hosts without background mode show the reason on a disabled control', async (t) => {
    const reason = 'Not available on Linux yet: closing the window quits Ouroboros.';
    withBothApis(t, { background: async () => ({ state: 'unavailable', reason }) });
    const block = fakeSection();
    bindAutostartControl(block.page);
    await settle();
    assert.equal(block.row.hidden, false);
    assert.equal(block.background.disabled, true);
    assert.equal(block.backgroundStatus.textContent, reason);
});

test('one Settings visit re-reads both host controls and one pagehide releases both', async (t) => {
    let reads = 0;
    let server = 'off';
    const win = withBothApis(t, { background: async () => { reads += 1; return { state: server }; } });
    const block = fakeSection();
    bindAutostartControl(block.page);
    await settle();
    server = 'on'; // the desktop's first-close answer landed meanwhile
    win.fire('ouro:page-shown', { detail: { page: 'settings' } });
    await settle();
    assert.equal(reads, 2);
    assert.equal(block.background.checked, true);
    win.fire('pagehide', { persisted: false });
    assert.equal(block.handlers.size, 0);
    assert.equal(win.has('ouro:page-shown'), false);
});
