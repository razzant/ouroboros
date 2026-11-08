/* The desktop app's own facts: version, storage and system notifications.

   Updated source can host the window, but the installed signed executable still
   determines its app identity and profile. Neither version proves capability. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
    PERSISTENT_STORAGE_SINCE,
    describeShell,
    loadDesktopShell,
    mountDesktopShell,
    parseAppVersion,
    readShellInfo,
    shellCapability,
    shellStorageText,
} from '../modules/desktop_shell.js';

test('a browser tab has no desktop facts to disclose', () => {
    assert.deepEqual(describeShell(), { desktop: false });
    assert.equal(shellStorageText(describeShell({ bridge: false, appVersion: '6.82.0' })), '');
});

test('an app without shell_info is judged by the version its launcher stamped', () => {
    assert.deepEqual(PERSISTENT_STORAGE_SINCE, [7, 2, 0]);
    const old = describeShell({ bridge: true, appVersion: '6.82.0' });
    assert.equal('legacy' in old, false, 'notification support is the notifier\'s check of the method, never a version');
    assert.equal(old.persistentStorage, false, 'pywebview private mode erases website data at every launch');
    assert.equal(old.native, null);
    const text = shellStorageText(old);
    assert.match(text, /This desktop app \(6\.82\.0\) erases/);
    assert.match(text, /Theme and Notifications return to their defaults/);
    // A fact about this window, never a remedy: installing or updating is Dashboard › Updates' to offer.
    assert.doesNotMatch(text, /\binstall|\bupdate|\breinstall/i);

    assert.equal(describeShell({ bridge: true, appVersion: '7.1.9' }).persistentStorage, false);
    assert.equal(describeShell({ bridge: true, appVersion: 'v7.2.0-rc.1' }).persistentStorage, true,
        'the release triple decides, as for sign-in startup');
    assert.equal(describeShell({ bridge: true, appVersion: '7.6.0' }).persistentStorage, true);
    const unknown = describeShell({ bridge: true, appVersion: 'dev' });
    assert.equal(unknown.persistentStorage, null);
    assert.equal(shellStorageText(unknown), '', 'an unknown version claims nothing');
    assert.equal(parseAppVersion('garbage'), null);
});

test('a current app reports itself and its notification permission', () => {
    const shell = describeShell({
        bridge: true,
        info: { shell_version: '7.7.0', persistent_storage: true,
            native_notifications: { available: true, status: 'authorized', platform: 'macos', reason: '' } },
        appVersion: 'ignored',
    });
    assert.deepEqual(shell, {
        desktop: true, version: '7.7.0', persistentStorage: true,
        native: { available: true, status: 'authorized', platform: 'macos', reason: '' },
    });
    assert.equal(shellStorageText(shell), '');
    assert.match(shellStorageText({ ...shell, persistentStorage: false }), /erases/);
});

test('loaded source and installed bundle stay distinct, and an identity loss is disclosed', () => {
    const info = { shell_version: '8.0.0', bundle_version: '7.7.0', persistent_storage: true,
        source_launcher: true, macos_source_entry: true, native_notifications: null };
    const current = describeShell({ bridge: true, info });
    assert.equal(current.version, '7.7.0');
    assert.equal(current.loadedVersion, '8.0.0');
    assert.equal(current.appEntry, true);
    assert.equal(shellStorageText(current), '');
    const plain = describeShell({ bridge: true, info: { ...info, macos_source_entry: false } });
    assert.equal(plain.persistentStorage, true, 'persistence alone does not prove profile continuity');
    assert.match(shellStorageText(plain), /may differ from an ordinary app launch/);
    assert.match(shellStorageText(plain), /older Python profile is left untouched/);
    assert.doesNotMatch(shellStorageText(plain), /\binstall|\bupdate|\breinstall/i, 'no remedy claim');
    const unknown = describeShell({ bridge: true, info: { ...info, macos_source_entry: null } });
    assert.equal(unknown.appEntry, null);
    assert.equal(shellStorageText(unknown), '', 'unknown app identity must not claim a profile defect');
});

test('loading prefers the app\'s own answer and never throws', async () => {
    const win = (api) => ({ pywebview: { api } });
    let healthAsked = 0;
    const appVersion = async () => { healthAsked += 1; return '6.82.0'; };

    assert.deepEqual(await loadDesktopShell({ win: {}, appVersion }), { desktop: false });
    assert.equal(healthAsked, 0, 'a browser never asks the server about an app it is not in');

    const legacy = await loadDesktopShell({ win: win({ request_attention() {} }), appVersion });
    assert.equal(legacy.persistentStorage, false);
    assert.equal(legacy.version, '6.82.0');
    assert.equal(healthAsked, 1);

    const current = await loadDesktopShell({
        win: win({ shell_info: async () => ({ shell_version: '7.7.0', persistent_storage: true, native_notifications: null }) }),
        appVersion,
    });
    assert.equal(current.version, '7.7.0');
    assert.equal(healthAsked, 1, 'the app answered for itself');

    for (const failing of [async () => { throw new Error('bridge gone'); }, async () => null]) {
        const failed = await loadDesktopShell({
            win: win({ shell_info: failing, show_native_notification() {}, request_native_notifications() {} }),
            appVersion: async () => '7.7.0',
        });
        assert.deepEqual(failed, { desktop: true, version: '7.7.0', persistentStorage: true, native: null },
            'a failed call leaves its facts unknown and claims nothing about the app\'s age');
    }

    const broken = await loadDesktopShell({
        win: win({ shell_info: async () => { throw new Error('bridge gone'); } }),
        appVersion: async () => { throw new Error('offline'); },
    });
    assert.equal(broken.desktop, true);
    assert.equal(broken.persistentStorage, null, 'unknown stays unknown');
});

test('Settings is painted now and again once pywebview injects its bridge', async () => {
    const node = { textContent: '' };
    const root = { querySelectorAll: (selector) => (selector === '[data-shell-storage-status]' ? [node] : []) };
    const listeners = new Map();
    const win = {
        addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type),
    };
    const answers = [describeShell(), describeShell({ bridge: true, appVersion: '6.82.0' })];
    const seen = [];
    const dispose = mountDesktopShell(root, (shell) => seen.push(shell), { win, load: async () => answers.shift() });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(node.textContent, '');
    assert.ok(listeners.has('pywebviewready'));
    listeners.get('pywebviewready')();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.match(node.textContent, /6\.82\.0/);
    assert.equal(seen.length, 2);
    dispose();
    assert.equal(listeners.has('pywebviewready'), false);
});

// One reading of a window capability: supported, explicitly unsupported,
// method or field missing, read error; nothing is missing before the bridge is injected; an earlier answer survives
// a later failed read; a browser or Telegram inherits nothing from the app the server runs in.
test('a capability read sorts into supported, unsupported, missing, error and pending', async () => {
    const read = (api, extra = {}) => readShellInfo({ win: { pywebview: api === undefined ? {} : { api }, ...extra } });
    const field = 'interface_size';
    assert.deepEqual(shellCapability(await readShellInfo({ win: {} }), field), { state: 'none' }, 'a browser tab');
    assert.deepEqual(shellCapability(await readShellInfo({ win: { Telegram: { WebApp: {} } } }), field), { state: 'none' },
        'Telegram has no desktop bridge, whatever app the server runs in');
    assert.equal(shellCapability(await read(undefined), field).state, 'pending', 'the object before pywebview fills it');
    assert.equal(shellCapability(await read({}), field).state, 'pending', 'an empty api is not an answer yet');
    assert.equal(shellCapability(await readShellInfo({ win: { pywebview: { api: {} } }, ready: true }), field).state, 'missing',
        'after pywebviewready an absent method is a fact');
    assert.equal(shellCapability(await read({ request_attention() {} }), field).state, 'missing', 'no shell_info at all');
    assert.equal(shellCapability(await read({ shell_info: async () => ({}) }), field).state, 'missing', 'no such field');
    const no = shellCapability(await read({ shell_info: async () => ({ interface_size: { supported: false } }) }), field);
    assert.deepEqual(no, { state: 'unsupported', value: { supported: false } });
    const yes = shellCapability(await read({ shell_info: async () => ({ interface_size: { supported: true, percent: 120 } }) }), field);
    assert.equal(yes.state, 'supported');
    assert.equal(yes.value.percent, 120);
    assert.equal(shellCapability(await read({ shell_info: async () => ({ persistent_storage: true }) }), 'persistent_storage').state,
        'supported', 'a boolean field is its own answer');
    for (const failing of [async () => { throw new Error('bridge gone'); }, async () => null, async () => 'odd']) {
        assert.equal(shellCapability(await read({ shell_info: failing }), field).state, 'error');
    }
    assert.equal(shellCapability(await read({ shell_info: async () => ({ interface_size: { percent: 100 } }) }), field).state,
        'error', 'an answer without its flag is unreadable, not a refusal');
});

test('an earlier answer survives a later failed or pending read, and is never turned into unsupported', async () => {
    const field = 'interface_size';
    const yes = shellCapability(await readShellInfo({ win: { pywebview: { api: {
        shell_info: async () => ({ interface_size: { supported: true } }) } } } }), field);
    const failed = shellCapability(await readShellInfo({ win: { pywebview: { api: {
        shell_info: async () => { throw new Error('bridge gone'); } } } } }), field, yes);
    assert.equal(failed.state, 'error');
    assert.equal(failed.last, yes, 'the window\'s answer travels on');
    const again = shellCapability({ bridge: true, ready: true, method: true, info: null, failed: true }, field, failed);
    assert.equal(again.last, yes, 'two failures in a row still keep the answer');
    assert.equal(shellCapability({ bridge: true, ready: false, method: false, info: null, failed: false }, field, yes).last, yes);
    assert.equal(shellCapability({ bridge: true, ready: false }, field, null).last, undefined, 'no answer, nothing to keep');
});

test('a later failed read never undoes what the window answered (Settings line)', async () => {
    const node = { textContent: '' };
    const root = { querySelectorAll: (selector) => (selector === '[data-shell-storage-status]' ? [node] : []) };
    const listeners = new Map();
    let calls = 0;
    const win = {
        pywebview: { api: { shell_info: async () => {
            calls += 1;
            if (calls === 1) return { shell_version: '7.7.0', persistent_storage: true, native_notifications: null };
            throw new Error('bridge gone');
        } } },
        addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type),
    };
    let health = 0;
    const seen = [];
    const load = (options) => loadDesktopShell({ ...options, appVersion: async () => { health += 1; return '6.82.0'; } });
    mountDesktopShell(root, (shell) => seen.push(shell), { win, load });
    await new Promise((resolve) => setTimeout(resolve, 0));
    listeners.get('pywebviewready')();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(calls, 2);
    assert.equal(node.textContent, '', 'the old stamped version does not overrule the window\'s own answer');
    assert.equal(seen.length, 1, 'the kept answer is the one already handed on, not a new one');
    assert.equal(seen.at(-1).persistentStorage, true);
    assert.equal(health, 0, 'nothing was asked of the server about an app that answered for itself');
});

test('before the bridge is injected nothing is claimed about the app', async () => {
    let asked = 0;
    const shell = await loadDesktopShell({ win: { pywebview: { api: {} } }, appVersion: async () => { asked += 1; return '6.82.0'; } });
    assert.deepEqual(shell, { desktop: false });
    assert.equal(asked, 0);
    assert.equal(shellStorageText(shell), '');
});


test('an overlapping failed read preserves an answer received while it was pending', async () => {
    const listeners = new Map();
    let resolveFirst, rejectSecond, calls = 0;
    const first = new Promise((resolve) => { resolveFirst = resolve; });
    const second = new Promise((_resolve, reject) => { rejectSecond = reject; });
    const win = {
        pywebview: { api: { shell_info: () => ++calls === 1 ? first : second } },
        addEventListener: (name, fn) => listeners.set(name, fn),
        removeEventListener: (name) => listeners.delete(name),
    };
    const label = { textContent: '' }, seen = [];
    const dispose = mountDesktopShell({ querySelectorAll: () => [label] }, (shell) => seen.push(shell), {
        win, load: (options) => loadDesktopShell({ ...options, appVersion: async () => '6.82.0' }),
    });
    try {
        listeners.get('pywebviewready')();
        resolveFirst({ shell_version: '7.7.0', persistent_storage: true });
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(seen.at(-1).persistentStorage, true);
        rejectSecond(new Error('overlapping bridge read failed'));
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(seen.at(-1).persistentStorage, true);
        assert.equal(label.textContent, '', 'a failed read does not replace the answer with a legacy-version guess');
    } finally {
        dispose();
    }
});

const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i += 1) await new Promise((done) => setImmediate(done)); };

test('a stamped version that arrives after the window answered never replaces that answer', async () => {
    const listeners = new Map();
    let reads = 0;
    let releaseHealth;
    const health = new Promise((resolve) => { releaseHealth = resolve; });
    const answer = { shell_version: '8.0.0', bundle_version: '7.7.0', persistent_storage: true, source_launcher: true,
        macos_source_entry: false, native_notifications: { available: true, status: 'authorized', platform: 'macos', reason: '' } };
    const win = {
        pywebview: { api: { shell_info: async () => {
            reads += 1;
            if (reads === 1) throw new Error('the first read failed');
            return answer;
        } } },
        addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type),
    };
    const label = { textContent: '' };
    const seen = [];
    const dispose = mountDesktopShell({ querySelectorAll: () => [label] }, (shell) => seen.push(shell), {
        win, load: (options) => loadDesktopShell({ ...options, appVersion: () => health }),
    });
    try {
        await settle();  // the first read failed and /api/health has not answered yet
        listeners.get('pywebviewready')();
        await settle();  // meanwhile the window answers for itself
        assert.equal(seen.length, 1);
        assert.match(label.textContent, /source launcher is running without the app entry/);
        releaseHealth('6.82.0');
        await settle();
        assert.equal(seen.length, 1, 'the older read hands nothing on: the window has answered since');
        assert.equal(seen[0].loadedVersion, '8.0.0');
        assert.equal(seen[0].appEntry, false);
        assert.deepEqual(seen[0].native, answer.native_notifications);
        assert.match(label.textContent, /source launcher is running without the app entry/, 'the warning stays');
        assert.doesNotMatch(label.textContent, /6\.82\.0/);
    } finally {
        dispose();
    }
});

test('the stamped version stays the fallback where the window has not answered since', async () => {
    const answered = await loadDesktopShell({ win: { pywebview: { api: { shell_info: async () => ({ shell_version: '7.7.0',
        persistent_storage: true }) } } } });
    const failing = { pywebview: { api: { shell_info: async () => { throw new Error('bridge gone'); } } } };
    const legacy = { pywebview: { api: { request_attention() {} } } };
    let latest = null;
    // No answer before or during the wait: the server's stamped version is all there is.
    const alone = await loadDesktopShell({ win: failing, appVersion: async () => '6.82.0', previous: () => latest });
    assert.equal(alone.version, '6.82.0');
    assert.equal(alone.persistentStorage, false);
    // An answer given while the server was asked is newer than this read.
    const during = await loadDesktopShell({ win: failing, previous: () => latest,
        appVersion: async () => { latest = answered; return '6.82.0'; } });
    assert.equal(during, answered);
    // An app without shell_info is its own fact; an answer it gave before this read does not hide that.
    latest = answered;
    const missing = await loadDesktopShell({ win: legacy, appVersion: async () => '6.82.0', previous: () => latest });
    assert.notEqual(missing, answered);
    assert.equal(missing.version, '6.82.0');
});

test('a shell kept through a failed read is not handed on again; a new answer is', async () => {
    const listeners = new Map();
    let fail = false;
    let status = 'not_determined';
    const win = {
        pywebview: { api: { shell_info: async () => {
            if (fail) throw new Error('bridge gone');
            return { shell_version: '7.7.0', persistent_storage: true, native_notifications: { status } };
        } } },
        addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type),
    };
    let health = 0;
    const seen = [];
    const dispose = mountDesktopShell({ querySelectorAll: () => [] }, (shell) => seen.push(shell), {
        win, load: (options) => loadDesktopShell({ ...options, appVersion: async () => { health += 1; return ''; } }),
    });
    try {
        await settle();
        assert.equal(seen.length, 1);
        fail = true;
        listeners.get('pywebviewready')();
        await settle();
        assert.equal(seen.length, 1, 'what a failed read kept was handed on already');
        fail = false;
        status = 'authorized';
        listeners.get('pywebviewready')();  // the recording window keeps its listener: a second successful read
        await settle();
        assert.equal(seen.length, 2, 'a new answer is a new observation');
        assert.equal(seen[1].native.status, 'authorized');
        assert.equal(health, 0, 'no read of the server for an app that answers for itself');
    } finally {
        dispose();
    }
});
