/* The desktop app's own facts: version, storage and system notifications.

   The disclosure exists because in-app updates replace the core and this page, never
   the frozen launcher that decides whether the WebView keeps website data. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
    PERSISTENT_STORAGE_SINCE,
    describeShell,
    loadDesktopShell,
    mountDesktopShell,
    parseAppVersion,
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
    assert.match(text, /updates inside Ouroboros replace its core, not the app around it/);

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
