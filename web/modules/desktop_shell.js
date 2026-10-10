/* The desktop app around this page, as it reports itself.

   In-app updates replace the core (server and this page) but never the desktop app
   that hosts the window: its launcher is frozen into the installed bundle, and it
   alone decides whether the WebView keeps website data and which system bridge
   methods exist. So the page asks the app, not the core, what it can do here:

   - a current app answers `shell_info()` with its own version, its storage and its
     system-notification permission;
   - an older app has no `shell_info`, and the version the server reports as the
     app's (`/api/health` `app_version`, stamped by that launcher) is all there is.
     Apps before 7.2.0 start the WebView in pywebview's private mode, which erases
     website data whenever a window opens, so Theme and Notifications come back
     as defaults after every launch; Settings says so instead of pretending.
   - a `shell_info` call that fails is read like a missing one: only the stamped
     version is left, and nothing else is inferred from the failure. Whether the
     app can send system notifications is the notifier's own check of its method,
     never a version, so a failed call cannot make a current app look old.

   A browser tab has no desktop bridge and none of these facts apply to it. */

import { apiFetch } from './api_client.js';
import { shellBridgeApi } from './ui_helpers.js';

/** The first app release whose launcher starts the WebView with persistent storage. */
export const PERSISTENT_STORAGE_SINCE = Object.freeze([7, 2, 0]);

export function parseAppVersion(raw) {
    const match = /^v?(\d+)\.(\d+)\.(\d+)/.exec(String(raw || '').trim());
    return match ? match.slice(1, 4).map(Number) : null;
}

function before(version, floor) {
    for (let i = 0; i < 3; i += 1) {
        if (version[i] !== floor[i]) return version[i] < floor[i];
    }
    return false;
}

/** Pure: what the bridge (`info`: `shell_info()`'s answer, or null when the app has no such
 *  method or its call failed) and the server said. */
export function describeShell({ bridge = false, info = null, appVersion = '' } = {}) {
    if (!bridge) return { desktop: false };
    if (info && typeof info === 'object') {
        const native = info.native_notifications;
        return {
            desktop: true,
            version: String(info.shell_version || appVersion || ''),
            persistentStorage: typeof info.persistent_storage === 'boolean' ? info.persistent_storage : null,
            native: native && typeof native === 'object' ? { ...native } : null,
        };
    }
    const parsed = parseAppVersion(appVersion);
    return {
        desktop: true,
        version: String(appVersion || ''),
        persistentStorage: parsed ? !before(parsed, PERSISTENT_STORAGE_SINCE) : null,
        native: null,
    };
}

const appName = (shell) => `This desktop app${shell?.version ? ` (${shell.version})` : ''}`;

/** One line for Settings → Appearance, or '' when there is nothing to disclose. */
export function shellStorageText(shell) {
    if (!shell?.desktop || shell.persistentStorage !== false) return '';
    return `${appName(shell)} erases this window's saved data every time it starts, so Theme and `
        + 'Notifications return to their defaults after a restart. Install the current Ouroboros app to keep '
        + 'them: updates inside Ouroboros replace its core, not the app around it.';
}

async function serverAppVersion() {
    const response = await apiFetch('/api/health', { cache: 'no-store' });
    if (!response.ok) return '';
    const data = await response.json();
    return String(data?.app_version || '');
}

/** Ask the app around this page; never throws (unknown facts stay unknown). */
export async function loadDesktopShell({ win = globalThis, appVersion = serverAppVersion } = {}) {
    const api = shellBridgeApi(win);
    if (!api) return describeShell();
    let info = null;
    if (typeof api.shell_info === 'function') {
        try { info = await api.shell_info(); } catch { info = null; }
    }
    let version = '';
    if (!info) {
        try { version = await appVersion(); } catch { version = ''; }
    }
    return describeShell({ bridge: true, info: info || null, appVersion: version });
}

/* Paint the shell facts into a mounted Settings page and hand them on. pywebview injects its
   bridge asynchronously, so a page opened before `pywebviewready` is painted again then. */
export function mountDesktopShell(root, onShell = () => {}, { win = globalThis, load = loadDesktopShell } = {}) {
    let disposed = false;
    const paint = () => load({ win }).then((shell) => {
        if (disposed) return;
        for (const node of root?.querySelectorAll?.('[data-shell-storage-status]') || []) {
            const next = shellStorageText(shell);
            if (node.textContent !== next) node.textContent = next;
        }
        onShell(shell);
    }).catch(() => { /* the facts stay unknown; nothing is claimed */ });
    void paint();
    win.addEventListener?.('pywebviewready', paint, { once: true });
    return () => {
        disposed = true;
        win.removeEventListener?.('pywebviewready', paint);
    };
}
