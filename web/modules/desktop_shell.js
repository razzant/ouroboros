/* The desktop app around this page, as it reports itself.

   In-app updates replace the managed source, including the source launcher and its
   helpers. They cannot replace the installed app's signed executables. The running
   shell reports its capabilities; a checkout version cannot prove app identity.

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

   One capability of this window is read from this window alone: `shellCapability` sorts one read into supported, unsupported (the window said
   so), missing (no method, or no such field in its answer) and error (the read failed);
   before pywebview has injected its bridge, nothing is missing yet (pending). An answer
   once given is never turned into "unsupported" by a later failed read. A browser tab or
   Telegram has no desktop bridge and none of these facts, whatever app the server runs in.
   Feature screens say what a capability does here; what to update is Dashboard › Updates'. */

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

/** One read of `shell_info()` from the window around this page; never throws.
 *  `bridge`: a desktop bridge exists (its object, even before pywebview filled it); `ready`: its methods are
 *  injected, or the caller saw `pywebviewready`; `method`: it has `shell_info`; `info`: the answer, when it was an
 *  object; `failed`: the call threw or answered something else. */
export async function readShellInfo({ win = globalThis, ready = false } = {}) {
    const api = shellBridgeApi(win);
    let present = false;
    try { present = Boolean(win?.pywebview); } catch { present = false; }
    if (!api) return { bridge: present, ready: false, method: false, info: null, failed: false };
    const injected = ready || Object.values(api).some((value) => typeof value === 'function');
    if (typeof api.shell_info !== 'function') {
        return { bridge: true, ready: injected, method: false, info: null, failed: false };
    }
    try {
        const info = await api.shell_info();
        const readable = Boolean(info) && typeof info === 'object';
        return { bridge: true, ready: true, method: true, info: readable ? info : null, failed: !readable };
    } catch {
        return { bridge: true, ready: true, method: true, info: null, failed: true };
    }
}

const ANSWERED = new Set(['supported', 'unsupported']);

/** What one read (`readShellInfo`) says about one capability, the answer's `field`: `{state, value?, last?}`.
 *  `state` is `none` (no desktop window: a browser, Telegram), `pending` (the bridge is not injected yet),
 *  `missing` (no `shell_info`, or no such field), `unsupported` / `supported` (the window's own answer, a boolean
 *  or an object's boolean `supported`; `value` is that answer) or `error` (the read failed, or its field could not
 *  be read). `previous` is the same capability's last result: its last answer travels on as `last` through a
 *  failed or pending read, so an earlier answer is never turned into "unsupported" by a failure. Pure. */
export function shellCapability(read, field, previous = null) {
    const last = ANSWERED.has(previous?.state) ? previous : (previous?.last || null);
    if (!read?.bridge) return { state: 'none' };
    if (read.failed) return { state: 'error', ...(last ? { last } : {}) };
    if (!read.method || !read.info) return read.ready ? { state: 'missing' } : { state: 'pending', ...(last ? { last } : {}) };
    const value = read.info[field];
    if (value === undefined || value === null) return { state: 'missing' };
    const flag = typeof value === 'boolean' ? value : (typeof value === 'object' ? value.supported : undefined);
    if (typeof flag !== 'boolean') return { state: 'error', ...(last ? { last } : {}) };
    return { state: flag ? 'supported' : 'unsupported', value };
}

/** Pure: what the bridge (`info`: `shell_info()`'s answer, or null when the app has no such
 *  method or its call failed) and the server said. */
export function describeShell({ bridge = false, info = null, appVersion = '' } = {}) {
    if (!bridge) return { desktop: false };
    if (info && typeof info === 'object') {
        const native = info.native_notifications;
        return {
            desktop: true,
            version: String(info.bundle_version || info.shell_version || appVersion || ''),
            ...(typeof info.source_launcher === 'boolean' ? {
                loadedVersion: String(info.shell_version || ''),
                sourceLauncher: info.source_launcher,
                appEntry: typeof info.macos_source_entry === 'boolean' ? info.macos_source_entry : null,
            } : {}),
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

/** One line for Settings → Appearance, or '' when there is nothing to disclose. A fact about this window, never a
 *  remedy: whether another app or an update would change it is not something this window can prove. */
export function shellStorageText(shell) {
    if (!shell?.desktop) return '';
    if (shell.sourceLauncher && shell.appEntry === false) {
        return 'This source launcher is running without the app entry. Its saved Theme and Notifications '
            + 'may differ from an ordinary app launch. The older Python profile is left untouched.';
    }
    if (shell.persistentStorage !== false) return '';
    return `${appName(shell)} erases this window's saved data every time it starts, so Theme and `
        + 'Notifications return to their defaults after a restart.';
}

async function serverAppVersion() {
    const response = await apiFetch('/api/health', { cache: 'no-store' });
    if (!response.ok) return '';
    const data = await response.json();
    return String(data?.app_version || '');
}

// The shells described from the window's own answer: a later failed read may keep one of them.
const answered = new WeakSet();

/** Ask the app around this page; never throws (unknown facts stay unknown). Before the bridge is injected nothing
 *  is claimed. `previous` is an answered shell or a getter for the latest one when reads overlap; it outlives a
 *  failed read, and one the window gave while the server was being asked outranks the server's stamped version. */
export async function loadDesktopShell({ win = globalThis, appVersion = serverAppVersion, previous = null } = {}) {
    const read = await readShellInfo({ win });
    if (!read.bridge || !read.ready) return describeShell();
    if (read.info) {
        const shell = describeShell({ bridge: true, info: read.info });
        answered.add(shell);
        return shell;
    }
    const latest = () => (typeof previous === 'function' ? previous() : previous);
    const before = latest();
    if (read.failed && answered.has(before)) return before;
    let version = '';
    try { version = await appVersion(); } catch { version = ''; }
    const after = latest();  // a newer read may have been answered meanwhile: it speaks for the window, not this one
    if (after !== before && answered.has(after)) return after;
    return describeShell({ bridge: true, info: null, appVersion: version });
}

/* Paint the shell facts into a mounted Settings page and hand them on. pywebview injects its
   bridge asynchronously, so a page opened before `pywebviewready` is painted again then. Only a new
   observation is handed on: the shell a failed read kept is the one already given, and giving it again
   would undo what its consumer learned since (a permission the owner granted meanwhile). */
export function mountDesktopShell(root, onShell = () => {}, { win = globalThis, load = loadDesktopShell } = {}) {
    let disposed = false;
    let previous = null;
    const paint = () => load({ win, previous: () => previous }).then((shell) => {
        if (disposed || shell === previous) return;
        previous = shell;
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
