import { apiFetch } from './api_client.js';
import { safeExternalUrl } from './utils.js';
export { renderSafeField, collectSafeFieldValues, normalizeTone, setInlineStatus } from './ui_primitives.js';
// Cycle note: toast.js imports normalizeTone from this module. Both edges only
// call the imported function inside function bodies (never at module eval), so
// the ES-module cycle is benign.
import { showToast } from './toast.js';


export function installedTime(item) {
    const time = Date.parse(item?.installed_at || item?.provenance?.installed_at || item?.provenance?.updated_at || '');
    return Number.isFinite(time) ? time : 0;
}

export function formatRelativeAge(time, freshLabel = 'Just installed') {
    if (!time) return '';
    const minutes = Math.floor(Math.max(0, Date.now() - time) / 60000);
    if (minutes < 2) return freshLabel;
    if (minutes < 90) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 48) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return days < 45 ? `${days}d ago` : new Date(time).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

// The row under a System or routed chat message: the composition owns spacing, wrapping and
// focus clearance, never the controls inside it.
export function createSystemMessageActions(...controls) {
    const row = document.createElement('div');
    row.className = 'system-message-actions';
    row.append(...controls);
    return row;
}

/**
 * A list editor's freshly added entry is shown where it landed and takes the
 * caret (docs/DESIGN.md "List editors"). The row is scrolled the SHORTEST
 * distance into view with no animation — a WebKit shell's smooth scroll would
 * race the focus below, and a browser test must see the final geometry at once
 * — and the caller's named field receives focus without a second scroll. Both
 * arguments are the caller's: every add path knows its row and its first
 * field, so no heuristic picks one. Detached or stub nodes are tolerated.
 */
export function revealNewRow(row, field) {
    row?.scrollIntoView?.({ block: 'nearest' });
    field?.focus?.({ preventScroll: true });
}

/**
 * A packaged launcher answers `{ok:false, error}` when its file bridge refuses
 * the URL — most often because its allowlist predates the route it was handed,
 * which no amount of retrying fixes. Throwing there leaves the owner with a
 * dead control and an error toast, so degrade to the same copy-link handoff the
 * no-bridge-at-all path already uses. The refusal is rethrown only when even
 * copying fails, so the caller's own error toast still gets its turn.
 */
async function bridgeRefusalFallback(url, error, browserUrl = '') {
    try {
        // Copy the canonical address when the caller carries both: the compat
        // form tracks the CURRENT file-browser root and may already dangle.
        await copyShellLinkWithToast(absoluteShellUrl(browserUrl || url, window), window, document, showToast);
    } catch {
        throw error;
    }
    return { ok: false, native: false, degraded: 'copy-link' };
}

export async function openViaHostBridge(url, filename = 'file', { browserUrl = '' } = {}) {
    // Resolved through the shared shell resolver: in the framed onboarding
    // wizard the bridge lives on the PARENT window, and reading only our own
    // window here would silently fall through to the dead window.open path.
    const api = shellBridgeApi(window);
    const openBridge = api?.open_file_with_default_app;
    if (openBridge) {
        const result = await openBridge(url, filename);
        if (!result?.ok) return bridgeRefusalFallback(url, new Error(result?.error || 'open failed'), browserUrl);
        return { ...result, native: true };
    }
    // Version-skew fallback: the served frontend auto-updates via the managed
    // repo, but the outer desktop launcher only changes on a full app reinstall,
    // so a packaged launcher can predate open_file_with_default_app (added
    // v6.58.3) while still shipping download_file_to_downloads with an
    // open_external flag (since v5.5.0). Reuse that long-shipped bridge — it
    // saves to ~/Downloads AND opens externally — instead of window.open, which
    // is a silent no-op in the desktop WKWebView.
    const downloadBridge = api?.download_file_to_downloads;
    if (downloadBridge) {
        const result = await downloadBridge(url, filename, true);
        if (!result?.ok) return bridgeRefusalFallback(url, new Error(result?.error || 'open failed'), browserUrl);
        return { ...result, native: true };
    }
    // True web / non-desktop: open in a new tab. This never navigates the app
    // itself; the browser previews (e.g. PDF) or downloads per its own handling.
    // A launcher-gate compat address is bridge-only — the browser gets the
    // canonical URL when the caller carries both.
    window.open(browserUrl || url, '_blank', 'noopener');
    return { ok: true, native: false };
}

export async function downloadViaHostBridge(url, filename = 'download', { openExternal = false, fetchOptions = {}, browserUrl = '', streaming = false } = {}) {
    const api = shellBridgeApi(window);
    const bridge = api?.download_file_to_downloads;
    if (bridge) {
        const result = await bridge(url, filename, Boolean(openExternal));
        if (!result?.ok) {
            return bridgeRefusalFallback(url, new Error(result?.error || 'desktop download failed'), browserUrl);
        }
        return { ...result, native: true };
    }
    if (api) {
        return bridgeRefusalFallback(url, new Error('Desktop file download is unavailable'), browserUrl);
    }
    // Browser fallback fetches the canonical address when the caller carries
    // both: the compat form depends on the CURRENT file-browser root and may
    // dangle after the owner re-roots it, while the artifact URL stays valid.
    if (streaming) {
        const source = browserUrl || url;
        const response = await apiFetch(source, { ...fetchOptions, method: 'HEAD' });
        if (!response.ok) throw new Error(`download failed: HTTP ${response.status}`);
        const link = document.createElement('a');
        Object.assign(link, { href: source, download: filename, rel: 'noopener' });
        document.body.appendChild(link);
        link.click();
        link.remove();
        return { ok: true, native: false };
    }
    const resp = await apiFetch(browserUrl || url, fetchOptions);
    if (!resp.ok) throw new Error(`download failed: HTTP ${resp.status}`);
    const blobUrl = URL.createObjectURL(await resp.blob());
    const link = document.createElement('a');
    Object.assign(link, { href: blobUrl, download: filename, rel: 'noopener' });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    return { ok: true, native: false };
}

// ---------------------------------------------------------------------------
// Desktop-shell link interception (pywebview parity).
//
// The embedded WebView is created without new-window or download delegates, so
// `window.open()`, `<a target="_blank">`, and `<a download>` are silent no-ops
// in the desktop shell. ONE shell-only interceptor closes the whole class: a
// delegated document click listener plus a `window.open` shim, installed only
// once the pywebview bridge exists (the bridge appears asynchronously after
// load, announced by the `pywebviewready` event). In an ordinary browser
// nothing is installed — zero behavior change. Bridge METHODS are still
// feature-detected per call: the packaged launcher only updates on reinstall
// while this served frontend updates with the managed repo, so each class has
// an explicit version-skew fallback (copy-link toast / honest "not available"
// toast) instead of a silently dead control.

const BRIDGE_ARTIFACTS_RE = /^\/api\/tasks\/[^/]+\/artifacts\//;

/**
 * Resolve the pywebview bridge for a document. The onboarding wizard is its
 * OWN document inside a same-origin overlay iframe, where pywebview injects
 * the bridge only into the top-level window — so a framed document reads it
 * from the parent. A cross-origin parent (not our shell) throws and resolves
 * to null.
 */
export function shellBridgeApi(win) {
    try {
        const host = win.pywebview || (win.parent && win.parent !== win ? win.parent.pywebview : null);
        return host?.api || null;
    } catch {
        return null;
    }
}

/** Classify a URL for the desktop shell: file | external | bytes | default. */
export function classifyShellUrl(rawUrl, baseHref = '') {
    const raw = String(rawUrl || '').trim();
    if (!raw) return { kind: 'default', url: '' };
    if (/^(data|blob):/i.test(raw)) return { kind: 'bytes', url: raw };
    // mailto: is a real chat-link surface (utils.js safeExternalUrl allows it)
    // and belongs to the OS default handler, exactly like an external page.
    if (/^mailto:/i.test(raw)) return { kind: 'external', url: raw };
    let parsed;
    try { parsed = new URL(raw, baseHref || undefined); } catch { return { kind: 'default', url: raw }; }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return { kind: 'default', url: raw };
    let sameOrigin = false;
    try { sameOrigin = Boolean(baseHref) && parsed.origin === new URL(baseHref).origin; } catch {}
    if (sameOrigin && (parsed.pathname === '/api/files/download'
        || parsed.pathname.startsWith('/api/extensions/')
        || BRIDGE_ARTIFACTS_RE.test(parsed.pathname))) {
        // Path-only form: the launcher guard re-joins it onto the loopback
        // origin itself, exactly like the existing bridge callers pass it.
        return { kind: 'file', url: parsed.pathname + parsed.search };
    }
    // Any other http(s) target ("leave the app" intent) belongs in the owner's
    // real browser — including same-origin pages, which have no tab here.
    return { kind: 'external', url: parsed.href };
}

function bridgeFilenameFromUrl(url, fallback = 'download') {
    try {
        const parsed = new URL(url, 'http://localhost');
        const fromQuery = parsed.searchParams.get('path') || '';
        const candidate = decodeURIComponent((fromQuery || parsed.pathname).split('/').pop() || '');
        return candidate || fallback;
    } catch { return fallback; }
}

// Subtypes whose conventional file extension differs from the MIME token.
const MIME_EXT_ALIASES = { plain: 'txt' };

function filenameForMime(mime) {
    const subtype = String(mime || '').split('/')[1]?.split(/[+;]/)[0]?.toLowerCase() || '';
    const ext = MIME_EXT_ALIASES[subtype] || subtype;
    return `download.${/^[a-z0-9]{1,10}$/.test(ext) ? ext : 'bin'}`;
}

function base64FromArrayBuffer(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
}

async function shellBytesPayload(url, filename, win) {
    if (/^data:/i.test(url)) {
        const comma = url.indexOf(',');
        const meta = comma >= 0 ? url.slice(5, comma) : '';
        if (/;base64$/i.test(meta)) {
            return {
                name: filename || filenameForMime(meta.split(';')[0]),
                b64: url.slice(comma + 1).replace(/\s+/g, ''),
            };
        }
    }
    // blob: URLs and non-base64 data: URLs: let fetch decode them.
    const fetcher = typeof win?.fetch === 'function' ? win.fetch.bind(win) : fetch;
    const blob = await (await fetcher(url)).blob();
    return {
        name: filename || filenameForMime(blob.type),
        b64: base64FromArrayBuffer(await blob.arrayBuffer()),
    };
}

/** Save an already-owned Blob or data/blob URL through the existing bytes owner.
 * HTTP responses/streams stay on the URL download path and are never buffered here.
 */
export async function downloadBlobViaHostBridge(source, filename = '', { win = window, doc = document } = {}) {
    const isBlob = typeof Blob !== 'undefined' && source instanceof Blob;
    if (!isBlob && (typeof source !== 'string' || !/^(data|blob):/i.test(source))) {
        throw new TypeError('Expected a Blob or data/blob URL');
    }
    const api = shellBridgeApi(win);
    if (api) {
        if (!api.save_bytes_to_downloads) return { ok: false, native: true, unavailable: true,
            error: "Saving isn't available in the app — open in a browser." };
        const payload = isBlob
            ? { name: filename || filenameForMime(source.type), b64: base64FromArrayBuffer(await source.arrayBuffer()) }
            : await shellBytesPayload(source, filename, win);
        const result = await api.save_bytes_to_downloads(payload.name, payload.b64);
        const name = String(result?.path || '').split(/[\\/]/).pop() || payload.name;
        return { ...result, ok: result?.ok === true, native: true, filename: name };
    }
    const mime = isBlob ? source.type : (/^data:([^;,]*)/i.exec(source)?.[1] || '');
    const name = filename || filenameForMime(mime);
    const url = isBlob ? URL.createObjectURL(source) : source;
    const link = doc.createElement('a');
    Object.assign(link, { href: url, download: name, rel: 'noopener' });
    doc.body.appendChild(link);
    try { link.click(); } finally {
        link.remove();
        if (isBlob) setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    return { ok: true, native: false, filename: name };
}

/**
 * Clipboard write with the execCommand-textarea fallback. Non-secure origins
 * and the desktop shell have no async clipboard, so a bare
 * `navigator.clipboard?.writeText(...)` is a silently dead control there.
 * Throws when nothing could be copied, so every caller can report honestly.
 */
async function copyTextToClipboard(url, win, doc) {
    try {
        if (!win.navigator?.clipboard?.writeText) throw new Error('no async clipboard');
        await win.navigator.clipboard.writeText(url);
    } catch {
        const area = doc.createElement('textarea');
        area.value = url;
        area.setAttribute('readonly', '');
        doc.body.appendChild(area);
        try {
            area.select();
            const copied = typeof doc.execCommand === 'function' && doc.execCommand('copy') === true;
            if (!copied) throw new Error('Clipboard access is unavailable');
        } finally {
            area.remove();
        }
    }
}

/**
 * Copy owner-visible text and report the outcome through the shared toast host.
 * ONE clipboard contract for controls whose whole job is "put this on the
 * clipboard" (a sign-in link, a device code, a setup command).
 */
export async function copyTextWithToast(text, {
    okMessage = 'Copied.', win = window, doc = document, toast = showToast,
} = {}) {
    const value = String(text ?? '');
    if (!value) {
        toast('Nothing to copy.', 'warn');
        return false;
    }
    try {
        await copyTextToClipboard(value, win, doc);
    } catch (error) {
        toast(`Could not copy: ${error?.message || error}`, 'error');
        return false;
    }
    toast(okMessage, 'ok');
    return true;
}

// With NO file-bridge method at all, openViaHostBridge/downloadViaHostBridge
// fall back to window.open / anchor clicks — dead in the shell, and a re-entry
// into the interceptor's own shim. routeShellUrl degrades that case to the
// copy-link fallback instead of routing through the helpers.
const fileBridgeReady = (api) => Boolean(api?.open_file_with_default_app || api?.download_file_to_downloads);

function absoluteShellUrl(url, win) {
    try { return new URL(url, win?.location?.href).href; } catch { return url; }
}

async function copyShellLinkWithToast(url, win, doc, toast) {
    await copyTextToClipboard(url, win, doc);
    toast('Link copied — open it in your browser.', 'info');
}

/** Open an external link through the active host while the user gesture is live. */
export async function openExternalViaHostBridge(url, {
    win = window, doc = document, toast = showToast, api = shellBridgeApi(win),
} = {}) {
    const target = safeExternalUrl(url);
    if (target === '#') throw new Error('Unsupported external link');
    if (api) {
        const result = api.open_external_url ? await api.open_external_url(target) : null;
        if (result?.ok) return { ...result, native: true };
        await copyShellLinkWithToast(target, win, doc, toast);
        return { ok: false, native: true, degraded: 'copy-link' };
    }
    let telegram = win.Telegram?.WebApp;
    let telegramHost = doc.documentElement?.dataset?.ouroborosHost === 'telegram' || Boolean(telegram);
    // The same-origin onboarding frame inherits its host's opener, just like
    // its pywebview bridge. A foreign parent grants no access here.
    try {
        if (!telegram && win.parent && win.parent !== win) {
            telegram = win.parent.Telegram?.WebApp;
            telegramHost ||= Boolean(telegram) || win.parent.document?.documentElement?.dataset?.ouroborosHost === 'telegram';
        }
    } catch { /* cross-origin parent is not our host */ }
    if (telegramHost && /^https?:/i.test(target)) {
        if (typeof telegram?.openLink !== 'function') throw new Error('Telegram link opener is not ready; try the link again');
        telegram.openLink(target);
        return { ok: true, native: false, host: 'telegram' };
    }
    win.open(target, '_blank', 'noopener');
    return { ok: true, native: false, host: 'browser' };
}

async function routeShellUrl(kind, url, deps) {
    const { api, win, doc, toast, openFile, downloadFile, filename = '', wantsDownload = false } = deps;
    try {
        if (kind === 'file') {
            if (!fileBridgeReady(api)) {
                // Ancient launcher with no file bridge at all: window.open and
                // anchor downloads are dead here, so hand the owner the link.
                await copyShellLinkWithToast(absoluteShellUrl(url, win), win, doc, toast);
                return;
            }
            const name = filename || bridgeFilenameFromUrl(url);
            // A refusal from inside these helpers already handed the owner a
            // copy-link toast; only a genuine throw reaches the catch below.
            if (wantsDownload) await downloadFile(url, name);
            else await openFile(url, name);
        } else if (kind === 'external') {
            await openExternalViaHostBridge(url, { api, win, doc, toast });
        } else if (kind === 'bytes') {
            const result = await downloadBlobViaHostBridge(url, filename, { win, doc });
            if (result.unavailable) { toast(result.error, 'warn'); return; }
            if (!result.ok) throw new Error(result.error || 'save failed');
            toast(`Saved to Downloads: ${result.filename}`, 'ok');
        }
    } catch (error) {
        const verb = kind === 'bytes' ? 'save file'
            : kind === 'file' ? (wantsDownload ? 'download file' : 'open file')
                : 'open link';
        toast(`Could not ${verb}: ${error?.message || error}`, 'error');
    }
}

/**
 * Install the shell-only interceptor. Idempotence and browser neutrality:
 * installs at most once per document/window pair, and only when the pywebview
 * bridge is (or becomes) present.
 */
export function installDesktopShellLinkInterceptor({
    win = window,
    doc = document,
    toast = showToast,
    openFile = openViaHostBridge,
    downloadFile = downloadViaHostBridge,
} = {}) {
    let installed = false;
    const install = () => {
        if (installed) return;
        installed = true;
        const nativeOpen = typeof win.open === 'function' ? win.open.bind(win) : () => null;
        const deps = (extra) => ({
            api: shellBridgeApi(win), win, doc, toast, openFile, downloadFile, ...extra,
        });
        doc.addEventListener('click', (event) => {
            const api = shellBridgeApi(win);
            if (!api || event.defaultPrevented) return;
            const anchor = typeof event.target?.closest === 'function' ? event.target.closest('a[href]') : null;
            if (!anchor) return;
            const wantsDownload = anchor.hasAttribute('download');
            if (anchor.getAttribute('target') !== '_blank' && !wantsDownload) return;
            const { kind, url } = classifyShellUrl(anchor.getAttribute('href'), win.location?.href);
            if (kind === 'default') return;
            event.preventDefault();
            void routeShellUrl(kind, url, deps({
                wantsDownload,
                filename: String(anchor.getAttribute('download') || ''),
            }));
        });
        win.open = (url, target, features) => {
            const api = shellBridgeApi(win);
            const { kind, url: routed } = api
                ? classifyShellUrl(url, win.location?.href)
                : { kind: 'default', url };
            if (kind === 'default') return nativeOpen(url, target, features);
            void routeShellUrl(kind, routed, deps({}));
            return null;
        };
    };
    if (shellBridgeApi(win)) {
        install();
        return;
    }
    win.addEventListener?.('pywebviewready', install, { once: true });
    // A framed document (the onboarding wizard) never receives pywebviewready
    // itself — the bridge announcement fires on the top-level window. The
    // frame is same-origin by contract; a cross-origin parent just no-ops.
    // Disposer rule: in an ordinary browser that parent listener never fires,
    // and it must not outlive the framed document — reopening the overlay
    // would otherwise accumulate closures on the parent window.
    try {
        if (win.parent && win.parent !== win && typeof win.parent.addEventListener === 'function') {
            const disposer = typeof AbortController === 'function' ? new AbortController() : null;
            win.parent.addEventListener('pywebviewready', install, { once: true, signal: disposer?.signal });
            if (disposer) win.addEventListener?.('pagehide', () => disposer.abort(), { once: true });
        }
    } catch { /* not our shell */ }
}

/**
 * Decision helper for Windows Alt/Alt+Shift keyboard layout switch focus-lock.
 * When focus is within an editable text input, prevents standalone Alt keydown
 * from triggering Windows window-menu activation (which beeps and drops the next keystroke).
 * AltGr is intentionally preserved in both engine shapes: Chromium reports it as
 * Ctrl+Alt (excluded via ctrlKey), Firefox as key='AltGraph' (excluded by name).
 */
export function shouldSuppressWindowsAltMenu(event, activeElement) {
    if (!event) return false;
    if (event.key === 'AltGraph') return false;
    const isAlt = (event.key === 'Alt' || event.code === 'AltLeft' || event.code === 'AltRight') && !event.ctrlKey;
    if (!isAlt) return false;
    if (!activeElement) return false;
    const tag = String(activeElement.tagName || '').toUpperCase();
    return tag === 'INPUT' || tag === 'TEXTAREA' || Boolean(activeElement.isContentEditable);
}

/**
 * Install the standalone-Alt menu-lock guard on a document's window. Chromium
 * hosts decide menu activation from the Alt KEYDOWN default while Firefox-style
 * hosts decide on KEYUP, so both phases carry the same suppression test. Each
 * top-level document installs its own guard (the onboarding wizard runs in an
 * iframe with its own window, where the SPA listener cannot see events). The
 * guard is an app-lifetime singleton like the other top-level app.js listeners,
 * so it deliberately registers no disposer.
 */
export function installAltMenuSuppression(target = window) {
    const doc = target.document || document;
    const suppress = (event) => {
        if (shouldSuppressWindowsAltMenu(event, doc.activeElement)) {
            event.preventDefault();
        }
    };
    target.addEventListener('keydown', suppress, { capture: true });
    target.addEventListener('keyup', suppress, { capture: true });
}
