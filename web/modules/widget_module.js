/* Framed widget mounts: the extension-route iframe and the reviewed module srcdoc iframe
   (the shared sandbox / permissions set, the module document CSP, the parent side of the one
   streaming bridge — fetch relay, abort, skill WebSocket event forwarding — plus the resize
   bridge and source-load timeout).
   widgets.js keeps the page host, the mountTab dispatcher, the card registry and the
   declarative renderer; every mount here returns its disposer to that dispatcher. */

import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePrefix } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bridgeChunkBuffer, moduleBridgeScript, moduleResizeScript } from './widget_frame.js';
import { boundedNumber, WIDGET_DISPOSE_ACK_TIMEOUT_MS, WIDGET_REQUEST_TIMEOUT_MS } from './widget_job.js';
import { onThemeChange } from './theme_palette.js';
import { openWidgetAssetLease, readWidgetAsset, revokeWidgetAssetLease, maintainWidgetAssetLease, reopenWidgetAssetLease } from './widget_assets.js';
import { setWidgetCardFault } from './widget_card.js';
import { downloadViaHostBridge, downloadBlobViaHostBridge, openExternalViaHostBridge } from './ui_helpers.js';

export const WIDGET_FRAME_DEFAULT_HEIGHT = 320;
export const WIDGET_FRAME_MAX_HEIGHT = 8192;
export const WIDGET_FRAME_BORDER_RESERVE = 2;
// Faults one mount sends to the host log (POST /api/widgets/fault); the frame's own cap is not a bound, since the
// widget's code can read the nonce and post directly. The kinds and clips are the frame's (widget_frame.js).
export const WIDGET_FAULT_REPORTS_PER_MOUNT = 10;
const FAULT_KINDS = ['error', 'rejection', 'csp'];
const faultText = (value, max) => (typeof value === 'string'
    ? value.replace(/(\/api\/widget-(?:assets|resources)\/)[A-Za-z0-9_-]+/g, '$1[redacted]').slice(0, max) : '');

// One capability set for both framed mounts (route iframe and module srcdoc):
// scripts, pointer lock and downloads; autoplay, fullscreen and clipboard
// writes through the permissions policy. Never the same-origin token (the frame
// stays an opaque origin without the SPA's cookies, storage or DOM), never top
// navigation, popups, forms, modals or clipboard reads.
export const WIDGET_FRAME_SANDBOX = 'allow-scripts allow-pointer-lock allow-downloads';
export const WIDGET_FRAME_ALLOW = 'autoplay; fullscreen; clipboard-write';

function createWidgetFrame() {
    const iframe = document.createElement('iframe');
    iframe.className = 'widgets-frame';
    iframe.setAttribute('sandbox', WIDGET_FRAME_SANDBOX);
    iframe.setAttribute('allow', WIDGET_FRAME_ALLOW);
    iframe.setAttribute('allowfullscreen', '');
    return iframe;
}

// Sources are absolute because an opaque frame's 'self' matches nothing.
// Classic widgets keep the skill's module sources and load passive images,
// media and fonts from this mount's GET/HEAD route lease; ESM uses only this
// mount's finite file prefix. Neither mode admits a scriptable network source:
// requests fall to default-src 'none', and the parent bridge carries fetch.
export function moduleFrameCsp(skill, origin = window.location.origin, assetBase = '', resourceBase = '') {
    const prefix = `${origin}${extensionRoutePrefix(skill)}`;
    if (assetBase) return [
        "default-src 'none'",
        `script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: ${assetBase}`,
        'worker-src blob:',
        `style-src 'unsafe-inline' ${assetBase}`,
        `img-src data: blob: ${assetBase}`,
        `media-src data: blob: ${assetBase}`,
        `font-src data: blob: ${assetBase}`,
        `base-uri ${assetBase}`,
    ].join('; ');
    const passive = ['data:', 'blob:', resourceBase].filter(Boolean).join(' ');
    return [
        "default-src 'none'",
        `script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: ${prefix}module/`,
        'worker-src blob:',
        "style-src 'unsafe-inline'",
        `img-src ${passive}`,
        `media-src ${passive}`,
        `font-src ${passive}`,
    ].join('; ');
}

export function frameHeight(render, fallback = WIDGET_FRAME_DEFAULT_HEIGHT) {
    return boundedNumber(render?.height, fallback, WIDGET_FRAME_DEFAULT_HEIGHT, WIDGET_FRAME_MAX_HEIGHT);
}

function frameMaxHeight(render) {
    return boundedNumber(render?.max_height, WIDGET_FRAME_MAX_HEIGHT, WIDGET_FRAME_DEFAULT_HEIGHT, WIDGET_FRAME_MAX_HEIGHT);
}

// The declared frame height is the one measured value the frame (and the
// stopped card's facade standing in for it) carries as a custom property.
export function setFrameHeight(node, height) {
    if (!node) return;
    node.style.setProperty('--widget-frame-height', `${Math.ceil(height)}px`);
}

// A fresh unguessable nonce per mount; getRandomValues also works on a plain-http LAN page.
function frameNonce() {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

const resolvedTheme = () => {
    const candidate = window.ouroTheme?.theme || document.documentElement?.dataset?.theme;
    return ['light', 'dark'].includes(candidate) ? candidate : 'dark';
};

// Mint a route lease within the mount's own deadline; a stop during the mint aborts it.
async function openRouteLease(tab, mountSignal) {
    const controller = new AbortController();
    const relayAbort = () => controller.abort();
    if (mountSignal?.aborted) controller.abort();
    else mountSignal?.addEventListener('abort', relayAbort, { once: true });
    const timer = setTimeout(() => controller.abort(), WIDGET_REQUEST_TIMEOUT_MS);
    try {
        return await openWidgetAssetLease(tab, controller.signal, 'routes');
    } finally {
        clearTimeout(timer);
        mountSignal?.removeEventListener('abort', relayAbort);
    }
}

export async function mountRouteIframeWidget(mount, tab, render, mountSignal = null, inline = null, messageHandlers = null) {
    const route = cleanExtensionRoute(render.route);
    if (!route) throw new Error('invalid widget iframe route');
    // The skill's own page in a sandboxed opaque-origin frame with the same
    // capability set as a module frame, loaded from this mount's GET/HEAD route
    // lease: its relative images, media, styles, scripts and reads stay under the
    // lease, with no SPA cookies or DOM and no referrer, and its own HTTP policy
    // and URL base. A page that loads /api/widget-bridge.js gets the same bridge
    // as a module frame (fetch with any method to its own routes, events, theme,
    // downloads), proven by the lease in its own URL. The URL goes in through the
    // property, never an attribute string.
    const lease = await openRouteLease(tab, mountSignal);
    if (mountSignal?.aborted) { revokeWidgetAssetLease(lease); return; }
    const iframe = createWidgetFrame();
    iframe.referrerPolicy = 'no-referrer';
    mount.replaceChildren(iframe);
    setFrameHeight(iframe, frameHeight(render));
    const dispose = attachFrameBridge(iframe, mount, tab, {
        accepts: (hello) => hello.lease === lease.token, rehello: true, resourceBase: lease.baseUrl,
        inline, messageHandlers, lease,
    });
    iframe.src = `${lease.baseUrl}${route}`;
    return dispose;
}

// `messageHandlers` is the page's Set of WebSocket message handlers (fanned from
// `ctx.ws.on('message')`); the frame's event subscription registers into it. `inline` is a
// frame shown in a message instead (a living card, card_frames.js): `{fit(contentHeight),
// fault(text)}` — the message's size policy replaces the board's floor and declared height,
// and a fault is the card's to show.
export async function mountModuleWidget(mount, tab, render, mountSignal = null, messageHandlers = null, inline = null) {
    // The parent owns API I/O and WebSocket forwarding. Native imports and
    // passive resources use the policy above; neither exposes the parent DOM
    // or session to the opaque frame.
    const entryName = String(render.entry).replace(/[^A-Za-z0-9._-]/g, '');
    const entryUrl = `${extensionRoutePrefix(tab.skill)}module/${encodeURIComponent(entryName)}`;
    const sourceController = new AbortController();
    const relayAbort = () => sourceController.abort();
    if (mountSignal?.aborted) sourceController.abort();
    else mountSignal?.addEventListener('abort', relayAbort, { once: true });
    let sourceTimedOut = false;
    const sourceTimeout = setTimeout(() => {
        sourceTimedOut = true;
        sourceController.abort();
    }, WIDGET_REQUEST_TIMEOUT_MS);
    let resp;
    let moduleSource;
    let lease = null;
    try {
        if (render.entry_type === 'module') {
            lease = await openWidgetAssetLease(tab, sourceController.signal);
        } else {
            resp = await apiFetch(entryUrl, { cache: 'no-store', signal: sourceController.signal });
            moduleSource = await resp.text();
            if (resp.ok) lease = await openWidgetAssetLease(tab, sourceController.signal, 'routes');
        }
    } catch (error) {
        if (sourceTimedOut) {
            const timeoutError = new Error('widget module request timed out');
            timeoutError.code = 'WIDGET_REQUEST_TIMEOUT';
            timeoutError.retryable = true;
            throw timeoutError;
        }
        throw error;
    } finally {
        clearTimeout(sourceTimeout);
        mountSignal?.removeEventListener('abort', relayAbort);
    }
    if (mountSignal?.aborted) { revokeWidgetAssetLease(lease); return; }
    if (resp && !resp.ok) {
        mount.innerHTML = `<div class="skills-load-error">module load failed: ${escapeHtml(moduleSource || `HTTP ${resp.status}`)}</div>`;
        return;
    }
    const expectedPrefix = extensionRoutePrefix(tab.skill);
    const nonce = frameNonce();
    const esm = render.entry_type === 'module';
    const assetBase = esm ? lease?.baseUrl || '' : '';
    const resourceBase = esm ? '' : lease?.baseUrl || '';
    const csp = moduleFrameCsp(tab.skill, window.location.origin, assetBase, resourceBase);
    const escapeScript = (value) => String(value || '')
        .replace(/<\/script/gi, '<\\/script')
        .replace(/<!--/g, '<\\!--');
    const autoHeight = render.height === undefined || render.height === null;
    const maxHeight = frameMaxHeight(render);
    const initialTheme = resolvedTheme();
    const bridge = moduleBridgeScript(nonce, `${window.location.origin}${expectedPrefix}`, initialTheme, assetBase, resourceBase);
    const resizeBridge = autoHeight
        ? moduleResizeScript(WIDGET_FRAME_DEFAULT_HEIGHT, maxHeight, WIDGET_FRAME_BORDER_RESERVE)
        : '';
    // The initial theme is carried inside the bridge bootstrap, not as a
    // document attribute: a module must explicitly subscribe before it gets
    // appearance data or changes its own palette.
    const base = assetBase ? `<meta name="referrer" content="no-referrer"><base href="${escapeHtml(assetBase)}">` : '';
    const entryScript = assetBase
        ? `<script type="module" crossorigin="anonymous" src="${escapeHtml(new URL(render.entry, assetBase).href)}"></script>`
        : `<script>${escapeScript(moduleSource)}</script>`;
    const srcdoc = `<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="${csp}">${base}</head><body><div id="root"></div><script>${bridge}</script><script>${resizeBridge}</script>${entryScript}</body></html>`;
    // The document goes in through the `srcdoc` property (no attribute
    // escaping round-trip of a module-sized payload); the frame carries the
    // shared sandbox / permissions set — nothing that re-exposes the SPA origin.
    const iframe = createWidgetFrame();
    if (assetBase) iframe.referrerPolicy = 'no-referrer';
    iframe.srcdoc = srcdoc;
    mount.replaceChildren(iframe);
    const appliedHeight = frameHeight(render);
    if (inline) inline.fit(autoHeight ? 0 : appliedHeight);
    else setFrameHeight(iframe, appliedHeight);
    return attachFrameBridge(iframe, mount, tab, {
        accepts: (hello) => hello.nonce === nonce, nonce, assetBase, resourceBase, autoHeight, maxHeight,
        appliedHeight, inline, messageHandlers, lease, leaves: true,
        leaseEnded: esm ? 'Widget file access ended' : 'Widget access ended',
    });
}

// The parent side of the one bridge, shared by module and route frames. A frame is heard only after its
// document says hello with a MessageChannel port and the mount's proof (`accepts`: the module nonce, or the
// route page's lease); every request, reply, event, theme, resize and dispose then travels on that port and
// nothing is ever posted to `iframe.contentWindow`, so whatever document later occupies the frame learns
// nothing. A route page may hello again from its next own document (`rehello`), which retires the previous
// document's requests and subscriptions; a module frame that loads a second document (`leaves`) ends at once.
function attachFrameBridge(iframe, mount, tab, {
    accepts, rehello = false, nonce = '', assetBase = '', resourceBase = '', autoHeight = false,
    maxHeight = WIDGET_FRAME_MAX_HEIGHT, appliedHeight: initialHeight = 0, inline = null, messageHandlers = null,
    lease = null, leaves = false, leaseEnded = 'Widget access ended',
}) {
    const expectedPrefix = extensionRoutePrefix(tab.skill);
    const wsPrefix = String(tab.ws_prefix || '').trim();
    let appliedHeight = initialHeight;
    // The fault a card shows also reaches the host's event log once per kind+message, under this mount's own
    // identity (captured now, never the frame's words). Best effort: no retry, and a failed report changes nothing.
    const faultIdentity = { skill: String(tab.skill || ''), tab_id: String(tab.tab_id || ''),
        revision: String(tab.revision ?? ''), where: inline ? 'message' : 'board' };
    const reportedFaults = new Set();
    const reportFault = (msg) => {
        const message = faultText(msg.message, 500);
        const key = `${msg.kind}\u0000${message}`;
        if (!FAULT_KINDS.includes(msg.kind) || reportedFaults.has(key) || reportedFaults.size >= WIDGET_FAULT_REPORTS_PER_MOUNT) return;
        reportedFaults.add(key);
        const line = Number.isInteger(msg.line) && msg.line >= 0 && msg.line <= 1e7 ? msg.line : 0;
        const payload = { ...faultIdentity, kind: msg.kind, message, source: faultText(msg.source, 200), line };
        Promise.resolve().then(() => apiClient.reportWidgetFault(payload)).catch(() => {});
    };
    const pendingRequests = new Map();
    let disposed = false;
    let left = false;
    let disposing = null;
    let onDisposed = null;
    let themeSubscribed = false;
    let stopTheme = () => {};
    let releaseLease = () => {};
    let port = null;
    // A reply belongs to the document that asked: it is dropped once that document's port was retired.
    const sender = () => {
        const target = port;
        return (message, transfer = []) => {
            if (!disposed && target && target === port) target.postMessage({ ...message, nonce }, transfer);
        };
    };
    const post = (message, transfer = []) => sender()(message, transfer);
    // A lease URL names this skill's own route; the bridge sends it as that route, with the owner's session.
    const routed = (parsed) => (resourceBase && parsed.href.startsWith(resourceBase)
        ? new URL(`${expectedPrefix}${parsed.href.slice(resourceBase.length)}`, window.location.origin) : parsed);
    const postTheme = () => {
        const theme = resolvedTheme();
        if (themeSubscribed) post({ type: 'ouro-widget-theme', theme });
    };
    const startTheme = () => {
        if (!themeSubscribed) {
            themeSubscribed = true;
            stopTheme = onThemeChange(postTheme);
        }
        postTheme();
    };
    const stopThemeSubscription = () => {
        themeSubscribed = false;
        stopTheme();
        stopTheme = () => {};
    };
    // The skill's namespaced WebSocket events — the same `ws_prefix` filter the
    // declarative `subscription` uses — forwarded only while the child subscribes.
    const onWsMessage = (msg) => {
        const type = String(msg?.type || '');
        if (!wsPrefix || !type.startsWith(wsPrefix)) return;
        post({ type: 'ouro-widget-event', event: type.slice(wsPrefix.length), data: msg.data ?? {} });
    };
    // One bridged fetch: the owning-prefix check, then the response streamed to
    // the child as frames — `headers` (status, status text, every header) first,
    // one `data` frame per body chunk as a transferred ArrayBuffer, then `end`;
    // `error` on any failure. HEAD and 204/205/304 answers have no body to pump.
    // No default timeout: `init.timeoutMs` is the author's opt-in bound;
    // `-fetch-abort`, dispose (after the ack window) and a vanished frame abort
    // through the same controller.
    const relayFetch = async (msg) => {
        const id = msg.id;
        const init = msg.init || {};
        const controller = new AbortController();
        let credit = false;
        let wake = null;
        controller.pull = () => { credit = true; wake?.(); };
        const waitForPull = async () => {
            if (!credit) await new Promise((resolve, reject) => {
                const abort = () => { wake = null; reject(new DOMException('Aborted', 'AbortError')); };
                wake = () => {
                    wake = null;
                    controller.signal.removeEventListener('abort', abort);
                    resolve();
                };
                if (controller.signal.aborted) abort();
                else controller.signal.addEventListener('abort', abort, { once: true });
            });
            credit = false;
        };
        pendingRequests.set(id, controller);
        const reply = sender();
        let timedOut = false;
        const timeoutMs = Number(init.timeoutMs);
        const timer = Number.isFinite(timeoutMs) && timeoutMs > 0
            ? setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, timeoutMs)
            : 0;
        const frame = (fields, transfer) => reply({ type: 'ouro-widget-fetch-chunk', id, ...fields }, transfer);
        try {
            const parsed = routed(new URL(String(msg.url || ''), assetBase || window.location.origin));
            const asset = assetBase && parsed.href.startsWith(assetBase);
            const method = String(init.method || 'GET').toUpperCase();
            if (parsed.origin !== window.location.origin
                || (!parsed.pathname.startsWith(expectedPrefix) && !asset)
                || (asset && !['GET', 'HEAD'].includes(method))) {
                throw new Error('module widget fetch outside extension route prefix');
            }
            const r = await apiFetch(parsed.pathname + parsed.search, {
                method,
                headers: init.headers || {},
                body: init.body ?? undefined,
                credentials: asset ? 'omit' : 'same-origin',
                // The prefix above is checked once, before the request. A followed
                // redirect would carry the frame's method, body, headers and the
                // owner's session to whatever the hop names — another skill's
                // routes, or any authenticated host API — and the check would
                // never see it. The relay refuses the hop instead of following it.
                redirect: 'error',
                signal: controller.signal,
            });
            frame({ phase: 'headers', status: r.status, statusText: r.statusText, headers: Array.from(r.headers) });
            const reader = r.body?.getReader();
            while (reader) {
                if (!iframe.isConnected) controller.abort();
                await waitForPull();
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = bridgeChunkBuffer(value);
                frame({ phase: 'data', chunk }, [chunk]);
            }
            frame({ phase: 'end' });
        } catch (err) {
            frame({ phase: 'error', error: timedOut ? 'widget request timed out' : (err?.message || String(err)) });
        } finally {
            clearTimeout(timer);
            if (pendingRequests.get(id) === controller) pendingRequests.delete(id);
        }
    };
    const relayDownload = async (msg) => {
        const reply = sender();
        let result;
        try {
            const source = msg.source;
            const name = String(msg.name || 'download');
            if (source instanceof Blob || (typeof source === 'string' && source.startsWith('data:'))) {
                result = await downloadBlobViaHostBridge(source, name);
            } else {
                const parsed = routed(new URL(String(source || ''), assetBase || window.location.origin));
                const asset = Boolean(assetBase) && parsed.href.startsWith(assetBase);
                if (parsed.origin !== window.location.origin || (!parsed.pathname.startsWith(expectedPrefix) && !asset)) {
                    throw new Error('module widget download outside extension route prefix');
                }
                // A declared file goes to the byte owner as captured bytes; a skill route may stream.
                result = asset
                    ? await downloadBlobViaHostBridge(await readWidgetAsset(parsed.pathname + parsed.search), name)
                    : await downloadViaHostBridge(parsed.pathname + parsed.search, name, { streaming: true });
            }
        } catch (error) {
            result = { ok: false, error: error?.message || String(error) };
        }
        reply({ type: 'ouro-widget-download-result', id: msg.id, result });
    };
    const relayExternal = async (msg) => {
        const reply = sender();
        let result;
        try {
            if (disposing || !iframe.isConnected) throw new Error('widget disposed');
            if (window.navigator?.userActivation && !window.navigator.userActivation.isActive) {
                throw new Error('Opening a link requires a user action');
            }
            // Invoked synchronously from onMessage, before any await/fetch.
            result = await openExternalViaHostBridge(msg.url);
        } catch (error) {
            result = { ok: false, error: error?.message || String(error) };
        }
        reply({ type: 'ouro-widget-open-external-result', id: msg.id, result });
    };
    const onMessage = (event) => {
        if (disposed || left) return;
        const msg = event.data || {};
        if (msg.nonce !== nonce) return;
        if (msg.type === 'ouro-widget-disposed') {
            onDisposed?.();
            return;
        }
        if (msg.type === 'ouro-widget-resize') {
            if (!autoHeight) return;
            const measured = Number(msg.height);
            if (!Number.isFinite(measured) || measured <= 0) return;
            if (inline) { inline.fit(measured + WIDGET_FRAME_BORDER_RESERVE); return; }
            const nextHeight = Math.min(maxHeight, Math.max(WIDGET_FRAME_DEFAULT_HEIGHT, measured + WIDGET_FRAME_BORDER_RESERVE));
            // Stamped before the early return: that return fires exactly when the
            // frame is pinned at max_height, which is the one case these two
            // attributes exist to explain ("grew past the ceiling" vs "painted nothing").
            iframe.dataset.widgetContentHeight = String(measured);
            iframe.dataset.widgetFrameCapped = String(nextHeight >= maxHeight);
            if (nextHeight === appliedHeight) return;
            appliedHeight = nextHeight;
            setFrameHeight(iframe, appliedHeight);
            return;
        }
        if (msg.type === 'ouro-widget-error') {
            const label = msg.kind === 'csp' ? 'Widget blocked by frame policy' : 'Widget script error';
            const detail = faultText(msg.kind === 'csp' ? (msg.source || msg.message) : msg.message, 500);
            if (inline) inline.fault(`${label}: ${detail || 'unknown'}`);
            else setWidgetCardFault(mount.closest('[data-widget-key]'), `${label}: ${detail || 'unknown'}`);
            reportFault(msg);
            return;
        }
        if (msg.type === 'ouro-widget-theme') {
            if (disposing) return;
            if (msg.op === 'subscribe') startTheme();
            else if (msg.op === 'unsubscribe') stopThemeSubscription();
            return;
        }
        if (msg.type === 'ouro-widget-events') {
            if (msg.op === 'subscribe') messageHandlers?.add(onWsMessage);
            else if (msg.op === 'unsubscribe') messageHandlers?.delete(onWsMessage);
            return;
        }
        if (msg.type === 'ouro-widget-open-external') {
            relayExternal(msg);
            return;
        }
        if (msg.type === 'ouro-widget-download') {
            relayDownload(msg);
            return;
        }
        if (msg.type === 'ouro-widget-fetch-pull') {
            pendingRequests.get(msg.id)?.pull();
            return;
        }
        if (msg.type === 'ouro-widget-fetch-abort') {
            pendingRequests.get(msg.id)?.abort();
            return;
        }
        if (msg.type === 'ouro-widget-fetch') relayFetch(msg);
    };
    // A document's work ends with it: its requests, its event and theme subscriptions and its port.
    const retireDocument = () => {
        pendingRequests.forEach((controller) => controller.abort());
        pendingRequests.clear();
        messageHandlers?.delete(onWsMessage);
        stopThemeSubscription();
        if (port) { port.onmessage = null; port.close(); }
        port = null;
    };
    let helloSinceLoad = false;
    const onHello = (event) => {
        if (disposed || left || !iframe || event.source !== iframe.contentWindow) return;
        const msg = event.data || {};
        const next = event.ports?.[0];
        if (msg.type !== 'ouro-widget-hello' || !next || !accepts(msg) || (port && !rehello) || disposing) return;
        retireDocument();
        port = next;
        port.onmessage = onMessage;
        helloSinceLoad = true;
        post({ type: 'ouro-widget-ready', routeBase: `${window.location.origin}${expectedPrefix}` });
    };
    window.addEventListener('message', onHello);
    // Ordered stop with acknowledgement: post the dispose message, keep
    // answering bridged fetches the child's hooks issue meanwhile, and tear down
    // on `ouro-widget-disposed` or after WIDGET_DISPOSE_ACK_TIMEOUT_MS —
    // whichever comes first. The frame stays in its card until then; nothing
    // here blocks a page switch. Returns the settle promise; repeat calls share it.
    const dispose = () => {
        if (disposing) return disposing;
        disposing = new Promise((resolve) => {
            let ackTimer = 0;
            const finish = () => {
                if (disposed) return;
                disposed = true;
                clearTimeout(ackTimer);
                releaseLease();
                stopThemeSubscription();
                retireDocument();
                window.removeEventListener('message', onHello);
                if (iframe?.parentNode === mount) iframe.remove();
                resolve();
            };
            onDisposed = finish;
            ackTimer = setTimeout(finish, WIDGET_DISPOSE_ACK_TIMEOUT_MS);
            if (port && !left) post({ type: 'ouro-widget-dispose' });
            else finish();
        });
        return disposing;
    };
    const fault = (text) => {
        if (inline) inline.fault(text);
        else setWidgetCardFault(mount.closest('[data-widget-key]'), text);
    };
    // A load without a hello since the previous one is a document that never joined (another site, or a page
    // without the bootstrap): the previous document is gone, so its channel retires. For a module frame the
    // first load is its own srcdoc and any later one means it navigated: the mount ends at once.
    let loads = 0;
    iframe.addEventListener('load', () => {
        loads += 1;
        if (disposed || left) return;
        if (leaves && loads >= 2) {
            left = true;
            fault('The widget left its own page and was stopped');
            dispose();
            return;
        }
        if (!helloSinceLoad && port) retireDocument();
        helloSinceLoad = false;
    });
    // A route lease outlives a failed renewal and a core restart (the same URL is re-established), so the
    // running document keeps its state; a file lease ends the mount as before.
    if (lease) releaseLease = maintainWidgetAssetLease(lease, () => {
        fault(leaseEnded);
        dispose();
    }, resourceBase ? { reestablish: (signal) => reopenWidgetAssetLease(tab, lease, signal) } : {});
    return dispose;
}
