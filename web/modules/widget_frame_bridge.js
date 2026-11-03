/* The frame side of the widget bridge: one classic script for both framed mounts.

   widget_frame.js inlines `ouroWidgetFrameBridge` into a module frame's srcdoc with that mount's config; a
   `kind: "iframe"` route page loads this same file from `/api/widget-bridge.js` (its own CSP admits it by nonce
   or 'self'), and the starter at the end runs it in route mode for a page under `/api/widget-resources/<lease>/`.
   The bridge says hello to its parent once, with a MessageChannel port and its proof (the module nonce, or the
   route page's lease), and from then on sends and receives only on that port. Everything here references the
   frame's globals through `window`, so tests can run it against a stand-in. */

function ouroWidgetFrameBridge(config) {
    const { mode = 'module', nonce = '', lease = '', assetBase = '', resourceBase = '', initialTheme = '' } = config || {};
    const route = mode === 'route';
    let routeBase = (config && config.routeBase) || '';
    // The same rule as the host's safeExternalUrl (utils.js): this file cannot import it.
    const safeExternalUrl = (value) => {
        const text = String(value ?? '').trim();
        if (!text) return '#';
        try {
            const parsed = new URL(text);
            return ['http:', 'https:', 'mailto:'].includes(parsed.protocol) ? parsed.href : '#';
        } catch {
            return '#';
        }
    };
    let seq = 0;
    let disposing = false;
    let disposed = false;
    // id → in-flight bridged fetch: settles its Response on the headers
    // frame, then feeds, ends or errors that Response's body stream.
    const pending = new Map();
    const downloads = new Map();
    const externalLinks = new Map();
    const originalOpen = window.open;
    const cleanup = new Set();
    const eventListeners = new Set();
    const themeListeners = new Set();
    let theme = ['light', 'dark'].includes(initialTheme) ? initialTheme : null;
    let themeRefreshNeeded = false;
    // The document-bound channel: this document's hello hands the parent one end; everything after it travels
    // on the port, so a document that later occupies the frame can neither hear nor speak for this one.
    const channel = new window.MessageChannel();
    const port = channel.port1;
    const post = (message) => port.postMessage({ ...message, nonce });
    let resolveReady = () => {};
    const ready = new Promise((resolve) => { resolveReady = resolve; });
    const abortError = () => new DOMException('The operation was aborted.', 'AbortError');
    const onDispose = (fn) => {
        if (typeof fn !== 'function') return;
        if (disposing) { try { fn(); } catch {} return; }
        cleanup.add(fn);
    };
    const notifyTheme = (callback) => {
        try { callback(theme); } catch (error) { console.error('widget theme listener failed', error); }
    };
    const onTheme = (callback) => {
        if (disposing || disposed || typeof callback !== 'function') return () => {};
        const firstListener = themeListeners.size === 0;
        const listener = { callback };
        themeListeners.add(listener);
        if (theme && !themeRefreshNeeded) notifyTheme(callback);
        if (firstListener && themeListeners.size && !disposed) {
            post({ type: 'ouro-widget-theme', op: 'subscribe' });
        }
        return () => {
            if (!themeListeners.delete(listener)) return;
            if (!themeListeners.size && !disposed) {
                themeRefreshNeeded = true;
                post({ type: 'ouro-widget-theme', op: 'unsubscribe' });
            }
        };
    };
    // Ordered dispose: every hook runs first (async hooks are awaited and
    // the bridge keeps streaming for them), then the parent gets the
    // acknowledgement, and only then are pending fetches rejected, open
    // body streams errored, event listeners dropped and the listener
    // removed. The parent bounds the whole wait on its side.
    const dispose = async () => {
        if (disposing) return;
        disposing = true;
        const hooks = Array.from(cleanup);
        cleanup.clear();
        await Promise.allSettled(hooks.map((fn) => Promise.resolve().then(fn)));
        window.document?.removeEventListener('click', clickDownload);
        window.document?.removeEventListener('click', clickExternal);
        window.open = originalOpen;
        blobUrls.clear();
        if (createUrl) urlApi.createObjectURL = createUrl;
        if (revokeUrl) urlApi.revokeObjectURL = revokeUrl;
        post({ type: 'ouro-widget-disposed' });
        disposed = true;
        pending.forEach((item) => item.fail(new Error('widget disposed')));
        pending.clear();
        downloads.forEach(({ reject }) => reject(new Error('widget disposed')));
        downloads.clear();
        externalLinks.forEach(({ reject }) => reject(new Error('widget disposed')));
        externalLinks.clear();
        eventListeners.clear();
        themeListeners.clear();
        theme = null;
        themeRefreshNeeded = false;
        port.onmessage = null;
        window.removeEventListener('error', onError);
        window.removeEventListener('unhandledrejection', onRejection);
        window.removeEventListener('securitypolicyviolation', onCsp);
    };
    const onMessage = (event) => {
        const msg = event.data || {};
        if (msg.nonce !== nonce) return;
        if (msg.type === 'ouro-widget-ready') {
            // A route page learns its skill's route prefix here; resourceUrl maps absolute route URLs from then on.
            if (route && typeof msg.routeBase === 'string') routeBase = msg.routeBase;
            resolveReady(true);
            return;
        }
        if (msg.type === 'ouro-widget-dispose') {
            dispose();
            return;
        }
        // The bridge answers during the hooks; frames are refused only once disposed.
        if (disposed) return;
        if (msg.type === 'ouro-widget-theme') {
            if (disposing || !themeListeners.size || !['light', 'dark'].includes(msg.theme)) return;
            if (msg.theme === theme && !themeRefreshNeeded) return;
            theme = msg.theme;
            themeRefreshNeeded = false;
            Array.from(themeListeners).forEach((listener) => {
                if (themeListeners.has(listener)) notifyTheme(listener.callback);
            });
            return;
        }
        if (msg.type === 'ouro-widget-event') {
            const detail = { type: String(msg.event || ''), data: msg.data };
            eventListeners.forEach((callback) => {
                try { callback(detail); } catch (err) { console.error('widget event listener failed', err); }
            });
            return;
        }
        if (msg.type === 'ouro-widget-open-external-result') {
            const item = externalLinks.get(msg.id);
            if (!item) return;
            externalLinks.delete(msg.id);
            if (msg.result?.ok || msg.result?.degraded) item.resolve(msg.result);
            else item.reject(new Error(msg.result?.error || 'widget link failed'));
            return;
        }
        if (msg.type === 'ouro-widget-download-result') {
            const item = downloads.get(msg.id);
            if (!item) return;
            downloads.delete(msg.id);
            if (msg.result?.ok) item.resolve(msg.result);
            else item.reject(new Error(msg.result?.error || 'widget download failed'));
            return;
        }
        if (msg.type !== 'ouro-widget-fetch-chunk') return;
        pending.get(msg.id)?.frame(msg);
    };
    const request = (url, init = {}) => new Promise((resolve, reject) => {
        if (disposed) {
            reject(new Error('widget disposed'));
            return;
        }
        const signal = init.signal || null;
        if (signal?.aborted) {
            reject(abortError());
            return;
        }
        const id = ++seq;
        const method = String(init.method || 'GET').toUpperCase();
        let settled = false;
        let body = null;
        let pulled = null;
        const finish = () => {
            pending.delete(id);
            signal?.removeEventListener('abort', onAbort);
            pulled?.();
            pulled = null;
        };
        const fail = (error) => {
            finish();
            if (!settled) {
                settled = true;
                reject(error);
                return;
            }
            try { body?.error(error); } catch {}
        };
        const cancel = () => {
            post({ type: 'ouro-widget-fetch-abort', id });
            finish();
        };
        const onAbort = () => {
            post({ type: 'ouro-widget-fetch-abort', id });
            fail(abortError());
        };
        const frame = (msg) => {
            if (msg.phase === 'headers') {
                if (settled) return;
                settled = true;
                // A Response refuses a body for HEAD and 204/205/304.
                const nullBody = method === 'HEAD' || [204, 205, 304].includes(Number(msg.status));
                const stream = nullBody ? null : new ReadableStream({
                    start(controller) { body = controller; },
                    pull() {
                        return new Promise((done) => {
                            pulled = done;
                            post({ type: 'ouro-widget-fetch-pull', id });
                        });
                    },
                    cancel,
                }, { highWaterMark: 0 });
                try {
                    resolve(new Response(stream, {
                        status: Number(msg.status) || 200,
                        statusText: String(msg.statusText || ''),
                        headers: Array.isArray(msg.headers) ? msg.headers : [],
                    }));
                } catch (error) {
                    cancel();
                    reject(error);
                    return;
                }
                if (nullBody) finish();
                return;
            }
            if (msg.phase === 'data') {
                try { body?.enqueue(new Uint8Array(msg.chunk)); } catch {}
                pulled?.();
                pulled = null;
                return;
            }
            if (msg.phase === 'end') {
                finish();
                try { body?.close(); } catch {}
                return;
            }
            if (msg.phase === 'error') fail(new Error(String(msg.error || 'widget fetch failed')));
        };
        pending.set(id, { frame, fail });
        signal?.addEventListener('abort', onAbort, { once: true });
        try {
            post({
                type: 'ouro-widget-fetch',
                id,
                // A route page's relative URL is its own: resolve it against its document, which lives under the lease.
                url: route ? new URL(String(url || ''), window.document.baseURI).href : String(url || ''),
                init: {
                    method,
                    headers: Array.from(new Headers(init.headers || {})),
                    body: init.body ?? null,
                    timeoutMs: init.timeoutMs ?? null,
                },
            });
        } catch (error) {
            fail(error);
        }
    });
    // The skill's own namespaced WebSocket events, forwarded by the
    // parent while at least one listener is registered.
    const onEvent = (callback) => {
        if (typeof callback !== 'function' || disposed) return () => {};
        if (!eventListeners.size) post({ type: 'ouro-widget-events', op: 'subscribe' });
        eventListeners.add(callback);
        return () => {
            if (!eventListeners.delete(callback)) return;
            if (!eventListeners.size && !disposed) post({ type: 'ouro-widget-events', op: 'unsubscribe' });
        };
    };
    // Fault channel: a script that throws at top level, an unhandled
    // rejection or a CSP refusal otherwise paints a blank frame while the
    // card still says Running. Bounded (10 posts, deduped on kind+message)
    // so a throwing animation loop cannot flood the parent. All three
    // listeners go on window, never document: securitypolicyviolation
    // bubbles to window, and not every host of this bridge has a document.
    const seenFaults = new Set();
    let faultCount = 0;
    const fault = (kind, message, source, line) => {
        const text = String(message ?? '').slice(0, 500);
        const key = kind + '\u0000' + text;
        if (faultCount >= 10 || seenFaults.has(key)) return;
        seenFaults.add(key);
        faultCount += 1;
        post({ type: 'ouro-widget-error', kind, message: text, source: String(source ?? '').slice(0, 200), line: Number(line) || 0 });
    };
    const onError = (event) => fault('error', event.message || event.error, event.filename, event.lineno);
    const onRejection = (event) => fault('rejection', event.reason, '', 0);
    const onCsp = (event) => fault('csp', event.violatedDirective || 'blocked', event.blockedURI, event.lineNumber);
    // A route page keeps its own fetch, links, popups and fault handling; the module frame's document is the host's.
    if (!route) {
        window.addEventListener('error', onError);
        window.addEventListener('unhandledrejection', onRejection);
        window.addEventListener('securitypolicyviolation', onCsp);
        window.fetch = request;
        window.__ouroWidgetReport = (message) => post(message);
    }
    port.onmessage = onMessage;
    window.__ouroWidgetOnDispose = onDispose;
    const download = (name, source) => new Promise((resolve, reject) => {
        if (disposed) { reject(new Error('widget disposed')); return; }
        const id = ++seq;
        downloads.set(id, { resolve, reject });
        try { post({ type: 'ouro-widget-download', id, name: String(name || 'download'), source: typeof source === 'string' ? (blobUrls.get(source) || source) : source }); }
        catch (error) { downloads.delete(id); reject(error); }
    });
    // Remember the Blob behind a frame-owned URL: the opaque frame's
    // URL cannot be fetched by the host, and its CSP forbids script IO.
    const blobUrls = new Map();
    const urlApi = window.URL;
    const createUrl = urlApi?.createObjectURL?.bind(urlApi);
    const revokeUrl = urlApi?.revokeObjectURL?.bind(urlApi);
    if (createUrl && !route) urlApi.createObjectURL = (source) => {
        const url = createUrl(source);
        if (source instanceof Blob) blobUrls.set(url, source);
        return url;
    };
    if (revokeUrl && !route) urlApi.revokeObjectURL = (url) => {
        blobUrls.delete(String(url));
        return revokeUrl(url);
    };
    const clickDownload = (event) => {
        if (event.defaultPrevented || event.button > 0) return;
        const anchor = event.target?.closest?.('a[download]');
        if (!anchor) return;
        const href = String(anchor.href || '');
        const source = blobUrls.get(href) || href;
        if (!blobUrls.has(href) && !href.startsWith('data:') && !(routeBase && href.startsWith(routeBase))
            && !(assetBase && href.startsWith(assetBase)) && !(resourceBase && href.startsWith(resourceBase))) return;
        event.preventDefault();
        download(anchor.download, source).catch((error) => fault('error', error.message, '', 0));
    };
    const openExternal = (url, trustedAnchor = false) => new Promise((resolve, reject) => {
        if (disposing || disposed) { reject(new Error('widget disposed')); return; }
        const activation = window.navigator?.userActivation;
        if (activation ? !activation.isActive : !trustedAnchor) {
            reject(new Error('Opening a link requires a user action'));
            return;
        }
        const target = safeExternalUrl(url);
        if (target === '#') { reject(new Error('Unsupported external link')); return; }
        const id = ++seq;
        externalLinks.set(id, { resolve, reject });
        try { post({ type: 'ouro-widget-open-external', id, url: target }); }
        catch (error) { externalLinks.delete(id); reject(error); }
    });
    const clickExternal = (event) => {
        if (event.defaultPrevented || event.button > 0 || !event.isTrusted) return;
        const anchor = event.target?.closest?.('a[href]');
        if (!anchor || anchor.hasAttribute('download')) return;
        const target = safeExternalUrl(anchor.getAttribute('href'));
        if (target === '#' || (routeBase && target.startsWith(routeBase))
            || (assetBase && target.startsWith(assetBase)) || (resourceBase && target.startsWith(resourceBase))) return;
        event.preventDefault();
        openExternal(target, true).catch((error) => fault('error', error.message, '', 0));
    };
    if (!route) {
        window.open = (url) => {
            openExternal(url).catch((error) => fault('error', error.message, '', 0));
            return null;
        };
        window.document?.addEventListener('click', clickExternal);
        window.document?.addEventListener('click', clickDownload);
    }
    // A classic frame's own elements (img, audio, video, @font-face) load the skill's routes natively
    // through this mount's GET/HEAD lease: resourceUrl maps a route URL or a relative path onto it,
    // and leaves any other URL as it is (the document policy decides).
    const resourceUrl = (url) => {
        const parsed = new URL(String(url ?? ''), resourceBase || routeBase);
        if (resourceBase && routeBase && parsed.href.startsWith(routeBase)) {
            return resourceBase + parsed.href.slice(routeBase.length);
        }
        return parsed.href;
    };
    window.OuroborosWidget = { fetch: request, onEvent, onTheme, download, openExternal: (url) => openExternal(url),
        resourceBase, resourceUrl, ready };
    // The proof the parent checks: a module frame's nonce lives only in its srcdoc; a route page's lease lives only
    // in its own URL path, which neither a redirect nor a navigation carries to another document.
    window.parent.postMessage({ type: 'ouro-widget-hello', ...(route ? { lease } : { nonce }) }, '*', [channel.port2]);
}

// Module import (the host page and tests) publishes the function; a route page's classic load starts it.
globalThis.ouroWidgetFrameBridge = ouroWidgetFrameBridge;
(() => {
    const here = typeof window === 'object' ? window : null;
    if (!here || here.parent === here || here.OuroborosWidget) return;
    const match = /^\/api\/widget-resources\/([A-Za-z0-9_-]{43})\//.exec(here.location?.pathname || '');
    if (match) {
        ouroWidgetFrameBridge({ mode: 'route', lease: match[1],
            resourceBase: `${here.location.origin}/api/widget-resources/${match[1]}/` });
    }
})();
