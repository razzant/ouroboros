/* Framed widget bootstrap scripts. The parent remains the route and lifecycle owner. */

import { safeExternalUrl } from './utils.js';

// The parent hands every streamed body chunk to the frame as a transferred
// ArrayBuffer. A reader's Uint8Array may be a window onto a larger buffer, so
// transfer exactly the bytes the view covers and nothing beside them.
export function bridgeChunkBuffer(view) {
    if (view instanceof ArrayBuffer) return view;
    if (view.byteOffset === 0 && view.byteLength === view.buffer.byteLength) return view.buffer;
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

// Child side of the one bridge grammar (nonce-bound, parent ⇄ frame):
//   child → parent  ouro-widget-fetch {id, url, init} · ouro-widget-fetch-abort {id}
//                   ouro-widget-fetch-pull {id} · ouro-widget-download {id, name, source}
//                   ouro-widget-open-external {id, url}
//                   ouro-widget-events {op: subscribe | unsubscribe} · ouro-widget-disposed
//                   ouro-widget-theme {op: subscribe | unsubscribe}
//                   ouro-widget-error {kind: error | rejection | csp, message, source, line}
//   parent → child  ouro-widget-fetch-chunk {id, phase: headers | data | end | error, …}
//                   ouro-widget-open-external-result {id, result}
//                   ouro-widget-event {event, data} · ouro-widget-theme {theme: light | dark}
//                   · ouro-widget-dispose
// Every bridged fetch streams: the child rebuilds a real Response over a
// ReadableStream fed by `data` frames (binary by default), so text/json/blob
// and incremental body reads all work. No default timeout — `init.timeoutMs`
// is the author's opt-in bound; `init.signal` aborts through the parent.
export function moduleBridgeScript(nonce, routeBase = '', initialTheme = '') {
    return `
        (() => {
            const nonce = ${JSON.stringify(nonce)};
            const routeBase = ${JSON.stringify(routeBase)};
            const initialTheme = ${JSON.stringify(initialTheme)};
            const safeExternalUrl = (${safeExternalUrl.toString()});
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
            const post = (message) => window.parent.postMessage({ ...message, nonce }, '*');
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
                window.removeEventListener('message', onMessage);
                window.removeEventListener('error', onError);
                window.removeEventListener('unhandledrejection', onRejection);
                window.removeEventListener('securitypolicyviolation', onCsp);
            };
            const onMessage = (event) => {
                if (event.source !== window.parent) return;
                const msg = event.data || {};
                if (msg.nonce !== nonce) return;
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
                        url: String(url || ''),
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
            window.addEventListener('error', onError);
            window.addEventListener('unhandledrejection', onRejection);
            window.addEventListener('securitypolicyviolation', onCsp);
            window.addEventListener('message', onMessage);
            window.__ouroWidgetOnDispose = onDispose;
            window.fetch = request;
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
            if (createUrl) urlApi.createObjectURL = (source) => {
                const url = createUrl(source);
                if (source instanceof Blob) blobUrls.set(url, source);
                return url;
            };
            if (revokeUrl) urlApi.revokeObjectURL = (url) => {
                blobUrls.delete(String(url));
                return revokeUrl(url);
            };
            const clickDownload = (event) => {
                if (event.defaultPrevented || event.button > 0) return;
                const anchor = event.target?.closest?.('a[download]');
                if (!anchor) return;
                const href = String(anchor.href || '');
                const source = blobUrls.get(href) || href;
                if (!blobUrls.has(href) && !href.startsWith('data:') && !(routeBase && href.startsWith(routeBase))) return;
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
                if (target === '#' || (routeBase && target.startsWith(routeBase))) return;
                event.preventDefault();
                openExternal(target, true).catch((error) => fault('error', error.message, '', 0));
            };
            window.open = (url) => {
                openExternal(url).catch((error) => fault('error', error.message, '', 0));
                return null;
            };
            window.document?.addEventListener('click', clickExternal);
            window.document?.addEventListener('click', clickDownload);
            window.OuroborosWidget = { fetch: request, onEvent, onTheme, download, openExternal: (url) => openExternal(url) };
        })();
    `;
}

export function moduleResizeScript(nonce, frameFloor, maxHeight, borderReserve) {
    return `
        (() => {
            const root = document.getElementById('root');
            const verticalOverflowState = [document.documentElement, document.body]
                .filter(Boolean)
                .map((element) => ({
                    element,
                    value: element.style.getPropertyValue('overflow-y'),
                    priority: element.style.getPropertyPriority('overflow-y'),
                }));
            let suppressingVerticalOverflow = false;
            let lastHeight = 0;
            let resizeFrame = null;
            let disposed = false;
            const setVerticalOverflowSuppressed = (suppressed) => {
                if (suppressed === suppressingVerticalOverflow) return;
                suppressingVerticalOverflow = suppressed;
                verticalOverflowState.forEach(({ element, value, priority }) => {
                    if (suppressed) element.style.setProperty('overflow-y', 'hidden', 'important');
                    else if (value) element.style.setProperty('overflow-y', value, priority);
                    else element.style.removeProperty('overflow-y');
                });
            };
            setVerticalOverflowSuppressed(true);
            const report = () => {
                if (!root || disposed) return;
                const box = root.getBoundingClientRect();
                const body = document.body;
                const bodyTop = body?.getBoundingClientRect().top || 0;
                // The root's bottom edge captures collapsed child margins; body
                // bottom padding and border complete the measured body box. This
                // also avoids treating a fixed 100vh body as small-module content.
                const bodyStyle = body ? getComputedStyle(body) : null;
                const paddingBottom = Number.parseFloat(bodyStyle?.paddingBottom);
                const borderBottom = Number.parseFloat(bodyStyle?.borderBottomWidth);
                const bodyBottomSpacing = Math.max(0,
                    (Number.isFinite(paddingBottom) ? paddingBottom : 0)
                    + (Number.isFinite(borderBottom) ? borderBottom : 0));
                const bodyHeight = body?.scrollHeight || 0;
                const bodyClientHeight = body?.clientHeight || 0;
                const fixedViewportBody = bodyStyle
                    && Math.abs((parseFloat(bodyStyle.height) || 0) - window.innerHeight) <= 1;
                const bodyContentHeight = !fixedViewportBody || bodyHeight > bodyClientHeight + 1
                    ? bodyHeight
                    : 0;
                const contentHeight = Math.max(
                    root.scrollHeight,
                    box.height,
                    box.bottom - bodyTop + bodyBottomSpacing,
                    bodyContentHeight,
                );
                const height = Math.ceil(contentHeight);
                const outerHeight = Math.min(
                    ${JSON.stringify(maxHeight)},
                    Math.max(
                        ${JSON.stringify(frameFloor)},
                        height + ${JSON.stringify(borderReserve)},
                    ),
                );
                setVerticalOverflowSuppressed(outerHeight < ${JSON.stringify(maxHeight)});
                if (!height || height === lastHeight) return;
                lastHeight = height;
                window.parent.postMessage({
                    type: 'ouro-widget-resize',
                    nonce: ${JSON.stringify(nonce)},
                    height,
                }, '*');
            };
            // report() toggles overflow-y, which can resize the observed root.
            // Inside ResizeObserver delivery that resize goes undelivered and the
            // browser fires a loop error on window, which the bridge's fault
            // channel reports as a widget script error. A microtask would still
            // run inside delivery; an animation frame runs before the next
            // observation pass, coalesces bursts into one measurement and is
            // cancelled on dispose. Hidden pages pause both alike.
            const scheduleReport = () => {
                if (disposed || resizeFrame !== null) return;
                resizeFrame = window.requestAnimationFrame(() => {
                    resizeFrame = null;
                    report();
                });
            };
            const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(scheduleReport) : null;
            if (observer && root) observer.observe(root);
            const onLoad = () => report();
            window.addEventListener('load', onLoad, { once: true });
            window.__ouroWidgetOnDispose?.(() => {
                disposed = true;
                if (resizeFrame !== null) window.cancelAnimationFrame(resizeFrame);
                resizeFrame = null;
                observer?.disconnect();
                window.removeEventListener('load', onLoad);
                setVerticalOverflowSuppressed(false);
            });
            report();
        })();
    `;
}
