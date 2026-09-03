/* Framed widget bootstrap scripts. The parent remains the route and lifecycle owner. */

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
//                   ouro-widget-events {op: subscribe | unsubscribe} · ouro-widget-disposed
//   parent → child  ouro-widget-fetch-chunk {id, phase: headers | data | end | error, …}
//                   ouro-widget-event {event, data} · ouro-widget-dispose
// Every bridged fetch streams: the child rebuilds a real Response over a
// ReadableStream fed by `data` frames (binary by default), so text/json/blob
// and incremental body reads all work. No default timeout — `init.timeoutMs`
// is the author's opt-in bound; `init.signal` aborts through the parent.
export function moduleBridgeScript(nonce) {
    return `
        (() => {
            const nonce = ${JSON.stringify(nonce)};
            let seq = 0;
            let disposing = false;
            let disposed = false;
            // id → in-flight bridged fetch: settles its Response on the headers
            // frame, then feeds, ends or errors that Response's body stream.
            const pending = new Map();
            const cleanup = new Set();
            const eventListeners = new Set();
            const post = (message) => window.parent.postMessage({ ...message, nonce }, '*');
            const abortError = () => new DOMException('The operation was aborted.', 'AbortError');
            const onDispose = (fn) => {
                if (typeof fn !== 'function') return;
                if (disposing) { try { fn(); } catch {} return; }
                cleanup.add(fn);
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
                post({ type: 'ouro-widget-disposed' });
                disposed = true;
                pending.forEach((item) => item.fail(new Error('widget disposed')));
                pending.clear();
                eventListeners.clear();
                window.removeEventListener('message', onMessage);
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
                if (msg.type === 'ouro-widget-event') {
                    const detail = { type: String(msg.event || ''), data: msg.data };
                    eventListeners.forEach((callback) => {
                        try { callback(detail); } catch (err) { console.error('widget event listener failed', err); }
                    });
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
                const finish = () => {
                    pending.delete(id);
                    signal?.removeEventListener('abort', onAbort);
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
                            cancel,
                        });
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
            window.addEventListener('message', onMessage);
            window.__ouroWidgetOnDispose = onDispose;
            window.fetch = request;
            window.OuroborosWidget = { fetch: request, onEvent };
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
                if (!root) return;
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
            const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(report) : null;
            if (observer && root) observer.observe(root);
            const onLoad = () => report();
            window.addEventListener('load', onLoad, { once: true });
            window.__ouroWidgetOnDispose?.(() => {
                observer?.disconnect();
                window.removeEventListener('load', onLoad);
                setVerticalOverflowSuppressed(false);
            });
            report();
        })();
    `;
}
