/* Framed widget bootstrap scripts. The parent remains the route and lifecycle owner. */

// The child bridge is one classic file (also served to route pages); importing it publishes the function.
import './widget_frame_bridge.js';

// The parent hands every streamed body chunk to the frame as a transferred
// ArrayBuffer. A reader's Uint8Array may be a window onto a larger buffer, so
// transfer exactly the bytes the view covers and nothing beside them.
export function bridgeChunkBuffer(view) {
    if (view instanceof ArrayBuffer) return view;
    if (view.byteOffset === 0 && view.byteLength === view.buffer.byteLength) return view.buffer;
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

// Child side of the one bridge grammar. The child posts one ouro-widget-hello {nonce | lease} to its
// parent window with a MessageChannel port; every message below then travels on that port only:
//   child → parent  ouro-widget-fetch {id, url, init} · ouro-widget-fetch-abort {id}
//                   ouro-widget-fetch-pull {id} · ouro-widget-download {id, name, source}
//                   ouro-widget-open-external {id, url}
//                   ouro-widget-events {op: subscribe | unsubscribe} · ouro-widget-disposed
//                   ouro-widget-theme {op: subscribe | unsubscribe}
//                   ouro-widget-error {kind: error | rejection | csp, message, source, line}
//                   ouro-widget-resize {height}
//   parent → child  ouro-widget-ready {routeBase} · ouro-widget-fetch-chunk {id, phase: headers | data | end | error, …}
//                   ouro-widget-open-external-result {id, result}
//                   ouro-widget-event {event, data} · ouro-widget-theme {theme: light | dark}
//                   · ouro-widget-dispose
// Every bridged fetch streams: the child rebuilds a real Response over a
// ReadableStream fed by `data` frames (binary by default), so text/json/blob
// and incremental body reads all work. No default timeout — `init.timeoutMs`
// is the author's opt-in bound; `init.signal` aborts through the parent.
export function moduleBridgeScript(nonce, routeBase = '', initialTheme = '', assetBase = '', resourceBase = '') {
    const config = { mode: 'module', nonce, routeBase, assetBase, resourceBase, initialTheme };
    return `(${globalThis.ouroWidgetFrameBridge})(${JSON.stringify(config)});`;
}

export function moduleResizeScript(frameFloor, maxHeight, borderReserve) {
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
                // Through the bridge's channel (installed by the bootstrap that runs first).
                window.__ouroWidgetReport?.({ type: 'ouro-widget-resize', height });
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
