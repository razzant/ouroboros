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
//                   ouro-widget-resize {height, held?, fit?}
//   parent → child  ouro-widget-ready {routeBase} · ouro-widget-fetch-chunk {id, phase: headers | data | end | error, …}
//                   ouro-widget-open-external-result {id, result}
//                   ouro-widget-event {event, data} · ouro-widget-theme {theme: light | dark}
//                   · ouro-widget-viewport {floor, ceiling, frame?, fit?} (a Widgets card's self-sizing module)
//                   · ouro-widget-dispose
// Every bridged fetch streams: the child rebuilds a real Response over a
// ReadableStream fed by `data` frames (binary by default), so text/json/blob
// and incremental body reads all work. No default timeout — `init.timeoutMs`
// is the author's opt-in bound; `init.signal` aborts through the parent.
export function moduleBridgeScript(nonce, routeBase = '', initialTheme = '', assetBase = '', resourceBase = '') {
    const config = { mode: 'module', nonce, routeBase, assetBase, resourceBase, initialTheme };
    return `(${globalThis.ouroWidgetFrameBridge})(${JSON.stringify(config)});`;
}

// The size reporter of a self-sizing module. `frameFloor` and `maxHeight` are the outer frame's floor and ceiling as
// the parent applies them: the document's vertical scrollbar stays hidden below the ceiling, where the frame grows
// to fit, and is released at it, where the frame scrolls. A Widgets card's parent moves both with
// `ouro-widget-viewport` (the owner's height, Show all, Back); the reporter then decides again, even for a height it
// already reported. A frame in a message keeps its baked values: nothing is sent to it. `wholeDocument` (a Widgets
// card, whose frame has no 320 px floor under short content) measures the whole document, margins included, as it is
// laid out with its scrollbar hidden, and again whenever its frame resizes (see `measure`), and stops a frame that
// following its content would never end (see `followsItsFrame`); `heldExcess` is that stop remembered from this
// card's previous document.
export function moduleResizeScript(frameFloor, maxHeight, borderReserve, { wholeDocument = false, heldExcess = 0 } = {}) {
    return `
        (() => {
            const root = document.getElementById('root');
            const wholeDocument = ${JSON.stringify(Boolean(wholeDocument))};
            let floor = ${JSON.stringify(frameFloor)};
            let ceiling = ${JSON.stringify(maxHeight)};
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
            // A board document's changes (below), whose records of the reporter's own overflow writes are dropped.
            let mutations = null;
            // Reported again for such a change only when what the document measures moved since the last report: the
            // reporter's own scrollbar writes, and a widget answering them without moving anything, start no round.
            let seenMeasure = '';
            let changeFrame = null;
            const ownWrite = (record) => record.type === 'attributes' && record.attributeName === 'style'
                && (record.target === document.documentElement || record.target === document.body);
            const setVerticalOverflowSuppressed = (suppressed) => {
                if (suppressed === suppressingVerticalOverflow) return;
                suppressingVerticalOverflow = suppressed;
                verticalOverflowState.forEach(({ element, value, priority }) => {
                    if (suppressed) element.style.setProperty('overflow-y', 'hidden', 'important');
                    else if (value) element.style.setProperty('overflow-y', value, priority);
                    else element.style.removeProperty('overflow-y');
                });
                if (mutations?.takeRecords().some((record) => !ownWrite(record))) scheduleChange();
            };
            // A board frame keeps its loading height until the module's entry has run (the host marks it right after
            // the entry, a module entry's top-level await included) or its page has loaded, whichever is first (a
            // module entry may wait in its top-level await long after): before that its document is the host's empty
            // page or a widget still starting, not a measure of it, and scrolls in that height. A frame in a message
            // keeps its 320 px floor and reports from the start.
            let entryRan = !wholeDocument;
            setVerticalOverflowSuppressed(entryRan);
            // Nor is a document with nothing in it yet a measure: a classic entry that returned before drawing (its
            // await), an empty one, one that threw. Its loading height stays until something is drawn; once it has
            // been, anything may empty out again and is measured as such.
            let drawnYet = !wholeDocument;
            const px = (value) => {
                const number = Number.parseFloat(value);
                return Number.isFinite(number) ? number : 0;
            };
            // Whether the body's own height or min-height is set (100vh, 100%, a length). Typed computed values say so
            // where the engine has them (unset: auto, or a zero min-height); elsewhere every body counts as sized.
            const sizedBody = (body) => {
                const typed = body.computedStyleMap?.();
                if (!typed) return true;
                const unset = (value) => value?.value === 'auto' || value?.value === 0;
                return !unset(typed.get('height')) || !unset(typed.get('min-height'));
            };
            // A box's bottom edge with what overflows it there (its scrollable overflow: an absolutely placed box
            // inside it, text past a height it was given), unless it clips or scrolls what it holds.
            const extentOf = (element, style) => {
                const box = element.getBoundingClientRect();
                return style.overflowY === 'visible'
                    ? Math.max(box.bottom, box.top + element.clientTop + element.scrollHeight) : box.bottom;
            };
            const inFlow = (style) => style.display !== 'none' && style.position !== 'absolute' && style.position !== 'fixed';
            // An in-flow box with the bottom margin it ends with: its own, or that of its last in-flow descendant, each
            // counted at its own box (a margin that collapses through the boxes around it lies below them).
            const flowEnd = (element, style) => {
                let end = element.getBoundingClientRect().bottom + Math.max(0, px(style.marginBottom));
                for (let box = element, boxStyle = style; boxStyle.overflowY === 'visible';) {
                    let last = box.lastElementChild;
                    let lastStyle = null;
                    for (; last; last = last.previousElementSibling) {
                        lastStyle = getComputedStyle(last);
                        if (inFlow(lastStyle)) break;
                    }
                    if (!last) break;
                    box = last;
                    boxStyle = lastStyle;
                    end = Math.max(end, box.getBoundingClientRect().bottom + Math.max(0, px(boxStyle.marginBottom)));
                }
                return end;
            };
            // A box placed absolutely against the body or the page (no positioned box between it and the body) lies
            // where that containing block puts it, outside the flow and the overflow of the boxes around it in the
            // tree: a control at the bottom of a 500 px body is at 500. It counts there, with what overflows it (its
            // own placed boxes). A box placed inside any other positioned box is in that box's overflow; a fixed box
            // is the viewport's, not content; an SVG drawing's parts are its own geometry.
            const placedBottom = (body) => {
                let bottom = 0;
                const pending = Array.from(body.children);
                while (pending.length) {
                    const element = pending.pop();
                    const style = getComputedStyle(element);
                    if (style.display === 'none') continue;
                    if (style.position === 'absolute') bottom = Math.max(bottom, extentOf(element, style));
                    else if ((style.display === 'contents' || style.position === 'static')
                        && element.namespaceURI !== 'http://www.w3.org/2000/svg') pending.push(...element.children);
                }
                return bottom;
            };
            // Whether the body holds anything drawn: a box with height, overflow or a placed box of its own, or text.
            const hasDrawn = (body) => placedBottom(body) > 0 || Array.from(body.childNodes).some((node) => (
                node.nodeType === 3 ? node.textContent.trim() !== ''
                    : node.nodeType === 1 && (node.getBoundingClientRect().height > 0 || node.scrollHeight > 0)));
            // What the body holds, where its layout draws it (the author's sizes, containing blocks, flex and
            // positioning as rendered; nothing is changed to measure): the bottom of its content box, which a body
            // with no height of its own ends at; for a body sized by its own height or min-height (100vh, 100%, a
            // length), which is not content, the bottom of each in-flow box and text it holds with its margins; and
            // what overflows them, and every box placed against the body or the page.
            const heldBottom = (body, bodyStyle) => {
                const sized = sizedBody(body);
                let bottom = sized ? 0
                    : body.getBoundingClientRect().bottom - px(bodyStyle.paddingBottom) - px(bodyStyle.borderBottomWidth);
                for (const node of body.childNodes) {
                    if (node.nodeType === 3 && sized && node.textContent.trim()) {
                        const text = document.createRange();
                        text.selectNodeContents(node);
                        bottom = Math.max(bottom, text.getBoundingClientRect().bottom);
                    }
                    if (node.nodeType !== 1) continue;
                    const style = getComputedStyle(node);
                    if (!inFlow(style)) continue;
                    bottom = Math.max(bottom, extentOf(node, style), sized ? flowEnd(node, style) : 0);
                }
                return Math.max(bottom, placedBottom(body));
            };
            const measure = () => {
                const box = root.getBoundingClientRect();
                const body = document.body;
                const bodyTop = body?.getBoundingClientRect().top || 0;
                // The root's bottom edge captures collapsed child margins; body
                // bottom padding and border complete the measured body box.
                const bodyStyle = body ? getComputedStyle(body) : null;
                const bodyBottomSpacing = Math.max(0, px(bodyStyle?.paddingBottom) + px(bodyStyle?.borderBottomWidth));
                const rootHeight = Math.max(root.scrollHeight, box.height, box.bottom - bodyTop + bodyBottomSpacing);
                // A board document counts whole, from its top to the body's bottom margin, as it is with its scrollbar
                // hidden (the body then holds its children's margins): what the body holds, then its bottom padding,
                // border and margin.
                if (wholeDocument && body) {
                    const bottomMargin = Math.max(0, px(bodyStyle.marginBottom));
                    return Math.ceil(Math.max(rootHeight,
                        heldBottom(body, bodyStyle) + bodyBottomSpacing + bottomMargin + (window.scrollY || 0)));
                }
                // This also avoids treating a fixed 100vh body as small-module content.
                const bodyHeight = body?.scrollHeight || 0;
                const fixedViewportBody = bodyStyle && Math.abs(px(bodyStyle.height) - window.innerHeight) <= 1;
                const bodyContent = !fixedViewportBody || bodyHeight > (body?.clientHeight || 0) + 1 ? bodyHeight : 0;
                return Math.ceil(Math.max(rootHeight, bodyContent));
            };
            const reserve = ${JSON.stringify(borderReserve)};
            const fits = (height) => Math.min(ceiling, Math.max(floor, height + reserve)) < ceiling;
            window.__ouroWidgetEntryRan = () => {
                if (entryRan) return;
                entryRan = true;
                report();
            };
            // The frame stops growing when growing has stopped helping: a document laid out against its own window
            // (100vh with margins outside it, 110vh, a bar at the window's bottom) asks for at least as much extra in
            // every frame it is given, up to the ceiling otherwise. The sign is measured, not a CSS cause: the frame
            // has become exactly the height last asked for, below the ceiling (an echo), the document asks for more,
            // no less than at the last echo (growing did not bring it nearer), and did so one animation frame later
            // too, unchanged. Then the reporter stops asking, says so, and lets the document scroll where it stands: a
            // conservative fallback, so what it places below the frame stays reachable. An echo at which the
            // document fits ends the episode; a different extra at a held height (content that really changed), new
            // bounds or another frame height take the hold off, and it measures on; a held height is also
            // remembered by the card (heldExcess, the same extra found again by its next document), so returning
            // to the board does not start the growth over. A document whose extra shrinks at each echo converges and
            // is followed to its height, an eased box among them, whose extra shrinks as it slows. Content that changes
            // in the same frame as an echo may look the same to this check: it is at worst held one step short, and
            // scrolls.
            const near = (first, second) => Math.abs(first - second) <= 1;
            // Whether something in the document may still be moving it: a finite transition or animation is running
            // (one that never ends, a spinner, does not count). A stop made meanwhile stands while it runs, the frame
            // still and its document scrolling, and is looked at once more at the first layout cue after nothing moves
            // (its end is one): the document then gets the height it asks for, once (probing), and stops there for
            // good if it still asks for more. An eased box, whose tail asks for a pixel or two at each echo as a
            // follower does, so ends whole; an animation that moves nothing costs a follower one step.
            const settling = () => (document.getAnimations?.() || []).some((animation) => animation.playState === 'running'
                && animation.effect?.getTiming?.().iterations !== Infinity);
            let held = null;
            let rememberedHold = ${JSON.stringify(Number(heldExcess) || 0)};
            let seenInner = 0;
            let echoed = null;
            let confirming = null;
            let awaitedFrame = 0;
            let probing = false;
            // The number of the parent's latest fresh fit (Natural height, Reset) this document has applied, on every
            // report from then on: the parent ignores a report from before it, measured in the old window and still on
            // its way when the fit started over.
            let fit = 0;
            const post = (message) => window.__ouroWidgetReport?.(fit ? { ...message, fit } : message);
            const followsItsFrame = (height) => {
                const inner = window.innerHeight;
                const excess = height - inner;
                const moved = !near(inner, seenInner);
                seenInner = inner;
                if (rememberedHold) {
                    if (excess > 0 && fits(height) && near(excess, rememberedHold)) held = { inner, excess };
                    rememberedHold = 0;
                }
                if (held) {
                    if (near(inner, held.inner) && near(excess, held.excess)) {
                        setVerticalOverflowSuppressed(false);
                        return true;
                    }
                    held = null;
                    echoed = null;
                    confirming = null;
                }
                const asksAgain = near(inner, lastHeight) && excess > 0 && fits(height) && (moved || confirming !== null);
                if (!asksAgain) {
                    confirming = null;
                    echoed = null;
                    if (moved) probing = false;
                    return false;
                }
                if (confirming === null || !near(confirming, excess)) {
                    confirming = moved ? excess : null;
                    if (moved) {
                        scheduleReport();
                        return true;
                    }
                    echoed = null;
                    return false;
                }
                confirming = null;
                if ((echoed !== null && excess >= echoed) || probing) {
                    held = { inner, excess, moving: !probing && settling() };
                    probing = false;
                    setVerticalOverflowSuppressed(false);
                    post({ type: 'ouro-widget-resize', height: lastHeight, held: excess });
                    return true;
                }
                echoed = excess;
                return false;
            };
            // The height the document measures as it is left, or null when that may differ from the last measure (a
            // scrollbar shown or hidden, a stop): its caller then measures again. measured is a measure just taken.
            const reportOnce = (measured) => {
                if (!root || disposed || !entryRan) return measured ?? null;
                // A board frame out of the page (Widgets left, the frame kept running) has no window to measure in: its
                // document is measured again when it has one (resize), not as nothing, which would toggle its scrollbar.
                if (wholeDocument && !window.innerHeight) return measured ?? null;
                if (awaitedFrame) {
                    if (!near(window.innerHeight, awaitedFrame)) return measured ?? null;
                    awaitedFrame = 0;
                }
                if (!drawnYet) {
                    if (!hasDrawn(document.body)) return measured ?? null;
                    drawnYet = true;
                }
                let height = measured ?? measure();
                // Released at the ceiling, a board document's margins collapse through its body and its scrollbar takes
                // width, so it can measure shorter than it is once hidden: whether it fits is decided, and its height
                // reported, in the geometry it has with the scrollbar hidden, as soon as it might fit.
                if (wholeDocument && !suppressingVerticalOverflow && fits(height)) {
                    setVerticalOverflowSuppressed(true);
                    height = measure();
                }
                if (wholeDocument && followsItsFrame(height)) return null;
                const hidden = suppressingVerticalOverflow;
                setVerticalOverflowSuppressed(fits(height));
                const settled = hidden === suppressingVerticalOverflow ? height : null;
                if (!height || height === lastHeight) return settled;
                lastHeight = height;
                // Through the bridge's channel (installed by the bootstrap that runs first).
                post({ type: 'ouro-widget-resize', height });
                return settled;
            };
            // What a change in the document can move: the height it measures (what its body holds, placed boxes
            // included, where its layout draws them) in the window it has now. Reading it changes nothing.
            const measuredNow = () => (root && document.body ? window.innerHeight + '/' + measure() : '');
            const report = (measured) => {
                const height = reportOnce(measured);
                if (wholeDocument) seenMeasure = height === null ? measuredNow() : window.innerHeight + '/' + height;
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
            // A change in a board document is measured once an animation frame, and reported only when it moved that;
            // the report goes on from that same measure.
            const scheduleChange = () => {
                if (disposed || changeFrame !== null) return;
                changeFrame = window.requestAnimationFrame(() => {
                    changeFrame = null;
                    if (!root || !document.body) return;
                    if (held?.moving && !settling()) {
                        held = null;
                        echoed = null;
                        confirming = null;
                        probing = true;
                        report();
                        return;
                    }
                    const height = measure();
                    if (window.innerHeight + '/' + height !== seenMeasure) report(height);
                });
            };
            const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(scheduleReport) : null;
            if (observer && root) observer.observe(root);
            const onLoad = () => {
                entryRan = true;
                report();
            };
            window.addEventListener('load', onLoad, { once: true });
            // A board document is measured as it is laid out, and what its layout places against the body or the page
            // (a bottom bar, centred content) moves when the frame resizes, with no change to the root's size. So does
            // a box placed in or beside the root, or overflowing it, that grows later: the document is measured again
            // whenever anything in it changes, when a transition or an animation ends or an image or a video gets its
            // size, and when the pointer or the focus moves, which a :hover or :focus-within rule may answer by
            // showing a box (a menu) without any change to the document.
            const layoutCues = ['transitionend', 'animationend', 'load', 'loadedmetadata', 'pointerover', 'pointerout',
                'focusin', 'focusout'];
            if (wholeDocument) {
                window.addEventListener('resize', scheduleReport);
                layoutCues.forEach((type) => document.addEventListener(type, scheduleChange, true));
                if (typeof MutationObserver === 'function') {
                    mutations = new MutationObserver(scheduleChange);
                    mutations.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
                }
            }
            // New bounds end a hold (the owner's height, Show all, Back); Natural height and Reset (frame: the height
            // being applied, fit: that fresh fit's number) also make it measure anew, once its window has that height.
            window.__ouroWidgetOnViewport?.((viewport) => {
                floor = viewport.floor;
                ceiling = viewport.ceiling;
                held = null;
                rememberedHold = 0;
                echoed = null;
                confirming = null;
                probing = false;
                awaitedFrame = viewport.frame || 0;
                if (viewport.frame) {
                    lastHeight = 0;
                    fit = Number(viewport.fit) || fit;
                }
                report();
            });
            window.__ouroWidgetOnDispose?.(() => {
                disposed = true;
                if (resizeFrame !== null) window.cancelAnimationFrame(resizeFrame);
                resizeFrame = null;
                if (changeFrame !== null) window.cancelAnimationFrame(changeFrame);
                changeFrame = null;
                observer?.disconnect();
                mutations?.disconnect();
                mutations = null;
                if (wholeDocument) layoutCues.forEach((type) => document.removeEventListener(type, scheduleChange, true));
                window.removeEventListener('load', onLoad);
                window.removeEventListener('resize', scheduleReport);
                setVerticalOverflowSuppressed(false);
            });
            report();
        })();
    `;
}
