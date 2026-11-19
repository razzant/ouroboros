import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    WIDGET_BOARD_FRAME_BORDER_RESERVE,
    WIDGET_FRAME_BORDER_RESERVE,
    WIDGET_FRAME_DEFAULT_HEIGHT,
    WIDGET_FRAME_MAX_HEIGHT,
} from '../modules/widgets.js';
import {
    moduleResizeScript,
} from '../modules/widget_frame.js';
import {
    classifyWidgetJobStatus,
    isRetryableWidgetError,
    readWidgetJobStatus,
    withWidgetRequestTimeout,
} from '../modules/widget_job.js';

function resizeHarness({
    floor = WIDGET_FRAME_DEFAULT_HEIGHT,
    maxHeight = 1000,
    borderReserve = WIDGET_FRAME_BORDER_RESERVE,
    initialHeight = 600,
    paddingBottom = 12,
    borderBottom = 4,
    bodyTop = 0,
    marginBottom = 0,
    childMargins = 0,
    wholeDocument = false,
    // The stop a card's previous document reached (widget_module.js bakes it from the card's memory).
    heldExcess = 0,
    // A body whose own height or min-height (100vh, 100%) makes it this tall; content taller than it still stretches it.
    imposedBodyHeight = 0,
    // The same, as 100vh: as tall as the frame's viewport, whatever that is.
    imposedFollowsViewport = false,
    // Margins the content's children keep inside the body only while its overflow is hidden (they collapse through it).
    marginsCollapseWhenReleased = false,
    // A root that follows the frame's viewport (height: 100%, 100vh): its height is the window's; `rootOf(innerHeight)`
    // any other relation (half of it, an offset).
    followsViewport = false,
    rootOf = followsViewport ? (height) => height : null,
    innerHeight = 768,
    // Declarations an author set on the body's own style attribute before the reporter ran.
    bodyInline = {},
    // The host's mark after the module's entry (widget_module.js), given at once unless a case withholds it.
    entryRan = true,
    // Boxes the root holds, placed absolutely against the body (the root is not positioned): { top, height, children }
    // in document pixels, `top` a number or a function of the frame's height (a bar at its bottom); a child is placed
    // against its parent box and lies in that box's overflow.
    placed = [],
    // The document's running animations (document.getAnimations()); none unless a case has some.
    animations = () => [],
} = {}) {
    const state = { height: initialHeight };
    const messages = [];
    const sequence = [];
    const disposeCallbacks = [];
    const viewportCallbacks = [];
    const listeners = new Map();
    const documentListeners = new Map();
    const styleWrites = { set: 0, remove: 0 };
    let observerCallback = null;
    let observerDisconnects = 0;
    let frameId = 0;
    const frames = new Map();
    const flushFrame = () => {
        const pending = [...frames.values()];
        frames.clear();
        pending.forEach((callback) => callback());
    };
    // The reporter owns overflow-y and writes nothing else: any other declaration it set would fail here.
    const createStyle = (label, initial = {}) => {
        const declarations = new Map();
        if (initial.value) declarations.set('overflow-y', initial);
        return {
            declarations,
            present: Boolean(initial.value),
            getPropertyValue(name) { return declarations.get(name)?.value || ''; },
            getPropertyPriority(name) { return declarations.get(name)?.priority || ''; },
            setProperty(name, value, priority = '') {
                assert.equal(name, 'overflow-y', `${label}: the reporter wrote ${name}`);
                this.present = true;
                declarations.set(name, { value, priority });
                styleWrites.set += 1;
                sequence.push(`${label}:set:${value}:${priority}`);
            },
            removeProperty(name) {
                assert.equal(name, 'overflow-y', `${label}: the reporter removed ${name}`);
                declarations.delete(name);
                styleWrites.remove += 1;
                sequence.push(`${label}:remove`);
            },
        };
    };
    const documentElement = { style: createStyle('html') };
    const follow = { rootOf };
    const contentHeight = () => (follow.rootOf ? follow.rootOf(window.innerHeight) : state.height);
    // An element of this stand-in: its box, its scrollable overflow (its children's boxes), the siblings the reporter
    // reads backwards, and its computed style.
    const element = (box, style, children = []) => {
        const node = {
            nodeType: 1, clientTop: 0, children, style: null, computed: { display: 'block', overflowY: 'visible',
                marginBottom: '0px', ...style },
            get lastElementChild() { return children.at(-1) || null; },
            previousElementSibling: null,
            getBoundingClientRect: () => { const { top, height } = box(); return { top, height, bottom: top + height }; },
            get scrollHeight() {
                const { top, height } = box();
                return Math.max(height, ...children.map((child) => child.getBoundingClientRect().top + child.scrollHeight - top));
            },
        };
        children.forEach((child, index) => { child.previousElementSibling = children[index - 1] || null; });
        return node;
    };
    const placedBox = ({ top, height, children = [] }, parentTop = 0) => {
        const at = () => parentTop + (typeof top === 'function' ? top(window.innerHeight) : top);
        return element(() => ({ top: at(), height }), { position: 'absolute' }, children.map((child) => placedBox(child, 0)));
    };
    const root = element(() => ({ top: bodyTop, height: contentHeight() }), { position: 'static' },
        placed.map((box) => placedBox(box)));
    // The root's own scroll height is its content; what is placed against the body lies outside it.
    Object.defineProperty(root, 'scrollHeight', { get: () => contentHeight() });
    Object.defineProperty(root, 'clientHeight', { get: () => contentHeight() });
    const hidden = () => documentElement.style.getPropertyValue('overflow-y') === 'hidden';
    const imposed = () => (imposedFollowsViewport ? window.innerHeight : imposedBodyHeight);
    const heldHeight = () => contentHeight() + (marginsCollapseWhenReleased && !hidden() ? 0 : childMargins);
    const sized = imposedFollowsViewport || imposedBodyHeight > 0 || 'height' in bodyInline || 'min-height' in bodyInline;
    const body = {
        style: Object.assign(createStyle('body'), {}),
        scrollHeight: 0,
        clientHeight: 0,
        children: [root],
        childNodes: [root],
        // The body's box holds the content, its bottom spacing and the margins the content's children keep inside it;
        // its own height or min-height (`imposed`) can make it taller.
        getBoundingClientRect: () => {
            const height = Math.max(imposed(), heldHeight() + paddingBottom + borderBottom);
            return { top: bodyTop, bottom: bodyTop + height, height };
        },
        // Typed computed values, as both engines give them: auto and a zero min-height unless the body is sized.
        computedStyleMap: () => new Map([
            ['height', { value: sized && !imposedFollowsViewport ? imposed() || 1 : 'auto' }],
            ['min-height', { value: sized ? imposed() || 1 : 0 }],
        ]),
    };
    // The style attribute as an engine serializes it; the author's own script may set it (`authorAttribute`), the
    // reporter never does.
    const serialize = () => [...body.style.declarations]
        .map(([name, { value, priority }]) => `${name}: ${value}${priority ? ' !important' : ''};`).join(' ');
    const setAuthorAttribute = (text) => {
        body.style.present = text !== null;
        body.style.declarations.clear();
        for (const part of String(text ?? '').split(';').map((item) => item.trim()).filter(Boolean)) {
            const [key, ...rest] = part.split(':');
            const raw = rest.join(':').trim();
            const important = raw.endsWith('!important');
            body.style.declarations.set(key.trim(), { value: important ? raw.slice(0, -10).trim() : raw, priority: important ? 'important' : '' });
        }
    };
    body.getAttribute = (name) => (name === 'style' && body.style.present ? serialize() : null);
    if (Object.keys(bodyInline).length) {
        setAuthorAttribute(Object.entries(bodyInline).map(([name, raw]) => `${name}: ${raw}`).join('; '));
    }
    const document = {
        body,
        documentElement,
        getElementById: (id) => (id === 'root' ? root : null),
        getAnimations: () => animations(),
        addEventListener(type, listener) { documentListeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (documentListeners.get(type) === listener) documentListeners.delete(type);
        },
    };
    const overflowState = () => Object.fromEntries(
        [['html', documentElement], ['body', body]].map(([label, element]) => [label, {
            value: element.style.getPropertyValue('overflow-y'),
            priority: element.style.getPropertyPriority('overflow-y'),
        }]),
    );
    const window = {
        innerHeight,
        requestAnimationFrame(callback) { frames.set(++frameId, callback); return frameId; },
        cancelAnimationFrame(id) { frames.delete(id); },
        // The bridge's channel, which the resize reporter posts through.
        __ouroWidgetReport(message) {
            messages.push(message);
            sequence.push('message');
        },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (listeners.get(type) === listener) listeners.delete(type);
        },
        __ouroWidgetOnDispose(callback) { disposeCallbacks.push(callback); },
        // The bridge hands the reporter a Widgets card's floor and ceiling (ouro-widget-viewport).
        __ouroWidgetOnViewport(callback) { viewportCallbacks.push(callback); },
    };
    class FakeResizeObserver {
        constructor(callback) { observerCallback = callback; }
        observe(target) { assert.equal(target, root); }
        disconnect() { observerDisconnects += 1; }
    }
    const getComputedStyle = (target) => {
        if (target !== body) return target.computed;
        return {
            display: 'block',
            position: 'static',
            overflowY: hidden() ? 'hidden' : 'visible',
            // The used content height, as an engine resolves it: an imposed height, else what the body holds.
            height: `${Math.max(imposed() - paddingBottom - borderBottom, heldHeight())}px`,
            paddingBottom: `${paddingBottom}px`,
            borderBottomWidth: `${borderBottom}px`,
            marginBottom: `${marginBottom}px`,
        };
    };
    Function(
        'document',
        'window',
        'ResizeObserver',
        'getComputedStyle',
        moduleResizeScript(floor, maxHeight, borderReserve, { wholeDocument, heldExcess }),
    )(document, window, FakeResizeObserver, getComputedStyle);
    if (entryRan) window.__ouroWidgetEntryRan?.();
    return {
        messages,
        sequence,
        overflowState,
        styleWrites,
        bodyStyle: () => Object.fromEntries(body.style.declarations),
        bodyAttribute: () => body.getAttribute('style'),
        // The author's own script changes the body's style attribute.
        authorAttribute: setAuthorAttribute,
        listeners,
        documentListeners,
        observerDisconnects: () => observerDisconnects,
        pendingFrames: () => frames.size,
        flushFrame,
        notify(height) { state.height = height; observerCallback(); },
        resize(height) {
            state.height = height;
            observerCallback();
            flushFrame();
        },
        // The root's relation to the frame changes at a standing frame (a resize of real content under a follower).
        setRootOf(next) { follow.rootOf = next; observerCallback(); flushFrame(); },
        dispose() { disposeCallbacks.forEach((callback) => callback()); },
        entryRan() { window.__ouroWidgetEntryRan?.(); },
        viewport(next) { viewportCallbacks.forEach((callback) => callback(next)); },
        // The parent applies a frame height: the window follows it (and fires resize), and a root that follows it
        // resizes.
        frame(height) {
            if (height === window.innerHeight) return;
            window.innerHeight = height;
            listeners.get('resize')?.();
            if (follow.rootOf) observerCallback();
            flushFrame();
        },
    };
}

test('resize observer defers layout writes, coalesces updates, and cancels on dispose', () => {
    const harness = resizeHarness();
    harness.notify(1200);
    harness.notify(1300);
    assert.equal(harness.pendingFrames(), 1);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 0 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616]);
    harness.flushFrame();
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1316]);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    harness.notify(500);
    assert.equal(harness.pendingFrames(), 1);
    harness.dispose();
    assert.equal(harness.pendingFrames(), 0);
    harness.flushFrame();
    harness.notify(400);
    assert.equal(harness.pendingFrames(), 0);
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1316]);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
});

test('widget frame contract keeps the bounded host geometry', () => {
    assert.equal(WIDGET_FRAME_DEFAULT_HEIGHT, 320);
    assert.equal(WIDGET_FRAME_MAX_HEIGHT, 8192);
    assert.equal(WIDGET_FRAME_BORDER_RESERVE, 2);
    // A board frame has no border (style.css), so nothing is added to what its content reports.
    assert.equal(WIDGET_BOARD_FRAME_BORDER_RESERVE, 0);
});

test('module auto-height owns only vertical overflow across cap transitions', () => {
    const harness = resizeHarness();
    assert.deepEqual(harness.overflowState(), {
        html: { value: 'hidden', priority: 'important' },
        body: { value: 'hidden', priority: 'important' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 0 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616]);
    assert.deepEqual(harness.sequence.slice(0, 3), [
        'html:set:hidden:important', 'body:set:hidden:important', 'message',
    ]);

    harness.resize(600);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 0 });
    assert.equal(harness.messages.length, 1);

    harness.resize(1200);
    assert.deepEqual(harness.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1216]);

    harness.resize(1200);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.equal(harness.messages.length, 2);

    harness.resize(500);
    assert.deepEqual(harness.overflowState(), {
        html: { value: 'hidden', priority: 'important' },
        body: { value: 'hidden', priority: 'important' },
    });
    assert.deepEqual(harness.styleWrites, { set: 4, remove: 2 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1216, 516]);
});

test('module overflow ownership covers floor equality, fixed-height no-op, and cleanup', () => {
    const floorCap = resizeHarness({ maxHeight: WIDGET_FRAME_DEFAULT_HEIGHT, initialHeight: 100 });
    assert.deepEqual(floorCap.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(floorCap.styleWrites, { set: 2, remove: 2 });

    const harness = resizeHarness();
    harness.dispose();
    assert.deepEqual(harness.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.equal(harness.observerDisconnects(), 1);
    assert.equal(harness.listeners.has('load'), false);

    // Page host plus the framed mounts split out of it: the resize bridge wiring
    // lives in widget_module.js; the negative pin must hold across both files.
    const framedSource = ['widgets.js', 'widget_module.js']
        .map((name) => readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8'))
        .join('\n');
    // A Widgets card bakes its board floor and ceiling; a frame in a message keeps the author's ceiling.
    assert.match(framedSource, /const resizeBridge = !autoHeight \? '' : placed\s*\? moduleResizeScript\(placed\.floor, placed\.ceiling, WIDGET_BOARD_FRAME_BORDER_RESERVE, \{ wholeDocument: true\b/);
    assert.match(
        framedSource,
        /moduleResizeScript\(WIDGET_FRAME_DEFAULT_HEIGHT,\s*maxHeight,\s*WIDGET_FRAME_BORDER_RESERVE\)/,
    );
    assert.doesNotMatch(framedSource, /scrolling="no"|syncModuleFrameScrolling/);
});

test('widget job retry classification distinguishes transport from terminal errors', () => {
    assert.equal(isRetryableWidgetError({ status: 408 }), true);
    assert.equal(isRetryableWidgetError({ status: 429 }), true);
    assert.equal(isRetryableWidgetError({ status: 503 }), true);
    assert.equal(isRetryableWidgetError({ name: 'TypeError' }), true);
    assert.equal(isRetryableWidgetError({ status: 400 }), false);
    assert.equal(isRetryableWidgetError({ status: 404 }), false);
    assert.equal(isRetryableWidgetError({ status: 200, retryable: false }), false);
    assert.equal(isRetryableWidgetError({ name: 'AbortError', retryable: true }), false);
});

test('widget jobs bound unknown status and reject a missing status', () => {
    assert.equal(classifyWidgetJobStatus('queued'), 'pending');
    assert.equal(classifyWidgetJobStatus('running'), 'pending');
    assert.equal(classifyWidgetJobStatus('done'), 'success');
    assert.equal(classifyWidgetJobStatus('failed'), 'failure');
    assert.equal(classifyWidgetJobStatus(''), 'invalid');
    assert.equal(classifyWidgetJobStatus(123), 'invalid');
    assert.equal(classifyWidgetJobStatus({}), 'invalid');
    assert.equal(classifyWidgetJobStatus([]), 'invalid');
    assert.equal(classifyWidgetJobStatus('mystery'), 'pending');
});

test('widget job status selection preserves explicit falsy status values', () => {
    assert.equal(readWidgetJobStatus({ status: 0, state: 'running' }), 0);
    assert.equal(readWidgetJobStatus({ status: false, state: 'running' }), false);
    assert.equal(readWidgetJobStatus({ status: '', state: 'running' }), '');
    assert.equal(readWidgetJobStatus({ state: 'running' }), 'running');
});

test('widget request timeout aborts the request and remains retryable', async () => {
    const controller = new AbortController();
    await assert.rejects(
        withWidgetRequestTimeout(
            (signal) => new Promise((_, reject) => {
                signal.addEventListener('abort', () => {
                    const error = new Error('aborted');
                    error.name = 'AbortError';
                    reject(error);
                }, { once: true });
            }),
            controller,
            5,
        ),
        (error) => error.code === 'WIDGET_REQUEST_TIMEOUT' && error.retryable === true,
    );
    assert.equal(controller.signal.aborted, true);
});

test('widget request timeout stays terminal when the task swallows abort', async () => {
    const controller = new AbortController();
    await assert.rejects(
        withWidgetRequestTimeout(
            () => new Promise((resolve) => setTimeout(() => resolve('late result'), 20)),
            controller,
            5,
        ),
        (error) => error.code === 'WIDGET_REQUEST_TIMEOUT' && error.retryable === true,
    );
    assert.equal(controller.signal.aborted, true);
});

test('a board ceiling decides overflow again for the same content height, and reports nothing new', () => {
    // 600 px of content (616 with the body's bottom spacing) under the author's 1000: hidden, the frame grows.
    const harness = resizeHarness({ floor: 0, maxHeight: 1000 });
    assert.equal(harness.overflowState().html.value, 'hidden');
    assert.deepEqual(harness.messages.map((item) => item.height), [616]);
    // The owner's 400 px: the frame stops there and its document scrolls, though no content changed.
    harness.viewport({ floor: 0, ceiling: 400 });
    assert.deepEqual(harness.overflowState(), { html: { value: '', priority: '' }, body: { value: '', priority: '' } });
    assert.deepEqual(harness.messages.map((item) => item.height), [616], 'an unchanged height is not reported again');
    // Show all: the author's ceiling again; the document hides its scrollbar while the frame grows to fit.
    harness.viewport({ floor: 0, ceiling: 8192 });
    assert.equal(harness.overflowState().body.value, 'hidden');
    assert.equal(harness.messages.length, 1);
    // Content that grows past the owner's ceiling is still reported (Show all counts it), and scrolls.
    harness.viewport({ floor: 0, ceiling: 400 });
    harness.resize(900);
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 916]);
    assert.equal(harness.overflowState().html.value, '');
    // Paired: shrinking under the ceiling hides the scrollbar again.
    harness.resize(200);
    assert.equal(harness.overflowState().html.value, 'hidden');
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 916, 216]);
});

test('a remount at a remembered owner height scrolls from its first report, with no message from the parent', () => {
    // Baked ceiling 400 (the owner's height) under 600 px of content: released at once, without any resize event.
    const harness = resizeHarness({ floor: 0, maxHeight: 400 });
    assert.deepEqual(harness.overflowState(), { html: { value: '', priority: '' }, body: { value: '', priority: '' } });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616]);
    // Its negative control: the same content under the author's ceiling keeps the scrollbar hidden.
    assert.equal(resizeHarness({ floor: 0, maxHeight: 8192 }).overflowState().html.value, 'hidden');
});

test('a board frame counts the margins its content sits between; a frame in a message keeps its measure', () => {
    // 18 px of text whose paragraph keeps 16 px margins inside a body with the default 8 px margins: 66 px of document.
    const page = { floor: 0, initialHeight: 18, paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8, childMargins: 32 };
    assert.deepEqual(resizeHarness({ ...page, wholeDocument: true }).messages.map((item) => item.height), [66]);
    // The frame in a message (no floor removed there) keeps reporting the body box it always reported.
    assert.deepEqual(resizeHarness(page).messages.map((item) => item.height), [18]);
});

// The parent's side of a board frame, round after round: each report becomes the frame's height (plus `reserve`),
// under `ceiling` (the owner's height, or the author's ceiling while shown all). An echo is looked at twice, one
// animation frame apart.
function settleBoard(harness, reserve, rounds = 6, ceiling = Infinity) {
    for (let round = 0; round < rounds; round += 1) {
        harness.frame(Math.min(ceiling, harness.messages.at(-1).height + reserve));
        harness.flushFrame();
    }
    return harness.messages.map((item) => item.height);
}

test('content that follows its board frame with nothing outside it rests at the frame it has', () => {
    const page = { floor: 0, maxHeight: 8192, wholeDocument: true, followsViewport: true, innerHeight: 320,
        paddingBottom: 0, borderBottom: 0, borderReserve: WIDGET_BOARD_FRAME_BORDER_RESERVE };
    // #root { height: 100% } or min-height: 100vh at the 320 px loading height: its report is the frame it has.
    assert.deepEqual(settleBoard(resizeHarness(page), WIDGET_BOARD_FRAME_BORDER_RESERVE), [320]);
    // Negative control: the message frame's 2 px on the border-less board frame climbs 2 px every report.
    assert.deepEqual(settleBoard(resizeHarness({ ...page, borderReserve: 2 }), 2), [320, 322, 324, 326, 328, 330, 332]);
});

test('content that follows its frame with space outside it stops growing and scrolls where it stands', () => {
    const heights = (harness) => harness.messages.map((item) => item.height);
    const follower = { floor: 0, wholeDocument: true, followsViewport: true, paddingBottom: 0, borderBottom: 0,
        bodyTop: 8, marginBottom: 8 };
    // Each report asks for 16 px more than the frame it got. The second time the same extra is confirmed at a height the
    // frame answered exactly, the reporter stops asking, says so once, and lets the document scroll where it stands.
    const free = resizeHarness({ ...follower, maxHeight: 8192, innerHeight: 320 });
    const climbed = settleBoard(free, 0, 8);
    assert.deepEqual(climbed.slice(0, 3), [336, 352, 352]);
    assert.equal(climbed.length, 3, 'nothing more is reported once it stopped');
    assert.equal(free.messages.at(-1).held, 16);
    assert.deepEqual(free.overflowState(), { html: { value: '', priority: '' }, body: { value: '', priority: '' } }, 'released: it scrolls');
    // A new look at the same height (a resize that changed nothing) stays stopped and still scrolls.
    free.flushFrame();
    free.notify(500);
    free.flushFrame();
    assert.equal(heights(free).length, 3);
    assert.equal(free.overflowState().html.value, '');
    // The owner's height is exact and the reporter's stop plays no part under it: the frame stays at 240, reported once.
    const owned = resizeHarness({ ...follower, maxHeight: 240, innerHeight: 240 });
    assert.deepEqual(settleBoard(owned, 0, 6, 240), [256]);
    assert.equal(owned.overflowState().html.value, '', 'at its ceiling: it scrolls');
    // Show all: free again, the follower stops two echoes later, without running to any ceiling.
    owned.viewport({ floor: 0, ceiling: 8192 });
    settleBoard(owned, 0, 12, 8192);
    assert.deepEqual(heights(owned).slice(-2), [272, 272]);
    assert.equal(owned.messages.at(-1).height, 272);
    assert.equal(owned.messages.at(-1).held, 16);
    assert.equal(owned.overflowState().html.value, '');
    // Back: the owner's 240 px again, one report at it, then quiet, scrolling.
    owned.viewport({ floor: 0, ceiling: 240 });
    owned.frame(240);
    const after = heights(owned).length;
    settleBoard(owned, 0, 6, 240);
    assert.equal(heights(owned).at(-1), 256);
    assert.equal(heights(owned).length, after, 'nothing reported at the owner height');
    assert.equal(owned.overflowState().html.value, '');
});

test('a document whose extra grows with its frame stops at the second echo too, and scrolls', () => {
    const page = { floor: 0, maxHeight: 8192, wholeDocument: true, innerHeight: 320, paddingBottom: 0, borderBottom: 0 };
    // #root{height:110vh} (no margins), the same in vh padding, and 105vh between 8 px margins: the extra is never the
    // same twice, and growing never brings it nearer. Each stops two echoes past the loading height, released.
    for (const [label, extra] of [['110vh', { rootOf: (inner) => Math.ceil(inner * 1.1) }],
        ['105vh with margins', { rootOf: (inner) => Math.ceil(inner * 1.05), bodyTop: 8, marginBottom: 8 }]]) {
        const harness = resizeHarness({ ...page, ...extra });
        const climbed = settleBoard(harness, 0, 12);
        assert.ok(climbed.length <= 4 && climbed.at(-1) < 480, `${label}: ${climbed}`);
        assert.ok(harness.messages.at(-1).held > 0, `${label}: it says it stopped`);
        assert.equal(harness.overflowState().html.value, '', `${label}: released, it scrolls`);
        // Remembered: its next document at the stop height stays there.
        const again = resizeHarness({ ...page, ...extra, innerHeight: climbed.at(-1), heldExcess: harness.messages.at(-1).held });
        assert.deepEqual(again.messages, [], `${label}: no growth on return`);
    }
    // Paired: an extra that shrinks at each echo converges and is followed to its height (inner/2 + 300 rests at 600),
    // step after step; only its last whole pixel, where the extra no longer shrinks, may be left to scroll.
    const converging = resizeHarness({ ...page, rootOf: (inner) => Math.ceil(inner / 2 + 300) });
    const settled = settleBoard(converging, 0, 16);
    assert.ok(settled.length >= 7 && Math.abs(settled.at(-1) - 600) <= 2, `${settled}`);
});

test('an echo at which the document fits ends its episode: a later growth is not held on an old echo', () => {
    const harness = resizeHarness({ floor: 0, wholeDocument: true, maxHeight: 8192, innerHeight: 100, initialHeight: 100,
        paddingBottom: 0, borderBottom: 0 });
    // 20 px arrive while the frame answers the report: one confirmed echo asking for 20 px, then the frame fits it.
    harness.resize(120);
    harness.notify(140);
    harness.frame(120);
    harness.flushFrame();
    harness.frame(140);
    harness.flushFrame();
    // It shrinks, and as the frame answers 60 px more arrive: a new episode's first echo, followed, not held.
    harness.resize(100);
    harness.notify(160);
    harness.frame(100);
    harness.flushFrame();
    harness.frame(160);
    harness.flushFrame();
    assert.deepEqual(harness.messages.map((item) => item.height), [100, 120, 140, 100, 160]);
    assert.equal(harness.messages.every((item) => item.held === undefined), true);
    assert.equal(harness.overflowState().html.value, 'hidden');
});

test('after a fresh fit every report carries its number; before it none does', () => {
    const harness = resizeHarness({ floor: 0, wholeDocument: true, maxHeight: 8192, innerHeight: 480, initialHeight: 180,
        paddingBottom: 0, borderBottom: 0 });
    assert.deepEqual(harness.messages, [{ type: 'ouro-widget-resize', height: 180 }]);
    // Natural height: fresh fit number 1 at the 320 px loading height. Nothing from the old window, then the report
    // at 320 and every later one carry it.
    harness.viewport({ floor: 0, ceiling: 8192, frame: 320, fit: 1 });
    harness.resize(181);
    assert.equal(harness.messages.length, 1, 'no report from the old window');
    harness.frame(320);
    harness.frame(181);
    harness.resize(381);
    assert.deepEqual(harness.messages.slice(1), [{ type: 'ouro-widget-resize', height: 181, fit: 1 },
        { type: 'ouro-widget-resize', height: 381, fit: 1 }]);
    // New bounds without a fresh fit keep the number.
    harness.viewport({ floor: 0, ceiling: 300 });
    harness.resize(400);
    assert.equal(harness.messages.at(-1).fit, 1);
});

test('real content that changes under a held frame takes the hold off, and a hold is only remembered if it still fits', () => {
    const follower = { floor: 0, wholeDocument: true, followsViewport: true, paddingBottom: 0, borderBottom: 0,
        bodyTop: 8, marginBottom: 8, maxHeight: 8192, innerHeight: 320 };
    const held = resizeHarness(follower);
    settleBoard(held, 0, 8);
    assert.equal(held.messages.at(-1).held, 16);
    // The document now holds 200 px of its own: at the same frame the extra differs, so it measures on and the frame follows.
    held.setRootOf(() => 200);
    assert.deepEqual(held.messages.map((item) => item.height).slice(-1), [216]);
    assert.equal(held.messages.at(-1).held, undefined);
    assert.equal(held.overflowState().html.value, 'hidden', 'hidden again: it fits and the frame follows');
    // A new document of the same card remembers the stop: the same extra at the remembered height, nothing moves.
    const remounted = resizeHarness({ ...follower, innerHeight: 352, heldExcess: 16 });
    assert.deepEqual(remounted.messages, [], 'no growth, no report: the card already stands there');
    assert.equal(remounted.overflowState().html.value, '', 'and its document scrolls');
    // A different extra (it holds more than when it stopped): the memory is dropped and it measures as usual.
    const grown = resizeHarness({ ...follower, innerHeight: 352, heldExcess: 16, rootOf: () => 500 });
    assert.deepEqual(grown.messages.map((item) => item.height), [516]);
    assert.equal(grown.messages.at(-1).held, undefined);
});

test('a stop made while an animation runs stands until nothing moves, then looks once more and stays', () => {
    const fade = { playState: 'running', effect: { getTiming: () => ({ iterations: 1 }) } };
    let running = [fade];
    const harness = resizeHarness({ floor: 0, wholeDocument: true, followsViewport: true, paddingBottom: 0,
        borderBottom: 0, bodyTop: 8, marginBottom: 8, maxHeight: 8192, innerHeight: 320, animations: () => running });
    const heights = () => harness.messages.map((item) => item.height);
    // An animation that moves nothing: the follower stops where it would without it and stands still while it runs.
    settleBoard(harness, 0, 8);
    assert.deepEqual(heights(), [336, 352, 352]);
    assert.equal(harness.messages.at(-1).held, 16);
    harness.documentListeners.get('pointerover')();
    harness.flushFrame();
    settleBoard(harness, 0, 6);
    assert.deepEqual(heights(), [336, 352, 352], 'still while it runs');
    // It ends, a layout cue: the document gets what it asks for once, asks again there, and stays.
    running = [];
    harness.documentListeners.get('animationend')();
    harness.flushFrame();
    settleBoard(harness, 0, 8);
    assert.deepEqual(heights().slice(3), [368, 368]);
    assert.equal(harness.messages.at(-1).held, 16);
    // That stop is for good: a later animation and its end change nothing, and the document scrolls.
    running = [fade];
    harness.documentListeners.get('transitionend')();
    harness.flushFrame();
    running = [];
    harness.documentListeners.get('animationend')();
    harness.flushFrame();
    settleBoard(harness, 0, 6);
    assert.deepEqual(heights().slice(3), [368, 368]);
    assert.equal(harness.overflowState().html.value, '');
});

test('new bounds end a hold, and a fresh start measures only once the frame has the height it was given', () => {
    const heights = (harness) => harness.messages.map((item) => item.height);
    const follower = { floor: 0, wholeDocument: true, followsViewport: true, paddingBottom: 0, borderBottom: 0,
        bodyTop: 8, marginBottom: 8, maxHeight: 8192, innerHeight: 320 };
    const harness = resizeHarness(follower);
    settleBoard(harness, 0, 8);
    assert.deepEqual(heights(harness), [336, 352, 352]);
    // Natural height / Reset: the host forgets the fit and takes the frame back to the loading height. Until the window
    // has that height the reporter stays quiet (the window it has now is not the one it was given).
    harness.viewport({ floor: 0, ceiling: 8192, frame: 320 });
    harness.flushFrame();
    assert.deepEqual(heights(harness), [336, 352, 352], 'no report from the old window');
    harness.frame(320);
    settleBoard(harness, 0, 6);
    assert.deepEqual(heights(harness).slice(3), [336, 352, 352], 'the same stop from the same start, no drift');
    assert.equal(harness.messages.at(-1).held, 16);
    // A newer change of bounds supersedes the wait: nothing stays stuck waiting for a height that will never come.
    harness.viewport({ floor: 0, ceiling: 8192, frame: 320 });
    harness.viewport({ floor: 0, ceiling: 400 });
    harness.frame(400);
    harness.flushFrame();
    assert.equal(heights(harness).at(-1), 416, 'measured at the frame it was given');
});

test('an ordinary root that grows in the same step as Show all is reported whole', () => {
    // 240 px of content under the owner's 120 px, released; it grows to 360 px exactly as Show all lifts the cap and the
    // frame takes the last natural 240 px: the reporter has no state to hold it back.
    const harness = resizeHarness({ floor: 0, maxHeight: 120, wholeDocument: true, innerHeight: 120, initialHeight: 240,
        paddingBottom: 0, borderBottom: 0 });
    assert.deepEqual(harness.messages.map((item) => item.height), [240]);
    harness.notify(360);
    harness.viewport({ floor: 0, ceiling: 8192 });
    harness.frame(240);
    assert.deepEqual(harness.messages.map((item) => item.height), [240, 360]);
});

test('growth that keeps changing is never held: rows arriving between looks, an honest grow on the echo', () => {
    const ordinary = { floor: 0, wholeDocument: true, paddingBottom: 0, borderBottom: 0, bodyTop: 0, marginBottom: 0, maxHeight: 8192 };
    // Rows arriving every frame: the frame answers, and more arrives before the second look at the same height.
    const stream = resizeHarness({ ...ordinary, innerHeight: 100, initialHeight: 100 });
    let rows = 100;
    for (let round = 0; round < 12; round += 1) {
        const asked = stream.messages.at(-1)?.height ?? 100;
        rows += 20;
        stream.notify(rows);
        stream.frame(asked);
        rows += 20;
        stream.notify(rows);
        stream.flushFrame();
    }
    assert.equal(stream.messages.every((item) => item.held === undefined), true, 'a stream is not stopped');
    assert.equal(stream.overflowState().html.value, 'hidden');
    stream.frame(rows);
    stream.flushFrame();
    assert.equal(stream.messages.at(-1).height, rows);
    // A skeleton that becomes 500 px exactly when the frame answers its first report: grown once, whole, not stopped.
    const skeleton = resizeHarness({ ...ordinary, innerHeight: 320, initialHeight: 116 });
    assert.deepEqual(skeleton.messages.map((item) => item.height), [116]);
    skeleton.notify(500);
    skeleton.frame(116);
    skeleton.flushFrame();
    skeleton.frame(500);
    skeleton.flushFrame();
    assert.deepEqual(skeleton.messages.map((item) => item.height), [116, 500]);
    assert.equal(skeleton.messages.at(-1).held, undefined);
});

test('ordinary growth in steps and a root that follows half of its frame both settle on their height', () => {
    const heights = (harness) => harness.messages.map((item) => item.height);
    // An ordinary root grown to exactly its frame (8 px margins) twice running: counted whole both times.
    const ordinary = resizeHarness({ floor: 0, wholeDocument: true, innerHeight: 216, initialHeight: 216,
        paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8 });
    ordinary.frame(232);
    ordinary.resize(232);
    for (const [content, frame] of [[300, 248], [400, 316], [401, 416]]) {
        ordinary.frame(frame);
        ordinary.resize(content);
    }
    assert.deepEqual(heights(ordinary), [232, 248, 316, 416, 417]);
    // inner/2 + 50 px between 8 px margins rests at 132 px (133 with whole pixels).
    const half = resizeHarness({ floor: 0, wholeDocument: true, innerHeight: 320, rootOf: (inner) => inner / 2 + 50,
        paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8 });
    const settled = settleBoard(half, 0, 12);
    assert.ok(Math.abs(settled.at(-1) - 132) <= 1, `${settled}`);
});

test('a board body as tall as the frame by its own height or min-height (100vh) counts what it holds', () => {
    // A 100vh body (min-height or height) at a 320 px frame holding 40 px between 8 px margins: 56 px of document.
    const imposed = resizeHarness({ floor: 0, wholeDocument: true, innerHeight: 320, imposedFollowsViewport: true,
        initialHeight: 40, paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8,
        bodyInline: { 'min-height': '100vh' } });
    assert.deepEqual(imposed.messages.map((item) => item.height), [56]);
    // Measured as it is laid out: the body's style attribute holds the author's declaration and the reporter's
    // overflow, the only property it writes (on the root element and the body, once each).
    assert.equal(imposed.bodyAttribute(), 'min-height: 100vh; overflow-y: hidden !important;');
    assert.deepEqual(imposed.styleWrites, { set: 2, remove: 0 });
    // Stable there: at its own 56 px the 100vh body is again as tall as the frame, and still 56 px of document.
    imposed.frame(56);
    imposed.resize(40);
    assert.deepEqual(imposed.messages.map((item) => item.height), [56]);
    // Content that outgrows the 100vh body still grows the frame (the body no longer holds it).
    imposed.resize(400);
    assert.deepEqual(imposed.messages.map((item) => item.height), [56, 416]);
    // An author's !important declarations come back with their priority and order, and a body with no style attribute
    // at all keeps having none.
    imposed.authorAttribute('height: 100vh !important; color: red;');
    imposed.resize(300);
    assert.equal(imposed.bodyAttribute(), 'height: 100vh !important; color: red;');
    imposed.authorAttribute(null);
    imposed.resize(200);
    assert.equal(imposed.bodyAttribute(), null);
});

test('a box placed against an authored 500 px body counts where it is drawn, not where a 20 px root ends', () => {
    // body{height:500px;position:relative;margin:0} #root{height:20px} with a control placed at the body's bottom
    // (480 px down, 20 px tall): the document ends at the control, 500 px.
    const page = { floor: 0, wholeDocument: true, innerHeight: 320, imposedBodyHeight: 500, initialHeight: 20,
        paddingBottom: 0, borderBottom: 0 };
    const anchored = resizeHarness({ ...page, placed: [{ top: 480, height: 20 }] });
    assert.deepEqual(anchored.messages.map((item) => item.height), [500]);
    assert.deepEqual(anchored.styleWrites, { set: 2, remove: 0 }, 'nothing but overflow is written to measure it');
    // Its negative control: with nothing placed in it the same body holds its 20 px root; its own 500 px are not content.
    assert.deepEqual(resizeHarness(page).messages.map((item) => item.height), [20]);
    // A placed box's own placed box lies in its overflow and counts there (a panel 400 px tall below a 20 px bar).
    const nested = resizeHarness({ ...page, placed: [{ top: 0, height: 20, children: [{ top: 20, height: 400 }] }] });
    assert.deepEqual(nested.messages.map((item) => item.height), [420]);
    // A body with no height of its own: a box placed against it below the root counts too (8 px margins around).
    const unsized = resizeHarness({ floor: 0, wholeDocument: true, initialHeight: 20, paddingBottom: 0, borderBottom: 0,
        bodyTop: 8, marginBottom: 8, placed: [{ top: 108, height: 300 }] });
    assert.deepEqual(unsized.messages.map((item) => item.height), [416]);
});

test('what a board document places at its frame\'s bottom follows the frame, and margins outside it stop the growth', () => {
    // A 100vh body (no margins) with a 20 px bar placed at its bottom: the bar ends where the frame ends, so the
    // document rests at the frame it has, measured again whenever the frame resizes.
    const flush = resizeHarness({ floor: 0, wholeDocument: true, innerHeight: 320, imposedFollowsViewport: true,
        initialHeight: 40, paddingBottom: 0, borderBottom: 0, placed: [{ top: (inner) => inner - 20, height: 20 }] });
    assert.deepEqual(settleBoard(flush, 0), [320]);
    // The default 8 px margins outside the body put the bar 8 px below the frame, with 8 px under it: growing cannot
    // reach it, so after two equal asks the frame stops and the document scrolls, the bar reachable.
    const margined = resizeHarness({ floor: 0, wholeDocument: true, maxHeight: 8192, innerHeight: 320,
        imposedFollowsViewport: true, initialHeight: 40, paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8,
        placed: [{ top: (inner) => 8 + inner - 20, height: 20 }] });
    const climbed = settleBoard(margined, 0, 12);
    assert.deepEqual(climbed.slice(0, 3), [336, 352, 352]);
    assert.equal(climbed.length, 3);
    assert.equal(margined.messages.at(-1).held, 16);
    assert.equal(margined.overflowState().html.value, '', 'stopped: it scrolls');
    // Only a board frame listens for its frame's resize, until it is disposed.
    assert.equal(margined.listeners.has('resize'), true);
    // and for what changes it with no change to the root's size: a transition or an animation ending, a loaded image.
    assert.deepEqual([...margined.documentListeners.keys()].sort(), ['animationend', 'focusin', 'focusout', 'load', 'loadedmetadata', 'pointerout', 'pointerover', 'transitionend']);
    margined.dispose();
    assert.equal(margined.listeners.has('resize'), false);
    assert.equal(margined.documentListeners.size, 0);
    assert.equal(resizeHarness().listeners.has('resize'), false);
    assert.equal(resizeHarness().documentListeners.size, 0);
});

test('an ordinary board body that happens to be as tall as the frame keeps its margins', () => {
    // Content grown to exactly the frame (216 px of root under a 216 px frame, 8 px margins): 232 px, its bottom
    // margin and the content's last 8 px included, not the frame's own 216.
    const fits = resizeHarness({ floor: 0, wholeDocument: true, innerHeight: 216, initialHeight: 216,
        paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8 });
    assert.deepEqual(fits.messages.map((item) => item.height), [232]);
    // A frame in a message keeps its measure (the root's box) for the same documents.
    assert.deepEqual(resizeHarness({ floor: 320, innerHeight: 216, initialHeight: 216, paddingBottom: 0, borderBottom: 0,
        bodyTop: 8, marginBottom: 8 }).messages.map((item) => item.height), [216]);
});

test('a board document released at its ceiling is measured with its scrollbar hidden once it might fit', () => {
    // body margin 20px 8px 0 over a 210 px paragraph with 16 px margins, which collapse through the body while its
    // overflow is released: 262 px hidden, 230 px measured released. The owner's 240 px releases it at once.
    const harness = resizeHarness({ floor: 0, maxHeight: 240, wholeDocument: true, initialHeight: 210, bodyTop: 20,
        paddingBottom: 0, borderBottom: 0, childMargins: 32, marginsCollapseWhenReleased: true });
    assert.deepEqual(harness.messages.map((item) => item.height), [262]);
    assert.equal(harness.overflowState().html.value, '');
    // Show all: the released document looks short enough, so it is measured hidden, where it is 262 px: no new
    // (shorter) height, and the frame, already at 262 px, keeps the paragraph's last margin.
    harness.viewport({ floor: 0, ceiling: 8192 });
    assert.equal(harness.overflowState().html.value, 'hidden');
    assert.deepEqual(harness.messages.map((item) => item.height), [262]);
    // Back: released again without a report.
    harness.viewport({ floor: 0, ceiling: 240 });
    assert.equal(harness.overflowState().html.value, '');
    assert.deepEqual(harness.messages.map((item) => item.height), [262]);
    // Paired shrink while released: 150 px of paragraph is 202 px hidden (170 released), reported hidden.
    harness.resize(150);
    assert.equal(harness.overflowState().html.value, 'hidden');
    assert.deepEqual(harness.messages.map((item) => item.height), [262, 202]);
    // And growing past the ceiling again releases it, with the hidden measure reported.
    harness.resize(400);
    assert.equal(harness.overflowState().html.value, '');
    assert.deepEqual(harness.messages.map((item) => item.height), [262, 202, 452]);
});

test('a board frame keeps its loading height until something is drawn, however short, and shrinks back after', () => {
    const page = { floor: 0, maxHeight: 8192, wholeDocument: true, paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8 };
    // An empty document (a classic entry that has not drawn yet, or never will) measures 16 px of margins: not a report.
    const empty = resizeHarness({ ...page, initialHeight: 0 });
    assert.deepEqual(empty.messages, []);
    empty.resize(0);
    assert.deepEqual(empty.messages, []);
    // What it draws later is reported as it appears.
    empty.resize(180);
    assert.deepEqual(empty.messages.map((item) => item.height), [196]);
    // Once drawn, a document may empty out again and is measured as such.
    empty.resize(0);
    assert.deepEqual(empty.messages.map((item) => item.height), [196, 16]);
    // A box placed against the body with nothing in the root counts as drawn.
    const placed = resizeHarness({ ...page, initialHeight: 0, placed: [{ top: 10, height: 30 }] });
    assert.deepEqual(placed.messages.map((item) => item.height), [48]);
});

test('a board frame reports nothing before the module\'s entry has run; a frame in a message reports at once', () => {
    // The host's empty page before the entry: 0 px of root between 8 px margins measures 16 px, and is not the widget.
    const page = { floor: 0, maxHeight: 8192, initialHeight: 0, paddingBottom: 0, borderBottom: 0, bodyTop: 8, marginBottom: 8 };
    const board = resizeHarness({ ...page, wholeDocument: true, entryRan: false });
    board.resize(0);
    board.viewport({ floor: 0, ceiling: 600 });
    assert.deepEqual(board.messages, [], 'the frame keeps its loading height');
    // Until then its document scrolls in that height: what a module draws before its top-level await ends is reachable.
    board.resize(900);
    assert.deepEqual(board.messages, []);
    assert.deepEqual(board.overflowState(), { html: { value: '', priority: '' }, body: { value: '', priority: '' } });
    assert.deepEqual(board.styleWrites, { set: 0, remove: 0 });
    // The entry drew 300 px synchronously, then the host's mark: measured from there on, its scrollbar hidden as it fits.
    board.notify(300);
    board.entryRan();
    assert.deepEqual(board.messages.map((item) => item.height), [316]);
    assert.equal(board.overflowState().html.value, 'hidden');
    board.entryRan();
    board.resize(120);
    assert.deepEqual(board.messages.map((item) => item.height), [316, 136], 'genuinely short content shrinks it after');
    // A frame in a message (320 px floor) reports from the start, mark or not.
    const message = resizeHarness({ ...page, floor: 320, initialHeight: 40, entryRan: false });
    assert.deepEqual(message.messages.map((item) => item.height), [40]);
});
