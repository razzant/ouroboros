/** The strip's travelling fill (docs/DESIGN.md "The strip"; ui.css .ui-segment-fill), one mechanism for every strip.
 *
 * `bindSegmentFill(group)` lays one fill behind the options of a `.ui-segment-group` (aria-hidden, never pressed) and
 * writes the chosen option's box onto the group as `--segment-fill-x/-y/-w/-h`. The chosen option is the one marked
 * `.active` or `aria-checked="true"` (every producer marks one of them). When the choice moves along one line the fill
 * travels there on `--motion-fade` with `--ease-sheet`; when the new choice stands on another line of a wrapped
 * track, on its first paint, after the track changes size or under reduced motion it is put in place, never sent on a
 * diagonal trip. With nothing chosen the fill is not shown. It follows the choice (one MutationObserver on the group),
 * the track's size (one ResizeObserver) and late fonts; it answers its dispose function, a second call on the same
 * group answers the first binding's, and a group that leaves the document lets its observers go.
 *
 * `watchSegmentFills(root)` binds every strip already under `root` and every one added later, so producers write
 * markup only: one watcher at boot covers the settings rows, the theme strip, the Evolution cycles, the trust choice
 * and renderSegmentedField alike. */

import { reducedMotion } from './motion.js';

const BOUND = new WeakMap();
const LIVE = new Set();     // the groups bound now, so a watcher can let go of those that left the document
const CHOSEN = '.ui-segment.active, .ui-segment[aria-checked="true"]';

function chosenOf(group) {
    for (const option of group.children) {
        if (option.matches?.(CHOSEN)) return option;
    }
    return null;
}

export function bindSegmentFill(group) {
    if (!group || BOUND.has(group)) return BOUND.get(group) || (() => {});
    const doc = group.ownerDocument;
    const win = doc?.defaultView;
    const fill = doc.createElement('span');
    fill.className = 'ui-segment-fill';
    fill.setAttribute('aria-hidden', 'true');
    let last = null;      // the box last written: {x, y, w, h, top} or null while nothing is chosen
    let frame = 0;
    let disposed = false;

    const place = (moved) => {
        frame = 0;
        if (disposed) return;
        if (!group.isConnected) { dispose(); return; }
        if (fill.parentNode !== group) group.append(fill);
        const option = chosenOf(group);
        const box = group.getBoundingClientRect();
        if (!option || box.width === 0) {
            fill.hidden = true;
            last = null;
            return;
        }
        const rect = option.getBoundingClientRect();
        const next = {
            x: rect.left - box.left - group.clientLeft,
            y: rect.top - box.top - group.clientTop,
            w: rect.width,
            h: rect.height,
            top: option.offsetTop,
        };
        // Along one line it travels; to another line, on first paint, after a resize or under reduced motion it lands.
        const travel = moved && last !== null && !fill.hidden && Math.abs(last.top - next.top) < 1 && !reducedMotion(win);
        group.dataset.segmentFill = travel ? 'travel' : 'still';
        fill.hidden = false;
        for (const [name, value] of [['x', next.x], ['y', next.y], ['w', next.w], ['h', next.h]]) {
            group.style.setProperty(`--segment-fill-${name}`, `${value}px`);
        }
        last = next;
    };
    const schedule = (moved) => {
        if (disposed) return;
        if (moved) { if (frame) win.cancelAnimationFrame(frame); place(true); return; }
        if (!frame) frame = win.requestAnimationFrame(() => place(false));
    };

    const mutations = new win.MutationObserver((records) => {
        // The fill's own attribute writes are not a change of choice.
        const choice = records.some((record) => record.target !== fill
            && (record.type === 'childList' ? [...record.addedNodes, ...record.removedNodes].some((node) => node !== fill) : true));
        if (choice) schedule(true);
    });
    mutations.observe(group, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'aria-checked', 'hidden'] });
    const Resize = win.ResizeObserver;
    let width = -1;
    let height = -1;
    const resize = Resize ? new Resize((entries) => {
        if (!group.isConnected) { dispose(); return; }
        const size = entries[entries.length - 1]?.contentRect;
        if (size && size.width === width && size.height === height) return;
        width = size?.width ?? -1;
        height = size?.height ?? -1;
        schedule(false);
    }) : null;
    resize?.observe(group);
    const fonts = doc.fonts;
    const onFonts = () => schedule(false);
    fonts?.addEventListener?.('loadingdone', onFonts);
    fonts?.ready?.then?.(onFonts);

    function dispose() {
        if (disposed) return;
        disposed = true;
        if (frame) win.cancelAnimationFrame(frame);
        mutations.disconnect();
        resize?.disconnect();
        fonts?.removeEventListener?.('loadingdone', onFonts);
        fill.remove();
        delete group.dataset.segmentFill;
        for (const name of ['x', 'y', 'w', 'h']) group.style.removeProperty(`--segment-fill-${name}`);
        BOUND.delete(group);
        LIVE.delete(group);
    }

    BOUND.set(group, dispose);
    LIVE.add(group);
    group.append(fill);
    fill.hidden = true;
    place(false);
    return dispose;
}

/** Bind every strip under `root` now and as strips come and go; answers a function that stops watching. */
export function watchSegmentFills(root = globalThis.document?.body) {
    if (!root) return () => {};
    const win = root.ownerDocument?.defaultView || globalThis.window;
    const scan = (node) => {
        if (node.nodeType !== 1) return;
        if (node.classList.contains('ui-segment-group')) bindSegmentFill(node);
        for (const group of node.getElementsByClassName('ui-segment-group')) bindSegmentFill(group);
    };
    scan(root);
    const observer = new win.MutationObserver((records) => {
        let removed = false;
        for (const record of records) {
            if (record.removedNodes.length) removed = true;
            record.addedNodes.forEach(scan);
        }
        if (removed) for (const group of [...LIVE]) if (!group.isConnected) BOUND.get(group)?.();
    });
    observer.observe(root, { childList: true, subtree: true });
    return () => observer.disconnect();
}
