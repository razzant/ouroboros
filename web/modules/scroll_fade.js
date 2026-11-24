/** Scrollers publish live edge facts to the shared mask in ui.css; a static fold uses data-fold-clipped instead. */
const activeBindings = new WeakMap();
const restoringFocus = new WeakSet();
export const isRestoringFocus = control => restoringFocus.has(control);

/** Swap or release the decoration owned by a replaced DOM region: the binding live on `previous`, whoever made it
 * (a module that rebinds its scroller leaves any earlier disposer spent). A kept node keeps its live binding. */
export function replaceScrollFade(previous, next = null, options) {
    if (previous !== next) activeBindings.get(previous)?.();
    if (next) bindScrollFade(next, options);
}

// A repaint restores the keyboard's control even when the owner has scrolled away from it.
export function restoreFocusWithoutReveal(control) {
    if (!control) return;
    restoringFocus.add(control);
    try {
        control.focus({ preventScroll: true });
    } finally {
        restoringFocus.delete(control);
    }
}

const px = value => parseFloat(value) || 0;

function scrollEdges(size, scrollSize, offset, overflow, rtl = false) {
    const extent = scrollSize - size;
    const scrolls = size > 0 && extent > 1 && (!overflow || /^(auto|scroll|overlay)$/.test(overflow));
    const at = (rtl ? -1 : 1) * offset;
    return { scrolls, start: scrolls && at > 1, end: scrolls && extent - at > 1, rtl };
}

export function measureEdges(scroller, style, horizontal) {
    return horizontal ? scrollEdges(scroller.clientWidth, scroller.scrollWidth, scroller.scrollLeft,
        style?.overflowX, style?.direction === 'rtl')
        : scrollEdges(scroller.clientHeight, scroller.scrollHeight, scroller.scrollTop, style?.overflowY);
}

function revealDelta(scroller, style, bounds, rect, horizontal) {
    const { start, end, rtl } = measureEdges(scroller, style, horizontal);
    const [near, far] = rtl ? [end, start] : [start, end];
    const [nearEdge, farEdge] = horizontal ? ['left', 'right'] : ['top', 'bottom'];
    const length = horizontal ? scroller.clientWidth : scroller.clientHeight;
    if (!length) return 0;
    const border = (horizontal ? scroller.clientLeft : scroller.clientTop) || 0;
    const maskBox = (horizontal ? scroller.offsetWidth : scroller.offsetHeight) || length;
    const token = name => px(style?.getPropertyValue?.(name));
    const inset = !horizontal && near ? Math.min(maskBox, token('--scroll-fade-top-clear')) : 0;
    const edge = token(horizontal ? '--scroll-fade-inline-edge' : '--scroll-fade-edge');
    const limit = token('--scroll-fade-limit');
    const depth = limit ? Math.min(edge, (maskBox - inset) * limit) : edge;
    const nearPadding = px(horizontal ? style?.scrollPaddingLeft : style?.scrollPaddingTop);
    const farPadding = px(horizontal ? style?.scrollPaddingRight : style?.scrollPaddingBottom);
    // Leave the owner's scroll padding inside the clear part. The mask starts at the border box, not the scrollport.
    const min = bounds[nearEdge] + border + nearPadding + (near ? Math.max(0, inset + depth - border) : 0);
    const max = bounds[nearEdge] + border + length - farPadding - (far ? Math.max(0, depth - (maskBox - border - length)) : 0);
    return rect[nearEdge] < min ? rect[nearEdge] - min : Math.max(0, rect[farEdge] - max);
}

/** Reveal in this scroller alone, including a row in the scrollport but under its actual mask. */
export function revealPastFade(scroller, row, { axis = 'y' } = {}) {
    const style = scroller.ownerDocument?.defaultView?.getComputedStyle?.(scroller);
    const bounds = scroller.getBoundingClientRect();
    const rect = row.getBoundingClientRect();
    const dx = axis === 'x' || axis === 'both' ? revealDelta(scroller, style, bounds, rect, true) : 0;
    const dy = axis === 'x' ? 0 : revealDelta(scroller, style, bounds, rect, false);
    if (dx) scroller.scrollLeft += dx;
    if (dy) scroller.scrollTop += dy;
}

const edgeAttrs = { x: ['data-scroll-start', 'data-scroll-end', 'data-scroll-rtl'], y: ['data-scroll-above', 'data-scroll-below'] };
const regionAttrs = ['tabindex', 'role', 'aria-label'];

/** Follow actual scroll travel on the named axis (x, y or both), never a settled edge. An optional `region` names
 * an otherwise-unlabelled body while it scrolls: WebKit gives no keyboard entry to a short or non-focusable box. */
export function bindScrollFade(scroller, { axis = 'y', region = '' } = {}) {
    if (!scroller) return () => {};
    const existing = activeBindings.get(scroller);
    if (existing) return existing;
    const win = scroller.ownerDocument?.defaultView;
    // Decoration only: a host without the observers (a unit-test DOM) keeps the strip undecorated.
    if (!win?.ResizeObserver || !win.MutationObserver || !win.requestAnimationFrame) return () => {};
    const horizontal = axis === 'x' || axis === 'both';
    const vertical = axis !== 'x';
    let frame = null;
    let disposed = false;
    let ownsRegion = false;
    const update = () => {
        frame = null;
        if (disposed) return;
        const style = win.getComputedStyle?.(scroller);
        const x = horizontal && measureEdges(scroller, style, true);
        const y = vertical && measureEdges(scroller, style, false);
        if (x) {
            scroller.toggleAttribute(edgeAttrs.x[0], x.start);
            scroller.toggleAttribute(edgeAttrs.x[1], x.end);
            scroller.toggleAttribute(edgeAttrs.x[2], x.scrolls && x.rtl);
        }
        if (y) {
            scroller.toggleAttribute(edgeAttrs.y[0], y.start);
            scroller.toggleAttribute(edgeAttrs.y[1], y.end);
        }
        if (!region) return;
        if (x?.scrolls || y?.scrolls) {
            if (!regionAttrs.some(attr => scroller.hasAttribute(attr))) {
                markRegion(scroller, region);
                ownsRegion = true;
            }
        } else if (ownsRegion) {
            markRegion(scroller, '');
            ownsRegion = false;
        }
    };
    const schedule = () => {
        if (!disposed && frame === null) frame = win.requestAnimationFrame(update);
    };
    // Where a row stood in the box when focus left it: focus handed back to that row, which has not moved since (a
    // reader or menu closed by a key, a copy done), leaves the scroller where the owner put it, as the closing promised
    // (DESIGN "Document reading": focus returns to the opener at the same place).
    let left = null;
    const offset = (node) => {
        const row = node.getBoundingClientRect(), box = scroller.getBoundingClientRect();
        return [horizontal ? row.left - box.left : 0, vertical ? row.top - box.top : 0];
    };
    const onBlur = (event) => {
        const row = event.target;
        left = row && row !== scroller && scroller.contains?.(row) ? { row, at: offset(row) } : null;
    };
    const onFocus = (event) => {
        // Only focus the keyboard can see is brought clear of the fade. A pointer's focus precedes its click (moving the
        // pressed row would lose the release), and focus handed back after a pointer's menu or dialog must not drag the
        // list after a row that moved meanwhile (a pinned project's row going to the top). A row the keyboard gets back
        // where it stood stays there; one that moved meanwhile is brought clear.
        const row = event.target;
        const unmoved = left?.row === row && offset(row).every((at, i) => Math.abs(at - left.at[i]) < 1);
        left = null;
        if (row && row !== scroller && !unmoved && !isRestoringFocus(row)
            && row.matches?.(':focus-visible') && scroller.contains?.(row)) {
            revealPastFade(scroller, row, { axis });
        }
        schedule();
    };
    const resize = new win.ResizeObserver(schedule);
    const observed = new Set([scroller]);
    const observeChild = (child) => {
        if (observed.has(child)) return;
        observed.add(child);
        resize.observe(child);
    };
    const forgetChild = (child) => {
        if (!observed.has(child)) return;
        observed.delete(child);
        resize.unobserve(child);
    };
    resize.observe(scroller);
    for (const child of scroller.children) observeChild(child);
    schedule();
    // Only direct replacements change which boxes are observed. A sideways strip can also gain overflow from text
    // in a descendant that does not itself resize (a code line or KaTeX). Watch its own content, never the document.
    const mutations = new win.MutationObserver(records => {
        const touched = new Set();
        for (const record of records) {
            if (record.type !== 'childList' || record.target !== scroller) continue;
            for (const nodes of [record.addedNodes, record.removedNodes]) for (const node of nodes) if (node.nodeType === 1) touched.add(node);
        }
        // A batch can add a row and take it out again, or move it away and back: only where each ended decides.
        for (const child of touched) (child.parentNode === scroller ? observeChild : forgetChild)(child);
        schedule();
    });
    mutations.observe(scroller, { childList: true, subtree: horizontal, characterData: horizontal });
    scroller.addEventListener('load', schedule, true);  // late media may overflow a box without resizing it
    scroller.addEventListener('scroll', schedule, { passive: true });
    scroller.addEventListener('focusin', onFocus);
    scroller.addEventListener('focusout', onBlur);
    const dispose = () => {
        if (disposed) return;
        disposed = true;
        if (frame !== null) win.cancelAnimationFrame(frame);
        resize.disconnect();
        observed.clear();
        mutations.disconnect();
        scroller.removeEventListener('load', schedule, true);
        scroller.removeEventListener('scroll', schedule);
        scroller.removeEventListener('focusin', onFocus);
        scroller.removeEventListener('focusout', onBlur);
        if (horizontal) for (const attr of edgeAttrs.x) scroller.toggleAttribute(attr, false);
        if (vertical) for (const attr of edgeAttrs.y) scroller.toggleAttribute(attr, false);
        if (ownsRegion) markRegion(scroller, '');
        left = null;
        if (activeBindings.get(scroller) === dispose) activeBindings.delete(scroller);
    };
    activeBindings.set(scroller, dispose);
    return dispose;
}

function markRegion(scroller, name) {
    if (!name) regionAttrs.forEach((attr) => scroller.removeAttribute(attr));
    else {
        scroller.tabIndex = 0;
        scroller.setAttribute('role', 'region');
        scroller.setAttribute('aria-label', name);
    }
}

/** Shared folded-Markdown keyboard-stop owner (Updates preview + mirrored Project answers).
 *  Holds stopped targets out of sequential Tab order via data-fold-stop / tabindex="-1" without
 *  inert holes, preserving authored and scroller-owned tabindex in one per-node record. */
const FOLD_STOP_SELECTOR = 'a[href], area[href], button, input, select, textarea, summary, [tabindex], pre, .katex-display, .md-mermaid, .md-mermaid-scroll, .md-chart, .md-table-wrap, .scroll-fade-x';
const foldStops = new WeakMap();

const rawTabIndex = (el) => (el.hasAttribute('tabindex') ? el.getAttribute('tabindex') : null);

function writeTabIndex(el, value) {
    if (value == null) {
        if (el.hasAttribute('tabindex')) el.removeAttribute('tabindex');
        return;
    }
    const next = String(value);
    if (rawTabIndex(el) !== next) el.setAttribute('tabindex', next);
}

function holdFoldStop(el, byFold = true) {
    const held = foldStops.get(el);
    if (el.hasAttribute('data-fold-stop') && held) {
        if (byFold) held.byFold = true;
    } else {
        foldStops.set(el, { tab: el.hasAttribute('data-fold-stop') ? null : rawTabIndex(el), byFold });
        if (!el.hasAttribute('data-fold-stop')) el.setAttribute('data-fold-stop', '');
    }
    writeTabIndex(el, '-1');
}

function releaseFoldStop(el) {
    const held = foldStops.get(el);
    foldStops.delete(el);
    if (el.hasAttribute('data-fold-stop')) el.removeAttribute('data-fold-stop');
    writeTabIndex(el, held ? held.tab : null);
}

/** Synchronize fold keyboard stops under `root`. Predicate reads complete before any DOM write. */
export function syncFoldStops(root, active, shouldStop = null) {
    if (!root?.querySelectorAll) return;
    if (!active) {
        for (const el of root.querySelectorAll('[data-fold-stop]')) releaseFoldStop(el);
        return;
    }
    const candidates = [...root.querySelectorAll(FOLD_STOP_SELECTOR)];
    const targets = new Set(shouldStop ? candidates.filter(shouldStop) : candidates);
    for (const el of root.querySelectorAll('[data-fold-stop]')) {
        if (!targets.has(el)) releaseFoldStop(el);
    }
    for (const el of targets) holdFoldStop(el, true);
}

/** Update a scroller/region tabindex while respecting active fold stops and folded Project answers.
 *  When `owned` is omitted, claims ownership iff no underlying (pre-fold) tabindex exists; returns ownership. */
export function syncFoldTabIndex(el, desired, { folded = false, owned } = {}) {
    let held = el.hasAttribute('data-fold-stop') ? foldStops.get(el) : null;
    if (!folded && held && !held.byFold) {
        releaseFoldStop(el);
        held = null;
    }
    const stopped = el.hasAttribute('data-fold-stop');
    const owns = owned ?? ((stopped ? (held ? held.tab : null) : rawTabIndex(el)) === null);
    if (stopped || folded) {
        if (!held) holdFoldStop(el, stopped);
        if (owns) foldStops.get(el).tab = desired;
        writeTabIndex(el, '-1');
        return owns;
    }
    if (owns) writeTabIndex(el, desired);
    return owns;
}
