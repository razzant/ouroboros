/** Reveal in this scroller alone, clear of its actual CSS fade, including rows inside the box but under the mask.
 * The sheet owns the edge size; Settings and the Dashboard read that same token, never a second pixel constant. */
const activeBindings = new WeakMap();
const restoringFocus = new WeakSet();
export const isRestoringFocus = control => restoringFocus.has(control);

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

export function revealPastFade(scroller, row, { axis = 'y' } = {}) {
    const style = scroller.ownerDocument?.defaultView?.getComputedStyle?.(scroller);
    const edge = parseFloat(style?.getPropertyValue?.('--scroll-fade-edge')) || 0;
    const bounds = scroller.getBoundingClientRect();
    const rect = row.getBoundingClientRect();
    if (axis === 'x') {
        const start = scroller.scrollLeft > 1 ? edge : 0;
        const end = scroller.scrollWidth - scroller.clientWidth - scroller.scrollLeft > 1 ? edge : 0;
        const left = bounds.left + (scroller.clientLeft || 0) + (parseFloat(style?.scrollPaddingInlineStart) || 0) + start;
        const right = bounds.left + (scroller.clientLeft || 0) + scroller.clientWidth
            - (parseFloat(style?.scrollPaddingInlineEnd) || 0) - end;
        if (rect.left < left) scroller.scrollLeft += rect.left - left;
        else if (rect.right > right) scroller.scrollLeft += rect.right - right;
        return;
    }
    const above = scroller.scrollTop > 1 ? edge : 0;
    const below = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 1 ? edge : 0;
    const top = bounds.top + (scroller.clientTop || 0) + (parseFloat(style?.scrollPaddingTop) || 0) + above;
    const bottom = bounds.top + (scroller.clientTop || 0) + scroller.clientHeight
        - (parseFloat(style?.scrollPaddingBottom) || 0) - below;
    if (rect.top < top) scroller.scrollTop += rect.top - top;
    else if (rect.bottom > bottom) scroller.scrollTop += rect.bottom - bottom;
}

/** Edge decoration follows actual hidden content, never masks a settled edge. Given a `region` name, a body that
 * scrolls on its axis is also a named region the keyboard can focus and scroll, as a cut table or notice stack is
 * (WebKit gives no other way into a scroller with nothing focusable inside); hidden or short, it is no tab stop. */
export function bindScrollFade(scroller, { axis = 'y', region = '' } = {}) {
    if (!scroller) return () => {};
    const existing = activeBindings.get(scroller);
    if (existing) return existing;
    const win = scroller.ownerDocument?.defaultView;
    // Decoration only: a host without the observers (a unit-test DOM) keeps the strip undecorated.
    if (!win?.ResizeObserver || !win.MutationObserver || !win.requestAnimationFrame) return () => {};
    let frame = null;
    let disposed = false;
    const update = () => {
        frame = null;
        if (disposed) return;
        const horizontal = axis === 'x';
        const position = horizontal ? Math.abs(scroller.scrollLeft) : scroller.scrollTop;
        const extent = horizontal ? scroller.scrollWidth - scroller.clientWidth
            : scroller.scrollHeight - scroller.clientHeight;
        scroller.toggleAttribute(horizontal ? 'data-scroll-start' : 'data-scroll-above', position > 1);
        scroller.toggleAttribute(horizontal ? 'data-scroll-end' : 'data-scroll-below', extent - position > 1);
        if (region && extent > 1 !== scroller.hasAttribute('tabindex')) markRegion(scroller, extent > 1 && region);
    };
    const schedule = () => {
        if (!disposed && frame === null) frame = win.requestAnimationFrame(update);
    };
    // Where a row stood in the box when focus left it: focus handed back to that row, which has not moved since (a
    // reader or menu closed by a key, a copy done), leaves the scroller where the owner put it, as the closing promised
    // (DESIGN "Document reading": focus returns to the opener at the same place).
    let left = null;
    const offset = (node) => node.getBoundingClientRect()[axis === 'x' ? 'left' : 'top']
        - scroller.getBoundingClientRect()[axis === 'x' ? 'left' : 'top'];
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
        const unmoved = left?.row === row && Math.abs(offset(row) - left.at) < 1;
        left = null;
        if (row && row !== scroller && !unmoved && !isRestoringFocus(row)
            && row.matches?.(':focus-visible') && scroller.contains?.(row)) {
            revealPastFade(scroller, row, { axis });
        }
        schedule();
    };
    const resize = new win.ResizeObserver(schedule);
    const observeContent = () => {
        resize.disconnect();
        resize.observe(scroller);
        for (const child of scroller.children) resize.observe(child);
        schedule();
    };
    // Direct child replacement changes the observed boxes; text/late media
    // within them changes their size and is covered by ResizeObserver.
    const mutations = new win.MutationObserver(observeContent);
    mutations.observe(scroller, { childList: true });
    scroller.addEventListener('scroll', schedule, { passive: true });
    scroller.addEventListener('focusin', onFocus);
    scroller.addEventListener('focusout', onBlur);
    observeContent();
    const dispose = () => {
        disposed = true;
        if (frame !== null) win.cancelAnimationFrame(frame);
        resize.disconnect();
        mutations.disconnect();
        scroller.removeEventListener('scroll', schedule);
        scroller.removeEventListener('focusin', onFocus);
        scroller.removeEventListener('focusout', onBlur);
        left = null;
        if (activeBindings.get(scroller) === dispose) activeBindings.delete(scroller);
    };
    activeBindings.set(scroller, dispose);
    return dispose;
}

function markRegion(scroller, name) {
    if (!name) ['tabindex', 'role', 'aria-label'].forEach((attr) => scroller.removeAttribute(attr));
    else {
        scroller.tabIndex = 0;
        scroller.setAttribute('role', 'region');
        scroller.setAttribute('aria-label', name);
    }
}
