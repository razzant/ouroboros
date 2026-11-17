/** Reveal in this scroller alone, clear of its actual CSS fade, including rows inside the box but under the mask.
 * The sheet owns the edge size; Settings and the Dashboard read that same token, never a second pixel constant. */
export function revealPastFade(scroller, row) {
    const style = scroller.ownerDocument?.defaultView?.getComputedStyle(scroller);
    const edge = parseFloat(style?.getPropertyValue?.('--scroll-fade-edge')) || 0;
    const bounds = scroller.getBoundingClientRect();
    const rect = row.getBoundingClientRect();
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
    observeContent();
    return () => {
        disposed = true;
        if (frame !== null) win.cancelAnimationFrame(frame);
        resize.disconnect();
        mutations.disconnect();
        scroller.removeEventListener('scroll', schedule);
    };
}

function markRegion(scroller, name) {
    if (!name) ['tabindex', 'role', 'aria-label'].forEach((attr) => scroller.removeAttribute(attr));
    else {
        scroller.tabIndex = 0;
        scroller.setAttribute('role', 'region');
        scroller.setAttribute('aria-label', name);
    }
}
