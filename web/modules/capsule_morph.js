/** A strip that opens alone grows from its capsule (docs/DESIGN.md "Composer owner controls"; owner 19.10, 2A).
 *
 * Beside the other controls a strip grows by its width (style.css). Alone, the others leave the line at once, so a
 * width that moved would wrap the line halfway, and the strip used to appear at once. Here only opacity, a clip and
 * the head's offset move, on the real elements: the capsule's glass grows from where the closed capsule stood to its
 * open box (300 ms, the iOS sheet curve) with its ring riding on the glass's leading end, the controls that step aside
 * fade where they stood (110 ms), and the levels appear one after another. Closing runs back, faster (220 ms): the
 * glass folds into the closed capsule, the ring and value ride back, and the controls fade in. The line never wraps
 * for a frame. A page without layout, reduced motion and a capsule the other one closes at once (data-snap) do not
 * move. */

export const MORPH = Object.freeze({
    open: 300,     // ms, the glass grows to the open strip
    close: 220,    // ms, and folds back
    leave: 110,    // ms, the controls that step aside fade out where they stood
    shadow: 14,    // px of the glass's shadow kept above and below it while it is clipped
    ease: 'cubic-bezier(0.32, 0.72, 0, 1)',
});
const LEAVE_PROPS = ['--leave-x', '--leave-y', '--leave-w', '--leave-h', '--leave-display'];

const running = new WeakMap();  // capsule → its morph's stop()
const fade = (node, from, to, timing) =>
    node.animate([{ opacity: from }, { opacity: to }], { easing: 'ease-out', ...timing });

/**
 * Before a capsule's strip opens or closes: the line as it stands. A closing strip remembers whether it had the line
 * alone; an opening one is judged by its fit once open. Null when nothing should move.
 */
export function beginMorph({ el, row, win = globalThis.window }) {
    running.get(el)?.();  // a morph still running ends where it was going
    if (typeof el?.animate !== 'function' || typeof row?.getBoundingClientRect !== 'function') return null;
    try {
        if (win?.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return null;
    } catch { return null; }
    const box = boxIn(row);
    if (!box) return null;
    const shown = new Map();
    for (const kid of Array.from(row.children || [])) {
        if (kid === el || !kid.getClientRects?.().length) continue;
        shown.set(kid, { ...box(kid), display: win.getComputedStyle(kid).display });
    }
    const head = el.querySelector(':scope > .chat-capsule-head');
    const from = { ...box(el), head: head ? box(head) : null };
    const wasAlone = Boolean(el.dataset.fit) && el.dataset.fit !== 'inline';
    return { play: (open) => play({ el, row, open, from, shown, wasAlone, win }) };
}

// A node's box in the row's padding box, where an absolutely placed child of the row lands.
function boxIn(row) {
    const frame = row.getBoundingClientRect();
    if (!(frame.width > 0)) return null;
    const dx = frame.left + (Number(row.clientLeft) || 0);
    const dy = frame.top + (Number(row.clientTop) || 0);
    return (node) => {
        const r = node.getBoundingClientRect();
        return { left: r.left - dx, top: r.top - dy, width: r.width, height: r.height };
    };
}

function play({ el, row, open, from, shown, wasAlone, win }) {
    if (open ? !el.dataset.fit || el.dataset.fit === 'inline' : !wasAlone) return;  // beside the others: style.css
    const box = boxIn(row);
    if (!box) return;
    const doc = el.ownerDocument;
    const to = box(el);
    const animations = [];
    const cleanups = [];

    // The glass: the capsule's own surface steps back (data-morph) and this layer, clipped, stands for it.
    const surface = doc.createElement('span');
    surface.className = 'chat-capsule-surface';
    surface.setAttribute('aria-hidden', 'true');
    el.dataset.morph = open ? 'open' : 'close';
    el.prepend(surface);
    cleanups.push(() => {
        // The capsule's own surface returns at once, without its colour transition.
        el.style.transition = 'none';
        surface.remove();
        delete el.dataset.morph;
        void el.offsetWidth;
        el.style.transition = '';
    });
    const s = MORPH.shadow;
    const clip = (left, right) => `inset(-${s}px ${Math.max(0, right)}px -${s}px ${Math.max(0, left)}px round 999px)`;
    const right = (b) => b.left + b.width;
    let frames;
    if (open) {  // the layer is the open capsule; it starts as the closed one inside it
        frames = [clip(from.left - to.left, right(to) - right(from)), clip(0, 0)];
    } else {  // the layer is the open strip around the closed capsule; it ends as the closed capsule
        surface.style.setProperty('--morph-left', `${from.left - to.left - (Number(el.clientLeft) || 0)}px`);
        surface.style.setProperty('--morph-width', `${from.width}px`);
        frames = [clip(0, 0), clip(to.left - from.left, right(from) - right(to))];
    }
    const glass = surface.animate([{ clipPath: frames[0] }, { clipPath: frames[1] }],
        { duration: open ? MORPH.open : MORPH.close, easing: MORPH.ease, fill: 'forwards' });
    animations.push(glass);

    // The head (the ring, and the value while closed) rides from one home to the other with the glass's leading end.
    const head = el.querySelector(':scope > .chat-capsule-head');
    const shift = head && from.head ? from.head.left - box(head).left : 0;
    if (shift) {
        animations.push(head.animate([{ transform: `translateX(${shift}px)` }, { transform: 'none' }],
            { duration: open ? MORPH.open : MORPH.close, easing: MORPH.ease }));
    }
    if (open) {
        // The strip comes in once the glass has started to move, its levels one after another. They only fade: each
        // level is in its place from the first frame, under the selection's pill (nothing slides into place).
        const body = el.querySelector(':scope > .chat-capsule-body');
        if (body) animations.push(fade(body, 0, 1, { duration: 160, delay: 60, fill: 'backwards' }));
        Array.from(el.querySelectorAll('.chat-effort-seg, .chat-seg')).forEach((seg, i) => animations.push(
            fade(seg, 0, 1, { duration: 140, delay: 70 + 12 * i, fill: 'backwards' })));
        // The controls that stepped aside stand where they were, out of the line's flow, and fade.
        const leaving = [...shown].filter(([kid]) => kid.isConnected && !kid.getClientRects().length);
        if (leaving.length) {
            row.dataset.ownerLeaving = '';
            const release = (kid) => {
                delete kid.dataset.capsuleLeaving;
                LEAVE_PROPS.forEach((prop) => kid.style.removeProperty(prop));
            };
            for (const [kid, b] of leaving) {
                kid.dataset.capsuleLeaving = '';
                kid.style.setProperty('--leave-x', `${b.left}px`);
                kid.style.setProperty('--leave-y', `${b.top}px`);
                kid.style.setProperty('--leave-w', `${b.width}px`);
                kid.style.setProperty('--leave-h', `${b.height}px`);
                kid.style.setProperty('--leave-display', b.display);
                // Mostly gone before the head rides under them, and out of the page as soon as it has faded.
                const out = fade(kid, 1, 0, { duration: MORPH.leave, fill: 'forwards' });
                animations.push(out);
                out.finished.then(() => release(kid), () => {});
            }
            cleanups.push(() => {
                delete row.dataset.ownerLeaving;
                for (const [kid] of leaving) release(kid);
            });
        }
    } else {
        // The controls that come back fade in under the folding glass.
        for (const kid of Array.from(row.children || [])) {
            if (kid === el || shown.has(kid) || !kid.getClientRects?.().length) continue;
            animations.push(fade(kid, 0, 1, { duration: 160, delay: 40, fill: 'backwards' }));
        }
    }

    let done = false;
    const stop = () => {
        if (done) return;
        done = true;
        if (running.get(el) === stop) running.delete(el);
        for (const animation of animations) { try { animation.cancel(); } catch { /* gone */ } }
        for (const cleanup of cleanups) cleanup();
    };
    running.set(el, stop);
    glass.finished.then(stop, stop);
}

/** A running morph ends at once (the capsule goes away, the composer is torn down). */
export function stopMorph(el) {
    running.get(el)?.();
}
