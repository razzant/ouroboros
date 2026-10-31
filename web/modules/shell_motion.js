/** The phone shell's motion between places (docs/DESIGN.md §10 "Motion"; owner 20.10, 4A; 21.10 floating shell),
 * presentation only.
 *
 * On a phone the room is the card the drawer pushes aside. A place chosen in the open drawer selects its row at once
 * and changes inside that card as the drawer closes: the leaving room's words fade out and the arriving room fades in
 * with one breath while the drawer's own close brings the card home.
 *
 * `settled()` is the end of what the shell is doing: the room card's own transform transition (motion.js). The layer
 * stack waits for it before it makes the covered room inert, and the app before it puts away a room that was left.
 * Every animation here is WAAPI on the real elements and ends in the stylesheet's own state, so a cancel (a new
 * navigation, a finger on the drawer, the layout switching, the motion preference changing, the page going away)
 * leaves nothing behind. It chooses no destination, writes no history, reads nothing from the server and keeps no
 * timer. Docked it does nothing; under reduced motion there is only a short fade of the arriving room. */

import { afterMotion, onReducedMotionChange, reducedMotion } from './motion.js';

export const MOTION = Object.freeze({
    out: 100,        // the leaving room's words fade (ease-in)
    inDelay: 60,     // then the arriving room starts …
    in: 220,         // … and materialises: opacity on `ease`, one breath of its content on the sheet curve
    breath: 0.97,    // the arriving content's starting scale
    reducedIn: 160,  // reduced motion: the arriving room's fade, nothing else
});
const SHEET = 'cubic-bezier(0.32, 0.72, 0, 1)';  // --ease-sheet's own value, where the token cannot be read

export function createShellMotion({ doc = globalThis.document, win = globalThis.window, content,
    isStack = () => true } = {}) {
    let painted = Boolean(doc?.body?.classList.contains('nav-drawer-open'));
    let wait = null;      // the motion settled() answers for: { done, cancel }
    let beat = null;      // the handover in flight
    let disposed = false;
    const ease = () => {
        try { return win.getComputedStyle(doc.documentElement).getPropertyValue('--ease-sheet').trim() || SHEET; } catch { return SHEET; }
    };
    const animatable = (el) => typeof el?.animate === 'function';

    // ---- the settle: the card's own transform transition
    function arm(next) {
        const previous = wait;
        wait = next;
        previous?.cancel();   // the latest motion is the one waited for
        next.done.then(() => { if (wait === next) wait = null; });
    }
    function settled() {
        const now = Boolean(doc?.body?.classList.contains('nav-drawer-open'));
        if (now !== painted) {
            painted = now;
            if (content) arm(afterMotion(content, { property: 'transform', win }));
        }
        return wait ? wait.done : Promise.resolve('none');
    }

    // ---- one beat at a time: what it animates and marks, so the next navigation or its end undoes it all
    function newBeat() {
        finish();
        const b = { anims: [], covers: [], marks: [] };
        b.animate = (el, frames, timing) => {
            if (!animatable(el)) return null;
            const anim = el.animate(frames, timing);
            b.anims.push(anim);
            return anim;
        };
        b.mark = (el) => { if (el && !el.hasAttribute('data-leaving')) { el.setAttribute('data-leaving', ''); b.marks.push(el); } };
        beat = b;
        return b;
    }
    function finish() {
        const b = beat;
        if (!b) return;
        beat = null;
        for (const anim of b.anims) { try { anim.cancel(); } catch { /* gone */ } }
        b.covers.forEach((node) => node.remove());
        b.marks.forEach((el) => el.removeAttribute('data-leaving'));
    }

    // A room's words fade. Main lies under every other room, so its words fade under a cover of the room's own
    // background inside it (the keyframes inverted) instead of by the page's opacity: animating the opacity of a room
    // of thousands of elements restyles all of them in WebKit. A room that lies over another fades itself, as the one
    // under it must show through.
    function fade(b, el, frames, timing) {
        if (!el || el.id !== 'page-chat') { b.animate(el, frames, timing); return; }
        const cover = doc.createElement('div');
        cover.className = 'shell-cover';
        cover.setAttribute('aria-hidden', 'true');
        el.append(cover);
        b.covers.push(cover);
        b.animate(cover, frames.map((f) => ({ opacity: 1 - f.opacity })), timing);
    }

    /** A place chosen in the open drawer. `leaving`: what holds the old room's words (a page, the Project panel, a
     * retired Project instance); `arriving`: the new room's element, `breathe`: what of it takes the breath (the
     * room itself, or the panel's content where the panel is the card). */
    function handover({ leaving = [], arriving = null, breathe = [] } = {}) {
        if (disposed || !isStack()) return;
        const b = newBeat();
        if (reducedMotion(win)) {
            fade(b, arriving, [{ opacity: 0 }, { opacity: 1 }], { duration: MOTION.reducedIn, easing: 'ease-out', fill: 'backwards' });
            Promise.all(b.anims.map((anim) => anim.finished.catch(() => null))).then(() => { if (beat === b) finish(); });
            return;
        }
        for (const el of leaving) {
            // A page that lost .active stays displayed for the handover.
            if (el?.classList?.contains('page') && !el.classList.contains('active')) b.mark(el);
            fade(b, el, [{ opacity: 1 }, { opacity: 0 }], { duration: MOTION.out, easing: 'ease-in', fill: 'forwards' });
        }
        const timing = { delay: MOTION.inDelay, duration: MOTION.in, fill: 'both' };
        fade(b, arriving, [{ opacity: 0 }, { opacity: 1 }], { ...timing, easing: 'ease' });
        for (const el of breathe) {
            b.animate(el, [{ transform: `scale(${MOTION.breath})` }, { transform: 'none' }], { ...timing, easing: ease() });
        }
        Promise.all(b.anims.map((anim) => anim.finished.catch(() => null))).then(() => {
            const done = () => { if (beat === b) finish(); };
            if (typeof win?.requestAnimationFrame === 'function') win.requestAnimationFrame(done);
            else done();
        });
    }

    /** Ends whatever runs, at once: animations to the stylesheet's state, covers and marks gone. */
    function cancel() {
        finish();
    }

    const disposeReduce = onReducedMotionChange(() => cancel(), win);
    const onPageHide = () => cancel();
    win?.addEventListener?.('pagehide', onPageHide);

    return {
        settled,
        handover,
        cancel,
        destroy() {
            disposed = true;
            cancel();
            wait?.cancel();
            disposeReduce();
            win?.removeEventListener?.('pagehide', onPageHide);
        },
    };
}
