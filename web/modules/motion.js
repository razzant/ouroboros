/** May anything move, and when has a motion ended (docs/DESIGN.md §10 "Motion").
 *
 * The stylesheet owns the surface motion's durations and curve (shell_motion.js keeps the handover's own beat).
 * `afterMotion(el)` waits for the element's own transition of one property: its `transitionend` or `transitioncancel`
 * (a child's, another property's, or one from a transition replaced while a new one runs, never ends it); else the
 * element's computed duration plus delay and one frame, because WebKit fires nothing for a transition it skipped (a
 * hidden element, a value that did not change, one interrupted before it ran); else, when nothing transitions (reduced
 * motion, `transition: none`), the next frame. Its promise never rejects and says how it ended. `reducedMotion()` is
 * the one reader of the system's motion preference, and `onReducedMotionChange` follows it while the app runs: the
 * desktop shell and the Telegram mini app have no reload key to pick a new one up. */

const REDUCE = '(prefers-reduced-motion: reduce)';

export function reducedMotion(win = globalThis.window) {
    try { return Boolean(win?.matchMedia?.(REDUCE)?.matches); } catch { return false; }
}

export function onReducedMotionChange(fn, win = globalThis.window) {
    let query = null;
    try { query = win?.matchMedia?.(REDUCE) || null; } catch { query = null; }
    if (!query?.addEventListener) return () => {};
    const listener = (event) => fn(Boolean(event.matches));
    query.addEventListener('change', listener);
    return () => query.removeEventListener('change', listener);
}

const toMs = (value) => {
    const n = parseFloat(value);
    if (!Number.isFinite(n)) return 0;
    return /ms\s*$/.test(String(value)) ? n : n * 1000;
};

// The longest duration plus delay the element's computed transitions give `property` (or `all`), in ms.
function transitionMs(el, property, win) {
    let style;
    try { style = win.getComputedStyle(el); } catch { return 0; }
    const list = (value, fallback) => String(value || fallback).split(',').map((part) => part.trim());
    const names = list(style.transitionProperty, 'all');
    const durations = list(style.transitionDuration, '0s');
    const delays = list(style.transitionDelay, '0s');
    let longest = 0;
    names.forEach((name, i) => {
        if (name !== property && name !== 'all') return;
        longest = Math.max(longest, toMs(durations[i % durations.length]) + toMs(delays[i % delays.length]));
    });
    return longest;
}

/** A motion token's duration in ms (`--motion-fade`), read from the stylesheet: script keeps no copy of it. */
export function motionMs(token, win = globalThis.window) {
    try { return toMs(win.getComputedStyle(win.document.documentElement).getPropertyValue(token).trim()); } catch { return 0; }
}

/** What closes leaves as its picture: a copy of `el` (no ids, inert, hidden from assistive tech) plays `frames` where
 * `el` stood while `el` itself goes at once, so a close decides nothing later than it did. Reduced motion, a dialog
 * (the top layer) or an engine without element animations: nothing. */
export function leaveAsPicture(el, frames, { duration = 0, easing = 'ease-in', win = globalThis.window } = {}) {
    if (!el?.isConnected || !duration || reducedMotion(win) || typeof el.animate !== 'function' || el.matches?.('dialog')) return null;
    const picture = el.cloneNode(true);
    for (const node of [picture, ...picture.querySelectorAll('[id]')]) node.removeAttribute('id');
    picture.setAttribute('aria-hidden', 'true');
    picture.setAttribute('data-leaving', '');
    picture.inert = true;
    picture.style.pointerEvents = 'none';
    el.after(picture);
    picture.animate(frames, { duration, easing, fill: 'forwards' }).finished.catch(() => {}).finally(() => picture.remove());
    return picture;
}

/** `{ done: Promise<'end'|'cancel'|'timeout'|'none'>, cancel() }` for `el`'s transition of `property`. */
export function afterMotion(el, { property = 'transform', win = globalThis.window } = {}) {
    let resolve;
    const done = new Promise((fn) => { resolve = fn; });
    if (!el || !win) { resolve('none'); return { done, cancel() {} }; }
    const frame = (fn) => (win.requestAnimationFrame ? win.requestAnimationFrame(fn) : (win.setTimeout ?? setTimeout)(fn, 16));
    let ended = false;
    let timer = 0;
    const finish = (how) => {
        if (ended) return;
        ended = true;
        el.removeEventListener?.('transitionend', onEnd);
        el.removeEventListener?.('transitioncancel', onEnd);
        if (timer) win.clearTimeout(timer);
        resolve(how);
    };
    // An end or a cancel that belongs to a transition this one replaced (the card sent home while it was still
    // opening) ends nothing while a transition of the property still runs.
    const running = () => (el.getAnimations?.() || []).some((a) => a.transitionProperty === property && a.playState === 'running');
    function onEnd(event) {
        if (event.target !== el || event.propertyName !== property || running()) return;
        finish(event.type === 'transitioncancel' ? 'cancel' : 'end');
    }
    const ms = transitionMs(el, property, win);
    if (ms > 0) {
        el.addEventListener?.('transitionend', onEnd);
        el.addEventListener?.('transitioncancel', onEnd);
        timer = win.setTimeout(() => frame(() => finish('timeout')), ms);
    } else frame(() => finish('none'));
    return { done, cancel: () => finish('cancel') };
}
