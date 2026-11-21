// The capsule machine: what every inline composer control shares. A capsule opens into its strip on the
// composer's line, a press pins it, a mouse may open it early, and the one hint above says what its value
// means. Model, Subagents, Context and Effort run on it (composer_model.js, composer_owner_controls.js,
// composer_subagents.js); the fit that keeps a strip on the line is capsule_fit.js, the open and close
// motion capsule_morph.js.
import { beginMorph, stopMorph } from './capsule_morph.js';
import { SEG_PAD, createCapsuleFit } from './capsule_fit.js';
import { createHint } from './ui_hint.js';

// When a capsule's change takes effect, said once for all three under each one's meaning (the hint's rule line).
export const CHANGE_RULE = 'Applies to new steps; running work finishes.';
const HOVER_OPEN_MS = 160;
// After the mouse leaves, the open strip waits this long before it closes. Still being there is the control, its hint
// and the corridor between the two, each a few pixels wider: the way to the hint is forgiven, the message field below
// is not (owner, Track09 #2873, 9 Nov 2026; was 1000 ms and a 32 px margin that reached into the field).
const HOVER_CLOSE_MS = 300;
const HOVER_ZONE_PX = 6;
const HOVER_MEDIA = '(hover: hover) and (pointer: fine)';
// An inline strip collapses in 0.4 s (style.css); the closed capsule's width has settled by then.
const COLLAPSE_MS = 450;

// Refitting changes the observed row/strip. Leave ResizeObserver delivery before writing
// layout, coalesce notifications, and release pending work with the composer's lifetime.
export function observeLayout(win, nodes, layout) {
    if (!nodes[0] || typeof win?.ResizeObserver !== 'function') return () => {};
    let frame = 0;
    const observer = new win.ResizeObserver(() => {
        if (!frame) frame = win.requestAnimationFrame(() => { frame = 0; layout(); });
    });
    nodes.forEach((node) => observer.observe(node));
    return () => {
        observer.disconnect();
        if (frame) win.cancelAnimationFrame(frame);
        frame = 0;
    };
}

/**
 * What the three capsules share: the open/pin state machine and the fit (capsule_fit.js). A
 * capsule opens INLINE into its strip, never as a dropdown or a panel, and the strip stays on the
 * composer's line: beside the other controls when it fits there, alone (they step aside, its own
 * name too if need be) when it does not, scrolling inside itself only when nothing else fits. One
 * capsule is open at a time: opening one closes the others at once (`peers`), so the fit measures
 * them as they stand closed.
 *
 * Hover is an accelerator only (`(hover: hover) and (pointer: fine)`): a press pins the strip
 * open and unpins it; touch and keyboard use the press. Hover never opens a capsule while
 * another one is open, since closing it would move this one under a still pointer.
 *
 * While a capsule is open, the one hint (ui_hint.js) stands above it and says what its value means
 * (`say(preview)` gives the words; a mouse resting on another value previews that one's), with
 * CHANGE_RULE under it unless `say` answers its own `rule`, and `onHintAction` for its action. The
 * hint lives inside the capsule, so a press, the pointer or focus there is in the capsule.
 *
 * `parts()` names what else is the capsule's own while it stands outside its element (Model's
 * suggestions float over the page): a press, the pointer or focus there is in the capsule too.
 */
export function createCapsule({
    el, row, kind, strip, segs, render, focusTarget, padMax = SEG_PAD.max, say = null, onHintAction = () => {},
    busy = () => false, onClosed = () => {}, afterOpen = () => {}, afterOverflow = afterOpen,
    onLayout = () => {}, onSettled = () => {}, parts = () => [], win, doc,
}) {
    const q = (selector) => el.querySelector(selector);
    const head = q('.chat-capsule-head');
    const body = q('.chat-capsule-body');
    // `hover` exists only with a mouse; the chat unit-test DOM carries no matchMedia at all.
    const hoverCapable = typeof win?.matchMedia === 'function' && win.matchMedia(HOVER_MEDIA)?.matches === true;
    const state = { open: false, pinned: false, collapsing: false, destroyed: false };
    const api = {
        el, kind, hoverCapable, peers: [], isOpen: () => state.open, isPinned: () => state.pinned,
        isSettled: () => !state.open && !state.collapsing,
    };
    el.dataset.open = 'false';
    head.setAttribute('aria-expanded', 'false');
    let hoverTimer = 0;
    let leaveTimer = 0;
    let lastPointer = null;

    // Focus keeps a hover-opened strip open only when the keyboard put it there: browsers report
    // a handle focused after a mouse press as :focus-visible, so the last input decides instead.
    let keyboardFocus = false;
    const hint = say ? createHint(el, { doc, onAction: onHintAction }) : null;
    if (hint) head.setAttribute('aria-describedby', hint.el.id);
    let preview = '';
    const sayHint = () => { if (hint) hint.say({ rule: CHANGE_RULE, ...say(preview) }); };
    const outer = () => parts().filter(Boolean);
    const own = (node) => el.contains(node) || outer().some((part) => part.contains(node));
    const stillHere = () => {
        try {
            return Boolean(el.matches(':hover') || outer().some((part) => part.matches(':hover'))
                || (keyboardFocus && own(doc.activeElement)));
        } catch { return false; }
    };
    // The open strip forgives a mouse that drifts a little: inside the zone (the control, its hint
    // and the corridor between them, plus a small margin) it stays; outside, it closes after the delay
    // unless the mouse comes back first. A pinned strip, a drag and keyboard focus keep it open regardless.
    function inZone(point) {
        if (!point) return false;
        const box = (node) => (node && !node.hidden && typeof node.getBoundingClientRect === 'function' ? node.getBoundingClientRect() : null);
        const near = (r) => Boolean(r) && point.x >= r.left - HOVER_ZONE_PX && point.x <= r.right + HOVER_ZONE_PX
            && point.y >= r.top - HOVER_ZONE_PX && point.y <= r.bottom + HOVER_ZONE_PX;
        const own = box(el), said = box(hint?.el);
        if (near(own) || near(said) || outer().some((part) => near(box(part)))) return true;
        if (!own || !said) return false;
        // The corridor: where the hint and the control overlap side to side, the gap between them.
        const corridor = { left: Math.max(own.left, said.left), right: Math.min(own.right, said.right),
            top: Math.min(own.bottom, said.bottom), bottom: Math.max(own.top, said.top) };
        return corridor.left < corridor.right && near(corridor);
    }
    function closeSoon() {
        clearTimeout(leaveTimer);
        startTracking();
        leaveTimer = setTimeout(() => {
            leaveTimer = 0;
            if (state.open && !state.pinned && !busy() && !stillHere() && !inZone(lastPointer)) setOpen(false);
        }, HOVER_CLOSE_MS);
    }
    let tracking = false;
    const onDocMove = (event) => {
        if (event.pointerType && event.pointerType !== 'mouse') return;
        lastPointer = { x: event.clientX, y: event.clientY };
        if (!state.open || state.pinned) { stopTracking(); return; }
        if (inZone(lastPointer)) { clearTimeout(leaveTimer); leaveTimer = 0; }
        else if (!leaveTimer) closeSoon();
    };
    const onDocLeave = () => { lastPointer = null; if (state.open && !state.pinned) closeSoon(); };
    function startTracking() {
        if (tracking || !hoverCapable) return;
        tracking = true;
        doc.addEventListener('pointermove', onDocMove);
        doc.documentElement?.addEventListener?.('pointerleave', onDocLeave);
    }
    function stopTracking() {
        if (!tracking) return;
        tracking = false;
        doc.removeEventListener('pointermove', onDocMove);
        doc.documentElement?.removeEventListener?.('pointerleave', onDocLeave);
    }

    /* --- fit: beside the other controls, alone, or scrolling (capsule_fit.js) --- */
    const { isRtl, measureName, measure, fit, onStripScroll, reveal, holdWidths, releaseWidths, settleClosed } =
        createCapsuleFit({ el, row, kind, strip, segs, padMax, isOpen: () => state.open, win, doc });

    // A layout change (opening, a resize, a new padding) moves the marks with the words at once;
    // only an owner's gesture animates them.
    function settle(update) {
        el.dataset.settling = 'true';
        update();
        void strip.offsetWidth;
        delete el.dataset.settling;
    }
    // An inline strip collapses with a transition; the closed capsule's width settles when it
    // ends (at once alone, scrolling, or snapped).
    let settleTimer = 0;
    function settled() {
        clearTimeout(settleTimer);
        settleTimer = 0;
        state.collapsing = false;
        releaseWidths();
        if (!state.destroyed) onSettled();
    }
    const onBodyTransitionEnd = (event) => {
        if (event.target !== body || event.propertyName !== 'max-width') return;
        if (state.collapsing) settled();
        else if (state.open) releaseWidths();
    };
    function setOpen(open, { focus = false } = {}) {
        if (state.open === open && !focus) return;
        if (open && !state.open) api.peers.forEach((peer) => peer.snapClosed());
        const flips = state.open !== open;
        const morph = flips && el.dataset.snap !== 'true' ? beginMorph({ el, row, win }) : null;  // alone, it grows in place
        const collapses = !open && state.open && el.dataset.fit === 'inline' && el.dataset.snap !== 'true';
        state.open = open;
        clearTimeout(settleTimer);
        state.collapsing = collapses;
        if (open) el.dataset.settling = 'true';
        if (open) fit();
        if (flips) holdWidths();
        el.dataset.open = String(open);
        head.setAttribute('aria-expanded', String(open));
        if (!open) {
            state.pinned = false;
            clearTimeout(hoverTimer);
            clearTimeout(leaveTimer);
            leaveTimer = 0;
            stopTracking();
            onClosed();
            settleClosed();
            if (collapses) settleTimer = setTimeout(settled, COLLAPSE_MS);
        }
        render();
        if (open) {
            void strip.offsetWidth;  // the opening frame lands without the marks' transitions
            delete el.dataset.settling;
            afterOpen();
        }
        preview = '';
        if (hint && open) { sayHint(); hint.show(); } else hint?.hide();
        if (open && focus) focusTarget()?.focus({ preventScroll: true });
        onLayout();
        if (!open && !collapses) settled();
        morph?.play(open);
    }
    // Closed at once, a running collapse included: the other capsule is opening and measures this one.
    function snapClosed() {
        stopMorph(el);  // its own strip, still growing or folding, goes too
        el.dataset.snap = 'true';
        setOpen(false);
        void el.offsetWidth;
        delete el.dataset.snap;
        if (state.collapsing) settled();
    }
    const onHeadClick = (event) => {
        const viaKeyboard = event.detail === 0;
        if (hoverCapable) {
            if (!state.open) { state.pinned = true; setOpen(true, { focus: viaKeyboard }); }
            else if (!state.pinned) state.pinned = true;
            else { state.pinned = false; setOpen(false); }
        } else {
            setOpen(!state.open, { focus: viaKeyboard && !state.open });
        }
    };
    // A press on another capsule is its own to answer: opening it closes this one at once,
    // before anything moves under the pointer.
    const onDocPointerDown = (event) => {
        keyboardFocus = false;
        if (state.open && !own(event.target) && !api.peers.some((peer) => peer.el.contains(event.target))) {
            state.pinned = false;
            setOpen(false);
        }
    };
    const onDocKeyDown = (event) => {
        keyboardFocus = true;
        if (event.key !== 'Escape' || !state.open || event.defaultPrevented) return;
        event.preventDefault();  // the capsule took this Escape: the Project room under it stays open
        state.pinned = false;
        setOpen(false);
        head.focus({ preventScroll: true });
    };
    const onEnter = (event) => {
        if (event.pointerType !== 'mouse') return;
        // The mouse, not the keyboard, is here now: a focus left by Escape or Tab no longer holds a strip hover opens.
        keyboardFocus = false;
        clearTimeout(leaveTimer);
        leaveTimer = 0;
        stopTracking();
        if (!state.open) {
            clearTimeout(hoverTimer);
            hoverTimer = setTimeout(() => {
                if (state.destroyed || api.peers.some((peer) => peer.isOpen())) return;
                // Hover opens only where the head stays under the mouse. Alone or scrolling, the
                // other controls step aside and the strip moves under a still pointer, so a click
                // meant to pin would land on a level: there the press opens it, as on touch.
                const fitted = measure();
                if (fitted && fitted.mode !== 'inline') return;
                setOpen(true);
            }, HOVER_OPEN_MS);
        }
    };
    const onLeave = (event) => {
        if (event.pointerType !== 'mouse') return;
        clearTimeout(hoverTimer);
        lastPointer = { x: event.clientX, y: event.clientY };
        if (!state.open || state.pinned) return;
        startTracking();
        if (!inZone(lastPointer)) closeSoon();
    };

    // The row and the strip change size without a window resize (a Project pane's divider, a
    // font that loads): measure again then, so the fit and the marks follow. A strip that starts
    // to scroll reveals the active target; one that already scrolls stays where the owner put it.
    const relayout = () => settle(() => {
        const scrolled = el.dataset.fit === 'overflow';
        if (state.open) fit();
        render();
        if (!scrolled && el.dataset.fit === 'overflow') afterOverflow();
        onStripScroll();
    });
    const onResize = () => { if (state.open) relayout(); };
    const stopObserving = observeLayout(win, [row, strip], () => { if (!state.destroyed) relayout(); });
    // A mouse resting on a value previews what it means; leaving the strip says the chosen one again.
    const previewed = (next) => { if (next !== preview) { preview = next; sayHint(); } };
    const onPreview = (event) => {
        if (event.pointerType === 'mouse' && state.open) previewed(event.target?.closest?.('.chat-seg')?.dataset?.mode || '');
    };
    const onPreviewEnd = () => previewed('');

    head.addEventListener('click', onHeadClick);
    body.addEventListener('transitionend', onBodyTransitionEnd);
    strip.addEventListener('scroll', onStripScroll, { passive: true });
    win?.addEventListener?.('resize', onResize);
    doc.addEventListener('pointerdown', onDocPointerDown, true);
    doc.addEventListener('keydown', onDocKeyDown);
    if (hoverCapable) {
        el.addEventListener('pointerenter', onEnter);
        el.addEventListener('pointerleave', onLeave);
        if (hint) {
            strip.addEventListener('pointerover', onPreview);
            strip.addEventListener('pointerleave', onPreviewEnd);
        }
    }

    return Object.assign(api, {
        setOpen, snapClosed, closeSoon, onStripScroll, reveal, measureName, isRtl, relayout, sayHint,
        destroy() {
            state.destroyed = true;
            stopMorph(el);
            clearTimeout(hoverTimer);
            clearTimeout(leaveTimer);
            clearTimeout(settleTimer);
            body.removeEventListener('transitionend', onBodyTransitionEnd);
            doc.removeEventListener('pointerdown', onDocPointerDown, true);
            doc.removeEventListener('keydown', onDocKeyDown);
            head.removeEventListener('click', onHeadClick);
            strip.removeEventListener('scroll', onStripScroll);
            win?.removeEventListener?.('resize', onResize);
            stopObserving();
            stopTracking();
            if (row.dataset?.ownerSolo === kind) delete row.dataset.ownerSolo;
            el.removeEventListener('pointerenter', onEnter);
            el.removeEventListener('pointerleave', onLeave);
            strip.removeEventListener('pointerover', onPreview);
            strip.removeEventListener('pointerleave', onPreviewEnd);
            hint?.destroy();
        },
    });
}
