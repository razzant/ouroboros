/** The drawer follows the finger (owner Q1=A; docs/DESIGN.md "Shell").
 *
 * A horizontal drag that starts anywhere on the room, the screen's edge and its controls included, opens the
 * drawer; a drag on the open drawer (or its scrim) closes it. Only a field, a slider, media, selected text and
 * anything that scrolls sideways keep their own touch (owner 19.10: from the edge, and from where the thumb is).
 *
 * Touch Events, so the finger stays on the drawer (owner 19.10: it came off halfway). Once the axis is decided,
 * every touchmove is cancelled, so the browser cannot start a scroll mid-drag: iOS WebKit would, and then cancel
 * the finger (WebKit 203335). That listener sits on the root element from the start, because a listener added only
 * when the touch begins may be unable to cancel it on iOS (WebKit 184251). The gesture itself is tracked on the
 * element it started on, which keeps receiving its touch even after a re-render removes it from the page (a live
 * card, a new message).
 * If the system still takes the touch, the drawer settles from where the finger left it, never stranded halfway.
 *
 * Motion is transform-only and driven by one number (`--drawer-p`, 0 closed … 1 open) that the scrim's opacity
 * follows, and with it whatever else moves with the drawer (`also`: on a phone the room, pushed aside as a card, and
 * the veil over its words). On release the drawer settles at the finger's speed (`--drawer-settle`). The owner of the drawer's state
 * commits the result, so history, focus and inert stay the layer stack's, and a gesture that does not commit leaves
 * no trace. The drag is a shortcut: every gesture has a visible control (☰ and ×). One constant switches it off in
 * one line: DRAWER_DRAG. While a pinch magnifies the page, a sideways finger pans it, as the browser means it to. */

import { pinchMagnified } from './layout.js';

export const DRAWER_DRAG = true;

export const DRAG = Object.freeze({
    slop: 10,            // px of movement before the axis is decided
    axisRatio: 1,        // |dx| > ratio·|dy| locks the horizontal axis: a thumb's arc is a sideways drag
    minTravel: 24,       // px of travel before a fling may commit
    flingVelocity: 0.3,  // px/ms
    openPast: 0.5,       // a slow release opens past this progress
    closePast: 0.3,      // a slow release closes once this much of the drawer has gone
    sampleWindow: 100,   // ms of velocity samples; a finger that rests longer before it lifts has no speed
    clickGuard: 400,     // ms after a drag during which the click that ends it is not a tap
    // The settle (style.css's curve starts at 2.25× its average speed): a fling keeps the finger's speed, a slow
    // release moves at least `settleSpeed`, within `settleMin`…`settleMax` (Ionic's iOS menu takes 400 ms).
    settleSpeed: 1.2,    // px/ms
    settleMin: 180,      // ms
    settleMax: 400,      // ms
    // Past fully open the drawer gives like a rubber band (Apple's curve, as in use-gesture): d·M·c / (M + c·d), which
    // keeps giving with every pixel of pull and approaches `rubberMax` px without reaching it. style.css paints the
    // drawer's surface 64 px past its edge to fill that gap.
    rubber: 0.15,
    rubberMax: 48,
});

// What a drag may not start on: a field (the finger moves its caret or picks), a slider, media (the finger scrubs).
// A button, a link or a summary under the finger is fine: a drag is not a tap, and the click that ends it is
// swallowed (onClick).
export const FIELD_SELECTOR = 'input, textarea, select, video, audio, [role="slider"], '
    + '[contenteditable]:not([contenteditable="false"])';
// What scrolls sideways itself: code, tables, tab strips, anything that declares it.
export const SIDEWAYS_SELECTOR = 'pre, code, table, .md-table-wrap, .app-tab-strip, [data-scroll-x]';
// What lies over the room and owns the finger while it is open (shown, not merely mounted): a dialog, the reader,
// a menu, the work list. A Project open on the phone is the room: the drag opens the drawer over it, as ☰ does. A
// menu's picture leaving after it closed (motion.js leaveAsPicture) and a work list folding away are closed already.
export const BLOCKING_SELECTOR = 'dialog[open], .marketplace-modal-backdrop, .ui-popup:not([data-leaving]), .task-control-menu, '
    + '.chat-photo-menu:not([data-leaving]), .rw-shade:not([hidden]):not([inert])';

/** The chain under the finger keeps its own sideways gesture: a `touch-action: none` element (the effort strip) or a
 * scroller that can move sideways. */
export function chainKeepsGesture(target, { win = globalThis.window, doc = globalThis.document } = {}) {
    for (let el = target; el && el !== doc?.body; el = el.parentElement) {
        const cs = win?.getComputedStyle?.(el);
        if (!cs) continue;
        const touch = String(cs.touchAction || '');
        if (touch === 'none' || touch.includes('pan-x')) return true;
        if (/(auto|scroll)/.test(String(cs.overflowX || '')) && el.scrollWidth > el.clientWidth + 1) return true;
    }
    return false;
}

/** Progress for a finger that pulled the drawer `raw` of its width: 0…1 as is, a damped give past fully open. */
export function stretch(raw, width) {
    const p = Math.max(0, Number(raw) || 0);
    if (p <= 1 || !(width > 0)) return Math.min(1, p) || 0;
    const d = (p - 1) * width;
    const give = (d * DRAG.rubberMax * DRAG.rubber) / (DRAG.rubberMax + DRAG.rubber * d);  // px, under rubberMax
    return 1 + give / width;
}
const touchIn = (list, id) => {
    for (let i = 0; i < (list?.length || 0); i += 1) if (list[i]?.identifier === id) return list[i];
    return null;
};

/** How long the drawer takes to cover `remaining` px after a release at `velocity` px/ms. */
export function settleMs(remaining, velocity) {
    const speed = Math.max(Math.abs(Number(velocity) || 0), DRAG.settleSpeed);
    return Math.round(Math.min(DRAG.settleMax, Math.max(DRAG.settleMin, (2.25 * Math.abs(remaining)) / speed)));
}

export function bindDrawerDrag({
    sidebar, scrim, isOpen, setOpen, isStack, isCoarse,
    // The drag takes the scrim: whoever fades it out after a close (app.js) lets go of it.
    onGrab = () => {},
    // The layout owner's change notification (layout.js onLayoutChange): returns its disposer.
    onLayout = () => () => {},
    // A finger's release latched the drawer open or closed (haptics.js `snap`); a spring back to where it was is no
    // change, and the system taking the touch is not the finger's.
    onSnap = () => {},
    // What else follows the finger (the room card and its veil): read when a drag takes hold, so a room opened since
    // is in it; each gets the same --drawer-p, the dragging class and the settle.
    also = () => [],
    win = globalThis.window, doc = globalThis.document, enabled = DRAWER_DRAG,
    // The eligibility of a start, injectable for tests; the defaults read the real DOM.
    isField = (el) => Boolean(el?.closest?.(FIELD_SELECTOR)),
    isSideways = (el) => Boolean(el?.closest?.(SIDEWAYS_SELECTOR)),
    isBlocked = () => [...(doc?.querySelectorAll?.(BLOCKING_SELECTOR) || [])].some((el) => el.getClientRects?.().length),
    hasSelection = () => Boolean(String(win?.getSelection?.()?.toString?.() || '').length),
} = {}) {
    if (!enabled || !sidebar || !scrim) return () => {};
    let g = null;
    let suppressUntil = 0;
    let settleTimer = 0;
    let scrimTimer = 0;
    const now = () => (typeof win?.performance?.now === 'function' ? win.performance.now() : Date.now());
    const later = (ms, fn) => (typeof win?.setTimeout === 'function' ? win.setTimeout(fn, ms) : 0);
    const cancel = (timer) => { if (timer && typeof win?.clearTimeout === 'function') win.clearTimeout(timer); };

    function eligibleOpenStart(target) {
        if (sidebar.contains?.(target) || scrim.contains?.(target)) return false;
        if (isField(target) || isSideways(target) || hasSelection()) return false;
        if (isBlocked()) return false;
        if (chainKeepsGesture(target, { win, doc })) return false;
        return true;
    }

    function eligibleCloseStart(target) {
        if (!(sidebar.contains?.(target) || scrim.contains?.(target))) return false;
        return !target?.closest?.('input, textarea, select, [role="slider"]');
    }

    let moving = [sidebar, scrim];  // the drawer, its scrim, and what moves with them (set when a drag takes hold)
    const setProgress = (p) => {
        for (const el of moving) el.style.setProperty('--drawer-p', String(p));
        g.p = p;
    };
    const paint = (on) => {
        for (const el of moving) {
            el.classList.toggle('dragging', on);
            if (!on) el.style.removeProperty('--drawer-p');
        }
    };
    // The settle runs at the finger's speed; a tap on ☰ or × keeps style.css's own duration.
    function settle(ms) {
        cancel(settleTimer);
        const settling = moving;
        for (const el of settling) el.style.setProperty('--drawer-settle', `${ms}ms`);
        settleTimer = later(ms + 60, () => {
            for (const el of settling) el.style.removeProperty('--drawer-settle');
        });
    }
    // A scrim that faded out with a closed drawer leaves the page (unless the drawer opened meanwhile).
    function hideScrimAfter(ms) {
        cancel(scrimTimer);
        scrimTimer = later(ms, () => { if (!g && !isOpen()) scrim.hidden = true; });
    }

    // The gesture's own listeners, on the element the touch started on.
    function listen(target) {
        target?.addEventListener?.('touchmove', onMove, { passive: false });
        target?.addEventListener?.('touchend', onEnd);
        target?.addEventListener?.('touchcancel', onCancel);
    }
    function unlisten(target) {
        target?.removeEventListener?.('touchmove', onMove);
        target?.removeEventListener?.('touchend', onEnd);
        target?.removeEventListener?.('touchcancel', onCancel);
    }
    function end() {
        const gg = g;
        g = null;
        if (gg) unlisten(gg.target);
        return gg;
    }

    function onStart(event) {
        syncGuard();
        const count = event.touches?.length || 0;
        // A second finger, or a new touch after a gesture whose end never came: back to the start.
        if (g) { restore(); if (count > 1) return; }
        if (count !== 1 || !isStack() || !isCoarse() || pinchMagnified(win)) return;
        const touch = event.changedTouches?.[0] || event.touches?.[0];
        const target = event.target;
        if (!touch || !target) return;
        const open = Boolean(isOpen());
        if (open ? !eligibleCloseStart(target) : !eligibleOpenStart(target)) return;
        g = { id: touch.identifier, target, startOpen: open, x0: touch.clientX, y0: touch.clientY, mode: 'wait',
            samples: [], lockX: 0, width: 0, p: open ? 1 : 0, viewWidth: win?.innerWidth || 0 };
        listen(target);
    }

    function onMove(event) {
        if (!g) return;
        if ((event.touches?.length || 0) > 1) { restore(); return; }
        const touch = touchIn(event.changedTouches, g.id) || touchIn(event.touches, g.id);
        if (!touch) return;
        const dx = touch.clientX - g.x0;
        const dy = touch.clientY - g.y0;
        if (g.mode === 'wait') {
            if (Math.abs(dy) >= DRAG.slop && Math.abs(dy) >= Math.abs(dx)) { end(); return; }  // the page scrolls
            if (Math.abs(dx) < DRAG.slop || Math.abs(dx) <= DRAG.axisRatio * Math.abs(dy)) return;
            if ((dx > 0) === g.startOpen) { end(); return; }  // the wrong way: nothing to move
            if (event.cancelable === false) { end(); return; }  // the browser already scrolls: the finger is its
            if (isBlocked()) { end(); return; }  // a menu opened under the still finger (a long press) keeps it
            event.preventDefault();
            g.mode = 'drag';
            g.lockX = touch.clientX;
            g.width = sidebar.getBoundingClientRect?.().width || 320;
            cancel(settleTimer);
            cancel(scrimTimer);
            for (const el of moving) el.style.removeProperty('--drawer-settle');
            moving = [sidebar, scrim, ...(also() || []).filter(Boolean)];
            for (const el of moving) el.style.removeProperty('--drawer-settle');
            onGrab();
            if (!g.startOpen) scrim.hidden = false;
            paint(true);
            setProgress(g.startOpen ? 1 : 0);
            return;
        }
        if (event.cancelable !== false) event.preventDefault();
        const t = event.timeStamp ?? now();
        g.samples.push([t, touch.clientX]);
        g.samples = g.samples.filter((sample) => t - sample[0] <= DRAG.sampleWindow);
        g.lastX = touch.clientX;
        setProgress(stretch((g.startOpen ? 1 : 0) + (touch.clientX - g.lockX) / g.width, g.width));
    }

    function onEnd(event) {
        if (!g) return;
        const touch = touchIn(event.changedTouches, g.id);
        if (touch) release(touch.clientX, true, event.timeStamp);
    }

    /** The finger left at `x` at time `at` (an end, or the system taking the touch): settle at the nearest state the
     * finger's speed points to, never stranded halfway. A finger that rested longer than the sample window before it
     * left has no speed: where it stopped decides. */
    function release(x, byFinger = false, at = now()) {
        const gg = end();
        if (!gg || gg.mode !== 'drag') return;  // a tap, not a drag
        const first = gg.samples[0];
        const last = gg.samples[gg.samples.length - 1];
        const moving = Boolean(last) && at - last[0] <= DRAG.sampleWindow;
        const velocity = moving && last[0] > first[0] ? (last[1] - first[1]) / (last[0] - first[0]) : 0;
        const travel = Math.abs((Number.isFinite(x) ? x : gg.lastX ?? gg.x0) - gg.x0);
        let open;
        if (Math.abs(velocity) >= DRAG.flingVelocity && travel >= DRAG.minTravel) open = velocity > 0;
        else open = gg.startOpen ? gg.p > 1 - DRAG.closePast : gg.p >= DRAG.openPast;
        const ms = settleMs((open ? 1 - gg.p : gg.p) * gg.width, velocity);
        settle(ms);
        paint(false);
        if (open !== gg.startOpen) {
            // The latch is felt as the finger leaves, before the state's commit does its work (owner 20.10 3A).
            if (byFinger) onSnap(open);
            setOpen(open);
        } else if (!open) hideScrimAfter(ms);  // sprang back closed: the scrim fades with it
        suppressUntil = now() + DRAG.clickGuard;
    }

    // The system took the touch (an incoming call, Telegram's edge, an iOS gesture): the drawer still settles from
    // where the finger left it.
    function onCancel(event) {
        if (!g) return;
        const touch = touchIn(event.changedTouches, g.id);
        if (touch) release(touch.clientX ?? g.lastX, false, event.timeStamp);
    }

    /** A second finger or a new width: back to the start, no history change. */
    function restore() {
        const gg = end();
        if (!gg || gg.mode !== 'drag') return;
        const ms = settleMs((gg.startOpen ? 1 - gg.p : gg.p) * gg.width, 0);
        settle(ms);
        paint(false);
        if (!gg.startOpen) hideScrimAfter(ms);
    }

    // The page itself never scrolls under a dragged drawer (registered up front: see the module comment).
    function onPageMove(event) {
        if (g?.mode === 'drag' && event.cancelable !== false) event.preventDefault();
    }
    // A height-only change is a browser bar or the keyboard moving (Safari, Telegram); a new width is a new layout.
    function onResize() {
        syncGuard();
        if (g && (win?.innerWidth || 0) !== g.viewWidth) restore();
    }

    // The click that ends a drag is not a tap on whatever lies under the finger.
    function onClick(event) {
        if (now() >= suppressUntil) return;
        suppressUntil = 0;
        event.stopPropagation();
        event.preventDefault();
    }

    // On the root element, not window, document or body: iOS treats those as passive by default (WebKit 182521).
    // The blocking move listener is there only where a drag can start (a phone layout under a finger), mounted
    // when the layout changes, before any touch in it: elsewhere a touch's first move never waits for the page.
    const root = doc.documentElement || doc;
    let guarded = false;
    function syncGuard() {
        const want = Boolean(isStack() && isCoarse());
        if (want === guarded) return;
        guarded = want;
        if (want) root.addEventListener('touchmove', onPageMove, { passive: false });
        else root.removeEventListener('touchmove', onPageMove);
    }
    root.addEventListener('touchstart', onStart, { passive: true });
    syncGuard();
    // theme.js switches the layout on its own media-query change, after a resize handler has run: the guard follows
    // that switch; resize and the first touch are fallbacks.
    const unwatchLayout = onLayout(syncGuard);
    doc.addEventListener('click', onClick, true);
    win.addEventListener('resize', onResize);
    return () => {
        restore();
        end();
        cancel(settleTimer);
        cancel(scrimTimer);
        root.removeEventListener('touchstart', onStart);
        root.removeEventListener('touchmove', onPageMove);
        guarded = false;
        unwatchLayout?.();
        doc.removeEventListener('click', onClick, true);
        win.removeEventListener('resize', onResize);
    };
}
