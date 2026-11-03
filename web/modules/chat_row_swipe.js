/** Swipe a message to the left to reply (owner 19.10, 4A; docs/DESIGN.md "Replies"), as in Telegram.
 *
 * A finger that moves a row offering Reply (a saved bubble, a work card) to the left takes the row with it, and a
 * reply mark appears at the room's right side, filling as the row nears the reply point. Crossing the point gives a
 * haptic (haptics.js `threshold`; again after pulling back under it), releasing past it replies exactly as the row's
 * Reply does, and the row settles back either way; past the point it gives like a rubber band. A rightward finger
 * stays the drawer's (drawer_drag.js); a field, a slider, media, selected words, anything that scrolls sideways and
 * an open menu keep their own finger. Every swipe has a visible double: Reply under the message, or in its menu.
 *
 * Touch Events, with every move cancelled once the axis is decided, so iOS cannot hand the finger to the feed's
 * scroll halfway (WebKit 203335). That listener sits on the feed from the start, because a listener added only when
 * the touch begins may be unable to cancel it on iOS (WebKit 184251), and only in the stack layout under a coarse
 * pointer, as the drawer's: elsewhere a touch on the feed never waits. A layout that leaves the stack mid-swipe ends it
 * without a reply. The gesture is tracked on the element it started on, which keeps receiving its touch after a
 * re-render removes it. */
import { BLOCKING_SELECTOR, FIELD_SELECTOR, SIDEWAYS_SELECTOR, chainKeepsGesture } from './drawer_drag.js';
import { haptic } from './haptics.js';

export const SWIPE = Object.freeze({
    slop: 10,         // px before the axis is decided
    replyAt: 56,      // px of pull that replies on release (Telegram: 45–60 pt on iOS, 50 dp on Android)
    give: 0.5,        // past the reply point the row follows the finger damped...
    giveMax: 32,      // ...approaching this many px more, never reaching them
    settle: 220,      // ms back to rest (style.css)
    clickGuard: 400,  // ms after a swipe during which the click that ends it is not a tap
    markGap: 12,      // px between the reply mark and the room's right edge
});

const ROW = '.chat-bubble, .chat-live-card:not(.subagent)';
const MARK = 32;  // px, the reply mark (style.css)

/** How far the row moves for `d` px of finger: 1:1 up to the reply point, then a rubber band. */
export function pull(d) {
    const x = Math.max(0, Number(d) || 0);
    if (x <= SWIPE.replyAt) return x;
    const over = x - SWIPE.replyAt;
    return SWIPE.replyAt + (over * SWIPE.giveMax * SWIPE.give) / (SWIPE.giveMax + SWIPE.give * over);
}

const touchIn = (list, id) => {
    for (let i = 0; i < (list?.length || 0); i += 1) if (list[i]?.identifier === id) return list[i];
    return null;
};

/**
 * `feed` is the room's transcript, `replyRef(row)` what its Reply would quote (null: the row offers none), and
 * `reply(ref)` that Reply. `icon` is the reply glyph's markup. Returns the disposer.
 */
export function createRowSwipe({
    feed, replyRef, reply, icon = '',
    isStack = () => false, isCoarse = () => false,
    // The layout owner's change notification (layout.js onLayoutChange): returns its disposer.
    onLayout = () => () => {},
    win = globalThis.window, doc = feed?.ownerDocument || globalThis.document,
    isBlocked = () => [...(doc?.querySelectorAll?.(BLOCKING_SELECTOR) || [])].some((el) => el.getClientRects?.().length),
    hasSelection = () => Boolean(String(win?.getSelection?.()?.toString?.() || '').length),
} = {}) {
    if (!feed || typeof replyRef !== 'function' || typeof reply !== 'function') return () => {};
    let g = null;
    let mark = null;
    let suppressUntil = 0;
    const now = () => (typeof win?.performance?.now === 'function' ? win.performance.now() : Date.now());
    const settling = new Map();  // row → the timer that ends its settle

    function eligible(target, row) {
        if (!row || !feed.contains(row)) return false;
        if (target.closest?.(FIELD_SELECTOR) || target.closest?.(SIDEWAYS_SELECTOR) || hasSelection()) return false;
        if (isBlocked() || chainKeepsGesture(target, { win, doc })) return false;
        return true;
    }

    // The mark stands at the room's right side, level with the row (kept inside the visible feed).
    function showMark(row) {
        if (!mark) {
            mark = doc.createElement('div');
            mark.className = 'chat-row-reply-mark';
            mark.setAttribute('aria-hidden', 'true');
            mark.innerHTML = icon;
        }
        const room = feed.getBoundingClientRect();
        const r = row.getBoundingClientRect();
        const top = Math.min(Math.max(r.top + r.height / 2 - MARK / 2, room.top + 8), room.bottom - MARK - 8);
        mark.style.left = `${room.right - SWIPE.markGap - MARK}px`;
        mark.style.top = `${top}px`;
        mark.style.setProperty('--reply-k', '0');
        delete mark.dataset.armed;
        delete mark.dataset.leaving;
        doc.body.append(mark);
    }
    function hideMark() {
        if (!mark?.isConnected) return;
        const leaving = mark;
        leaving.dataset.leaving = 'true';
        setTimeout(() => { if (leaving.dataset.leaving === 'true') leaving.remove(); }, SWIPE.settle);
    }

    function settleRow(row) {
        clearTimeout(settling.get(row));
        row.dataset.rowSwipe = 'settle';
        row.style.setProperty('--row-swipe', '0px');
        settling.set(row, setTimeout(() => {
            settling.delete(row);
            if (row.dataset.rowSwipe !== 'settle') return;
            delete row.dataset.rowSwipe;
            row.style.removeProperty('--row-swipe');
        }, SWIPE.settle + 40));
    }

    function listen(target) {
        target.addEventListener('touchmove', onMove, { passive: false });
        target.addEventListener('touchend', onEnd);
        target.addEventListener('touchcancel', onCancel);
    }
    function end() {
        const gg = g;
        g = null;
        if (gg) {
            gg.target.removeEventListener('touchmove', onMove);
            gg.target.removeEventListener('touchend', onEnd);
            gg.target.removeEventListener('touchcancel', onCancel);
        }
        return gg;
    }

    function onStart(event) {
        syncGuard();
        const count = event.touches?.length || 0;
        if (g) { finish(false); if (count > 1) return; }
        if (count !== 1 || !isStack() || !isCoarse()) return;
        const touch = event.changedTouches?.[0] || event.touches?.[0];
        const target = event.target;
        const row = target?.closest?.(ROW);
        if (!touch || !eligible(target, row)) return;
        const ref = replyRef(row);
        if (!ref) return;
        g = { id: touch.identifier, target, row, ref, x0: touch.clientX, y0: touch.clientY, mode: 'wait', lockX: 0,
            armed: false };
        listen(target);
    }

    function onMove(event) {
        if (!g) return;
        if ((event.touches?.length || 0) > 1) { finish(false); return; }
        const touch = touchIn(event.changedTouches, g.id) || touchIn(event.touches, g.id);
        if (!touch) return;
        const dx = touch.clientX - g.x0;
        const dy = touch.clientY - g.y0;
        if (g.mode === 'wait') {
            if (Math.abs(dy) >= SWIPE.slop && Math.abs(dy) >= Math.abs(dx)) { end(); return; }  // the feed scrolls
            if (Math.abs(dx) < SWIPE.slop || Math.abs(dx) <= Math.abs(dy)) return;
            if (dx > 0) { end(); return; }  // rightwards: the drawer's
            if (event.cancelable === false) { end(); return; }  // the browser already scrolls: the finger is its
            if (isBlocked()) { end(); return; }  // a long press opened the row's menu under the still finger
            event.preventDefault();
            g.mode = 'drag';
            g.lockX = touch.clientX;
            clearTimeout(settling.get(g.row));
            g.row.dataset.rowSwipe = 'drag';
            showMark(g.row);
            return;
        }
        if (event.cancelable !== false) event.preventDefault();
        const d = Math.max(0, g.lockX - touch.clientX);
        g.row.style.setProperty('--row-swipe', `${pull(d)}px`);
        mark?.style.setProperty('--reply-k', String(Math.min(1, d / SWIPE.replyAt)));
        const armed = d >= SWIPE.replyAt;
        if (armed !== g.armed) {
            g.armed = armed;
            if (mark) mark.dataset.armed = String(armed);
            if (armed) haptic('threshold');
        }
    }

    // The finger left: past the reply point it replies; the row settles back either way.
    function finish(byFinger) {
        const gg = end();
        if (!gg || gg.mode !== 'drag') return;
        settleRow(gg.row);
        hideMark();
        suppressUntil = now() + SWIPE.clickGuard;
        if (byFinger && gg.armed) reply(gg.ref);
    }
    function onEnd(event) {
        if (g && touchIn(event.changedTouches, g.id)) finish(true);
    }
    // The system took the touch: back to rest, no reply.
    function onCancel(event) {
        if (g && touchIn(event.changedTouches, g.id)) finish(false);
    }

    // The click that ends a swipe is not a tap on whatever lies under the finger.
    function onClick(event) {
        if (now() >= suppressUntil) return;
        suppressUntil = 0;
        event.stopPropagation();
        event.preventDefault();
    }

    // The feed never scrolls under a swiped row (registered up front: see the module comment).
    function onFeedMove(event) {
        if (g?.mode === 'drag' && event.cancelable !== false) event.preventDefault();
    }
    let guarded = false;
    function syncGuard() {
        const want = Boolean(isStack() && isCoarse());
        if (want === guarded) return;
        guarded = want;
        if (want) feed.addEventListener('touchmove', onFeedMove, { passive: false });
        else { feed.removeEventListener('touchmove', onFeedMove); finish(false); }  // a docked room has no swipe
    }
    feed.addEventListener('touchstart', onStart, { passive: true });
    syncGuard();
    const unwatchLayout = onLayout(syncGuard);
    doc.addEventListener('click', onClick, true);
    return () => {
        finish(false);
        for (const [row, timer] of settling) {
            clearTimeout(timer);
            delete row.dataset.rowSwipe;
            row.style.removeProperty('--row-swipe');
        }
        settling.clear();
        mark?.remove();
        feed.removeEventListener('touchstart', onStart);
        feed.removeEventListener('touchmove', onFeedMove);
        guarded = false;
        unwatchLayout?.();
        doc.removeEventListener('click', onClick, true);
    };
}
