/** The chat's one reading intent. Readiness comes from data/paint owners, not
 * a frame deadline. Frames only settle geometry after those owners are ready.
 * No fetch, pager, message cache or timer lives here.
 */
import { DRAG } from './drawer_drag.js';  // a touch's axis is decided by the drawer's own rule
import { batch } from './layout_batch.js';  // reads the write defers run after it, before the place is restored

// A width that moves again within MOVE_FRAMES frames keeps moving (a drag, a live window resize); one that then
// holds WIDTH_HELD_FRAMES frames (about 200ms at 60Hz) has stopped: a drag paused that long is laid out once.
const MOVE_FRAMES = 3, WIDTH_HELD_FRAMES = 12;

export function createChatReadingPosition({ visible, alive, ready, feed, anchors, changed, afterWrite, activity, updateButton }) {
    let generation = 0, scheduled = false;
    let mutationDepth = 0;
    let viewportAnchor = null, width = feed.clientWidth;
    // Every room opens following its newest message (owner decision 2026-10-05).
    let intent = null;
    // dragging: 'held'/'released' feed scrollbar, or 'owed' to a downward gesture's animation.
    let approximate = false, dragging = '', claimed = null, scrolls = 0, arriving = null, relayout = false;
    // The feed's box at a drag's release: the drag's own last scrolls arrive in it, a resize's clamp in another.
    let releasedBox = '';
    const feedBox = () => `${feed.clientWidth}x${feed.clientHeight}`;
    // Where the app's own write (mutate) moved the feed while a drag was released: a scroll there is that write's.
    let placed = null;
    // WebKit also ends each step of an animated page scroll, and each batch of a
    // released drag's last scrolls: the owed scrolls end only once a frame
    // passes without another, and only this gesture's: a newer one's may not
    // have begun to arrive. A resize that waited on them lays out then; its own
    // wait looks again while they still arrive.
    const owedEnd = (wait = false) => {
        const token = generation, seen = scrolls;
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if ((dragging === 'owed' || dragging === 'released') && token === generation && scrolls === seen) dragging = '';
            if (!relayout) return;
            if (dragging !== 'owed') { relayout = false; if (alive()) state.reflow(); }
            else if (wait) owedEnd(true);
        }));
    };
    // While the feed's width keeps moving, its rows off the screen keep their size (style.css: lazy rows under
    // [data-resizing]) and only those near the screen are laid out again. A width that moves once (a window
    // jump, a layout switch) is laid out at once, as before. Once a moving width has held, every row is laid out at
    // the final width and the place is re-applied in that frame. The feed's own box counts, not its content width:
    // a scrollbar appearing as history fills the feed is no move, and neither is a feed shown again.
    let rowsWidth = feed.offsetWidth, sinceMove = -1, lazy = false;
    const countFrames = () => requestAnimationFrame(() => {
        if (++sinceMove < (lazy ? WIDTH_HELD_FRAMES : MOVE_FRAMES)) { countFrames(); return; }
        sinceMove = -1;
        if (!lazy) return;
        lazy = false;
        feed.toggleAttribute?.('data-resizing', false);
        if (alive()) state.reflow();
        // Whatever measures rows on a resize waits for this (project_answer.js): a row the browser skipped while the
        // width moved would otherwise be laid out again just to be measured, on every frame of the move.
        feed.dispatchEvent?.(new Event('ouro:feed-width-settled', { bubbles: true }));
    });
    const noteWidth = () => {
        const now = feed.offsetWidth;
        if (now && rowsWidth && now !== rowsWidth) {
            if (sinceMove < 0) countFrames();
            else if (!lazy) { lazy = true; feed.toggleAttribute?.('data-resizing', true); }
            sinceMove = 0;
        }
        rowsWidth = now;
    };
    const state = {
        top: 0,
        stick: true,
        // Whether the owner's own gesture moved the feed last, not the app's navigation (a waiting pill's jump to a
        // question, ↓, a quote): the phone composer folds only for the owner's own reading (composer_fold.js).
        ownMove: false,
        get generation() { return generation; },
        // The unread opening still owed (`claim(true)`): its claim's generation, else null.
        get arriving() { return arriving === generation ? arriving : null; },
        get pending() { return Boolean(intent); },
        get target() { return intent; },
        get approximate() { return approximate; },
        remember(force = false) {
            if (intent || !visible() || (!force && width !== feed.clientWidth)) return;
            width = feed.clientWidth; viewportAnchor = anchors.capture();
        },
        reflow() {
            noteWidth();
            // The animation a downward gesture still owes scrolls to is the reader's:
            // a resize neither writes under it (a write stops it short) nor ends it,
            // so its own last scroll still decides following. The resize lays out
            // once those scrolls end, no part of the gesture.
            if (dragging === 'owed') { if (!relayout) { relayout = true; owedEnd(true); } return; }
            if (dragging === 'released') dragging = ''; // a resize is no part of a released drag
            if (intent) { state.position(); return; }
            if (!visible()) return;
            if (state.stick) feed.scrollTop = feed.scrollHeight;
            else anchors.restore(viewportAnchor);
            state.top = feed.scrollTop; state.remember(true); updateButton();
        },
        scroll() {
            scrolls++;
            if (!visible()) return;
            if (!intent) {
                // A feed scrollbar drag, and the scrolls it (or a downward
                // gesture) still owes after its release, follow exactly where
                // it leaves the feed. A resize is no part of a released drag
                // (reflow), even when its clamped scroll arrives first; nor is
                // the app's own write (mutate), whose clamp leaves the feed where it put it.
                if (dragging === 'released' && feedBox() !== releasedBox) dragging = '';
                if (dragging && !(dragging === 'released' && feed.scrollTop === placed)) {
                    state.stick = feed.scrollHeight > feed.clientHeight && state.nearBottom();
                }
                state.top = feed.scrollTop;
                state.remember(relayout); // a resize waiting on these scrolls keeps where they end
            }
            updateButton();
        },
        nearBottom: (threshold = 48) => feed.scrollHeight - feed.scrollTop - feed.clientHeight <= threshold,
        mutate(write, { forceFollow = false, remoteContent = false, excludeAnchorNode = null } = {}) {
            if (typeof write !== 'function') return undefined;
            if (!alive()) return false;
            if (mutationDepth) return write();
            if (intent || !visible()) {
                const result = batch(write, afterWrite); afterWrite();
                if (remoteContent && result && !state.stick) activity();
                return result;
            }
            const follow = forceFollow || (state.stick && state.nearBottom());
            const anchor = follow ? null : anchors.capture(excludeAnchorNode);
            const height = feed.scrollHeight, top = feed.scrollTop;
            mutationDepth++;
            let result;
            // The write's deferred reads (layout_batch.js) run once its rows have settled, before the place is restored.
            try { result = batch(write, afterWrite); afterWrite(); return result; }
            finally {
                mutationDepth--;
                if (visible()) {
                    if (follow) {
                        if (forceFollow || height !== feed.scrollHeight) feed.scrollTop = feed.scrollHeight;
                        else if (feed.scrollTop !== top) feed.scrollTop = top;
                    } else anchors.restore(anchor);
                    if (remoteContent && result && !follow) activity();
                    state.top = feed.scrollTop; state.stick = follow;
                    if (dragging === 'released' && state.top !== top) placed = state.top;
                    state.remember();
                    updateButton();
                }
            }
        },
        cancel() {
            generation++; scheduled = false; intent = null; dragging = '';
            const notify = approximate;
            approximate = false; state.remember(true);
            if (notify) changed();
        },
        /** An explicit navigation (↓) that awaits history before it moves: the
         * reader's own later gesture supersedes it, even one a bounded box absorbs. */
        claim(arrive = false) {
            state.cancel();
            state.ownMove = false;
            const token = claimed = generation;
            arriving = arrive ? token : null;
            return () => token === generation;
        },
        bindGestures(navigate) {
            let touchY = null, pressed = null, latched = null, wheelAt = -Infinity, boxScrolled = null, gestures = 0;
            let touchStart = null, axis = null;  // where the touch began, and its axis once decided
            const doc = feed.ownerDocument;
            // Keys scroll the focused control's scroller or, with nothing focused,
            // the one last pressed; neither sends keydown to the feed itself.
            const keyOrigin = ({ target, key }) => feed.contains(target)
                ? (key === ' ' && target.closest?.('button, a[href], summary, [role="button"]') ? null : target)
                : (target === doc.body || target === doc.documentElement ? pressed : null);
            // A bounded box (full output, Review detail, card timeline) that can
            // still move absorbs the gesture; only at its edge does the feed move.
            const nestedMoves = (node, direction) => {
                for (let box = node; box && box !== feed; box = box.parentElement) {
                    if (!(box.scrollHeight > box.clientHeight + 1)
                        || !/auto|scroll|overlay/.test(doc.defaultView?.getComputedStyle?.(box)?.overflowY || '')) continue;
                    if (direction < 0 ? box.scrollTop > 0 : box.scrollTop + box.clientHeight < box.scrollHeight - 1) return true;
                }
                return false;
            };
            const gesture = event => {
                if (event.type === 'pointerup' || event.type === 'pointercancel') {
                    // Released, a drag decides by where it is. Engines report its last
                    // scrolls after the release, WebKit in batches that each end in a
                    // scrollend, or already before it; they decide until they end (owedEnd).
                    if (dragging) {
                        state.stick = feed.scrollHeight > feed.clientHeight && state.nearBottom();
                        dragging = 'released'; releasedBox = feedBox(); placed = null;
                    }
                    return;
                }
                if (event.type === 'pointerdown') {
                    pressed = feed.contains(event.target) ? event.target : null;
                    if (event.target === feed) { state.cancel(); state.stick = false; state.ownMove = true; }
                    dragging = event.target === feed ? 'held' : '';
                    return; // a scrollbar drag owns position, not archive traversal
                }
                if (event.type === 'touchstart') {
                    const t = event.touches?.[0];
                    touchY = t?.clientY; touchStart = { x: t?.clientX ?? 0, y: t?.clientY ?? 0 }; axis = null; latched = null;
                    return;
                }
                if (event.type === 'touchmove' && axis !== 'y') {
                    // The axis is decided once, as drawer_drag.js decides it: a sideways finger (the drawer, a swipe to
                    // reply) is not reading, nor is the drift before the axis is decided (it ended following).
                    const t = event.touches?.[0];
                    const dx = Math.abs((t?.clientX ?? touchStart?.x ?? 0) - (touchStart?.x ?? 0));
                    const dy = Math.abs((t?.clientY ?? 0) - (touchStart?.y ?? t?.clientY ?? 0));
                    if (axis === null && dy >= DRAG.slop && dy >= dx) axis = 'y';
                    else if (axis === null && dx >= DRAG.slop && dx > DRAG.axisRatio * dy) axis = 'x';
                    if (axis !== 'y') return;
                }
                const origin = event.type === 'keydown' ? keyOrigin(event) : event.target;
                if (event.type === 'keydown' && (event.defaultPrevented
                        || event.target?.closest?.('input, textarea, select, [contenteditable="true"]')
                        || !['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)
                        || !origin)) return;
                const y = event.touches?.[0]?.clientY;
                const direction = event.type === 'wheel' ? Math.sign(event.deltaY)
                    : event.type === 'keydown' ? (['ArrowUp', 'PageUp', 'Home'].includes(event.key) || (event.key === ' ' && event.shiftKey) ? -1 : 1)
                    : Math.sign((touchY ?? y) - y);
                if (event.type === 'touchmove') touchY = y;
                if (!direction) return;
                // A browser keeps one wheel burst or touch on the scroller it began on,
                // even once that box reaches its edge; so does this reading.
                const continuing = latched !== null && (event.type === 'touchmove'
                    || (event.type === 'wheel' && event.timeStamp - wheelAt < 200));
                if (event.type === 'wheel') wheelAt = event.timeStamp;
                const nested = continuing ? latched : nestedMoves(origin, direction);
                latched = event.type === 'keydown' ? null : nested;
                // The reader's own box scrolling supersedes a pending place or a ↓
                // still awaiting history, which must not override it later; other
                // layout stays. The feed pages nothing, and following can only
                // end: turning back or reading away from the live edge.
                const follow = state.stick, absorb = () => { state.stick = follow && direction > 0 && state.nearBottom(); };
                if (nested) {
                    // Box scrolling also ends an earlier feed gesture's queued frame and
                    // the scrolls it still owes, though not a scrollbar drag still held,
                    // and an unread opening still owed after its claim reached the present.
                    gestures++; if (dragging !== 'held') dragging = '';
                    if (intent || approximate || claimed === generation || arriving === generation) state.cancel();
                    absorb();
                    return;
                }
                state.cancel(); state.stick = false; state.ownMove = true;
                const token = generation, own = ++gestures, since = event.timeStamp;
                requestAnimationFrame(() => {
                    if (!alive() || !visible() || token !== generation || own !== gestures) return;
                    // A browser may move the box before it reports the gesture: when the
                    // box under it scrolled, the gesture was that box's after all.
                    for (let box = origin; boxScrolled?.at >= since - 50 && box && box !== feed; box = box.parentElement) {
                        if (box === boxScrolled.box) { latched = true; absorb(); return; }
                    }
                    state.stick = direction > 0 && feed.scrollHeight > feed.clientHeight && state.nearBottom();
                    // A key, wheel or swipe the engine animates may still be moving the feed:
                    // like a released drag, its own later scrolls decide until scrollend.
                    if (direction > 0 && !state.stick) dragging = 'owed';
                    state.top = feed.scrollTop;
                    navigate(direction);
                });
            };
            const types = ['wheel', 'touchstart', 'touchmove'];
            for (const type of types) feed.addEventListener(type, gesture, { passive: true });
            const noteBoxScroll = event => { if (event.target && event.target !== feed) boxScrolled = { box: event.target, at: event.timeStamp }; };
            feed.addEventListener('scroll', noteBoxScroll, { passive: true, capture: true });
            const settle = () => {
                if (dragging === 'released' || dragging === 'owed') owedEnd();
            };
            feed.addEventListener('scrollend', settle, { passive: true });
            const pointer = ['pointerdown', 'pointerup', 'pointercancel'];
            doc.addEventListener('keydown', gesture, { passive: true });
            for (const type of pointer) doc.addEventListener(type, gesture, { passive: true, capture: true });
            return () => {
                for (const type of types) feed.removeEventListener(type, gesture);
                feed.removeEventListener('scroll', noteBoxScroll, true);
                feed.removeEventListener('scrollend', settle);
                doc.removeEventListener('keydown', gesture);
                for (const type of pointer) doc.removeEventListener(type, gesture, true);
            };
        },
        request() {
            dragging = '';
            if (!intent) intent = { scrollTop: state.top, stick: state.stick, historyAnchor: anchors.serialize() };
            state.position();
        },
        /** An unread room opened (`claim(true)`) lands once on its newest message, `measure()`
         * px from the bottom, unless the reader moved first; `token` is that opening's
         * own claim, so a superseded one spends nothing. Nothing to move (on screen
         * there, unknown, never reached) leaves the claim's own navigation in charge. */
        land(measure, token) {
            if (arriving !== generation || token !== generation) return;
            arriving = null;
            if (!visible()) return;
            feed.scrollTop = feed.scrollHeight;
            const offset = measure();
            if (!offset) return;
            state.cancel();
            feed.scrollTop += offset;
            state.top = feed.scrollTop; state.stick = false; state.remember(true); updateButton();
        },
        followAfterLayout() {
            const token = generation;
            let passes = 0;
            const apply = () => {
                if (!alive() || token !== generation) return;
                feed.scrollTop = feed.scrollHeight; state.top = feed.scrollTop;
                state.stick = true; state.remember(); updateButton();
                if (++passes < 2) requestAnimationFrame(apply);
                else if (claimed === token) claimed = null; // the present is reached: box reading cancels no later layout
            };
            requestAnimationFrame(apply);
        },
        position() {
            if (!intent || scheduled || !visible() || !ready()) return;
            scheduled = true;
            const token = generation, target = intent;
            let passes = 0;
            const apply = () => {
                if (token !== generation) return;
                if (!visible() || !ready()) { scheduled = false; return; }
                if (target.stick) feed.scrollTop = feed.scrollHeight;
                else {
                    const exact = anchors.restore(target.historyAnchor, { exact: true });
                    approximate = !exact && Boolean(target.historyAnchor);
                    if (!exact) {
                        if (!anchors.restore(target.historyAnchor, { cardOnly: true })) {
                            feed.scrollTop = Math.max(0, Math.min(target.scrollTop || 0,
                                feed.scrollHeight - feed.clientHeight));
                        }
                    }
                }
                if (++passes < 2) { requestAnimationFrame(apply); return; }
                state.top = feed.scrollTop;
                state.stick = target.stick !== false;
                intent = null; scheduled = false;
                state.remember(true);
                changed();
            };
            requestAnimationFrame(apply);
        },
    };
    return state;
}
