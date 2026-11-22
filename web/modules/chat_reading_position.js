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
    // Where the feed is laid out in the page (its offsetTop up the offsetParent chain; a transform does not count): a
    // Project room's bar lies outside the feed's own offsetParent, so its offsetTop alone misses the bar entering the flow.
    const feedOrigin = () => {
        let y = 0;
        for (let node = feed; node; node = node.offsetParent) y += node.offsetTop || 0;
        return y;
    };
    let viewportAnchor = null, viewportTop = null, width = feed.clientWidth, anchorOrigin = feedOrigin();
    // Every room first opens following its newest message (owner decision 2026-10-05).
    let intent = null;
    // dragging: 'held'/'released' feed scrollbar, or 'owed' to a downward gesture's animation.
    // returning: the claim of a ↓ still waiting for the present to be read (drawn busy on ↓).
    let approximate = false, dragging = '', claimed = null, scrolls = 0, arriving = null, returning = null, relayout = false;
    let owedShift = 0, owedArmed = false, owedScrolls = -1;
    // The feed's box at a drag's release: the drag's own last scrolls arrive in it, a resize's clamp in another.
    let releasedBox = '';
    // The reader's place was last chosen by a released feed scrollbar drag (until a newer gesture or navigation).
    // Rule: a place a scrollbar drag decided is not held by the feed's limit, so a taller window that clamps it and
    // a later reply move nothing (History edges: new replies stay below without moving the passage); the limit's hold
    // (heldByLimit) keeps every other place, such as the one read before the phone keyboard went away.
    let dragDecided = false;
    const feedBox = () => `${feed.clientWidth}x${feed.clientHeight}`;
    // The box the last reflow applied, or handed to a pending place. Until a changed box has had its reflow, a reader following the
    // newest message follows it still, whatever a write finds in between (mutate).
    let settledBox = feedBox();
    // Where the app's own write (mutate) moved the feed while a drag was released: a scroll there is that write's.
    let placed = null;
    // A node's top in the feed's own scroll pixels (a phone's drawer scales the room).
    const topIn = (node) => {
        const box = feed.getBoundingClientRect?.();
        return box && node?.getBoundingClientRect
            ? (node.getBoundingClientRect().top - box.top) / (box.height / (feed.offsetHeight || feed.clientHeight) || 1) : NaN;
    };
    const unappliedShift = (anchor) => {
        if (typeof anchor?.shift === 'function') return anchor.shift();
        if (!anchor?.node?.getBoundingClientRect || !feed.getBoundingClientRect || !Number.isFinite(anchor.offset)
            || anchor.node.isConnected === false) return 0;
        return topIn(anchor.node) - anchor.offset;
    };
    // WebKit also ends each step of an animated page scroll, and each batch of a
    // released drag's last scrolls: the owed scrolls end only once a frame
    // passes without another, and only this gesture's: a newer one's may not
    // have begun to arrive. A resize that waited on them lays out then; its own
    // wait looks again while they still arrive.
    const owedEnd = (wait = 0) => {
        const token = generation, seen = scrolls;
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if ((dragging === 'owed' || dragging === 'released') && token === generation && scrolls === seen
                && (!wait || wait > 4 || relayout || scrolls !== owedScrolls)) dragging = '';
            if (dragging === 'owed') { if (wait && (relayout || owedArmed)) owedEnd(token === generation ? wait + 1 : 1); return; }
            owedArmed = false;
            const shift = owedShift; owedShift = 0;
            if (relayout) { relayout = false; if (alive()) state.reflow(); }
            if (shift && !state.stick && !intent && alive() && visible()) {
                const nestedBox = viewportAnchor?.nested && viewportAnchor.node?.isConnected && feed.contains(viewportAnchor.node)
                    ? viewportAnchor.node : null;
                feed.scrollTop += shift; state.top = feed.scrollTop;
                if (nestedBox) viewportAnchor.offset = topIn(nestedBox);
                else state.remember(true);
                updateButton();
            }
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
    // Only a mounted, ordinary row still at its saved offset can be kept without a fresh visible-row search.
    // A nested/card bookmark, a missing row, or a clamp that moves it yields to the normal anchor owner.
    const rowStillPlaced = (anchor) => {
        const node = anchor?.node;
        if (!node?.isConnected || node.parentElement !== feed || !anchor.historyId || anchor.cardChain?.length) return false;
        const rect = node.getBoundingClientRect(), box = feed.getBoundingClientRect();
        const scale = box.height / feed.offsetHeight || 1;
        return rect.width > 0 && rect.height > 0 && rect.bottom > box.top && rect.top < box.bottom
            && Math.abs((rect.top - box.top) / scale - anchor.offset) <= 0.5;
    };
    // The feed's end keeps a mounted row below its saved offset (a taller room, the keyboard gone). The reader's own
    // scroll never leaves that: scrolling back leaves the end, and nothing scrolls past it. Such a place stays the place,
    // and every placement (a reflow, a write) brings the row as near it as the feed reaches, until it is reached or the
    // reader moves.
    const heldByLimit = (anchor) => {
        const node = anchor?.node;
        return Boolean(node?.isConnected) && !anchor.nested && feed.contains(node)
            && feed.scrollTop >= feed.scrollHeight - feed.clientHeight - 1 && topIn(node) - anchor.offset > 0.5;
    };
    // A bounded box grown with the viewport (an expanded card timeline, style.css) can clamp its own scroll though its top
    // stays put: the passage read in it drops by the clamp. The outer feed lifts the box by that much instead (`shift`),
    // and a box left unmoved since gets its scroll back as far as it reaches once it can (the keyboard rising again).
    const keepPassage = anchor => {
        const box = anchor.node;
        if (anchor.shift && box.scrollTop === anchor.left) box.scrollTop = anchor.inner;
        const clamp = anchor.inner - box.scrollTop;
        anchor.shift = clamp > 0.5 && box.scrollTop >= box.scrollHeight - box.clientHeight - 1 ? clamp : 0;
        if (anchor.shift) feed.scrollTop += anchor.shift;
        anchor.left = box.scrollTop;
    };
    // The reader's own movement in that box; a clamp (below the last place, at the new limit) is geometry's.
    const noteInner = (anchor, box) => {
        const top = box.scrollTop;
        if (top < anchor.inner && top >= box.scrollHeight - box.clientHeight - 1) return;
        anchor.offset -= anchor.shift; anchor.shift = 0; anchor.inner = top;
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
        get returning() { return returning === generation; },
        get pending() { return Boolean(intent); },
        get target() { return intent; },
        get approximate() { return approximate; },
        remember(force = false, box = null) {
            viewportTop = null;
            if (intent || !visible() || (!force && width !== feed.clientWidth)) return;
            const boxTop = box ? topIn(box) : 0;
            if (box && !Number.isFinite(boxTop)) return;
            width = feed.clientWidth;
            anchorOrigin = feedOrigin();
            // `offset`: the box's place while it is scrolled to `inner`, where the reader last moved it; `shift`: the lift
            // standing in for its clamp (keepPassage).
            viewportAnchor = box ? { node: box, offset: boxTop, nested: true, inner: box.scrollTop, shift: 0 } : anchors.capture();
        },
        reflow(writeGeometry) {
            const previousRowsWidth = rowsWidth;
            // The resize already changed layout. Apply reserves and any resulting composer remeasure before the one
            // restoration, not as a second capture/rebase over the same feed. This callback writes geometry only.
            if (writeGeometry) {
                mutationDepth++;
                try { writeGeometry(); } finally { mutationDepth--; }
            }
            noteWidth();
            // The animation a downward gesture still owes scrolls to is the reader's:
            // a resize neither writes under it (a write stops it short) nor ends it,
            // so its own last scroll still decides following. The resize lays out
            // once those scrolls end, no part of the gesture.
            if (dragging === 'owed') { if (!relayout) { relayout = true; owedEnd(1); } return; }
            if (dragging === 'released') dragging = ''; // a resize is no part of a released drag
            if (intent) { if (visible()) settledBox = feedBox(); state.position(); return; }
            if (!visible()) return;
            const nextOrigin = feedOrigin();
            if (nextOrigin !== anchorOrigin && viewportAnchor) {
                const shift = nextOrigin - anchorOrigin;
                viewportAnchor.offset -= shift;
                if (Number.isFinite(viewportAnchor.topOffset)) viewportAnchor.topOffset -= shift;
                for (const entry of viewportAnchor.cardChain || []) entry.offset -= shift;
                for (const entry of viewportAnchor.fallbacks || []) entry.offset -= shift;
            }
            anchorOrigin = nextOrigin;
            const heightOnly = writeGeometry && previousRowsWidth === rowsWidth && width === feed.clientWidth && !lazy && !dragging;
            const kept = heightOnly && !state.stick && rowStillPlaced(viewportAnchor);
            let keptNested = false;
            if (state.stick) {
                if (!heightOnly || !state.nearBottom(0)) feed.scrollTop = feed.scrollHeight;
            } else if (!kept) keptNested = anchors.restore(viewportAnchor) && viewportAnchor?.nested === true;
            if (keptNested) keepPassage(viewportAnchor);
            settledBox = feedBox(); state.top = feed.scrollTop;
            const keptHeld = !dragDecided && !state.stick && !kept && heldByLimit(viewportAnchor);
            // A later reflow still belongs to the box edge, not a line the reader can move inside it.
            // A box that is gone or hidden fails restore and is reselected by the ordinary anchor owner. A row put back
            // still stays the row read through a passing layout (a rotation): a fresh search there would pick another
            // row of the band, and the way back would land a line off.
            if (keptNested || keptHeld || (heightOnly && state.stick) || (!state.stick && rowStillPlaced(viewportAnchor))) {
                width = feed.clientWidth;
                if (state.stick) viewportAnchor = null; // an edge follower has no reading row to reselect
                viewportTop = state.top;
            } else { state.remember(true); viewportTop = state.top; }
            updateButton();
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
                const top = feed.scrollTop;
                // A resize clamp may notify before its reflow; it doesn't replace the box the reader chose.
                const nestedGeometry = !dragging && viewportAnchor?.nested && (top === state.top || settledBox !== feedBox());
                state.top = top;
                if (!nestedGeometry && (dragging || viewportTop !== top || settledBox !== feedBox())
                    && !heldByLimit(viewportAnchor)) state.remember(relayout);
                else viewportTop = null;
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
            const follow = forceFollow || (state.stick && (state.nearBottom() || feedBox() !== settledBox));
            const anchor = follow ? null : !dragging && !excludeAnchorNode && heldByLimit(viewportAnchor)
                ? viewportAnchor : anchors.capture(excludeAnchorNode);
            const height = feed.scrollHeight, top = feed.scrollTop;
            const box = !follow && viewportAnchor?.nested ? viewportAnchor.node : null, stood = box ? topIn(box) : 0;
            mutationDepth++;
            let result;
            // The write's deferred reads (layout_batch.js) run once its rows have settled, before the place is restored.
            try { result = batch(write, afterWrite); afterWrite(); return result; }
            finally {
                mutationDepth--;
                if (visible()) {
                    if (follow) {
                        owedShift = 0;
                        if (forceFollow || height !== feed.scrollHeight) feed.scrollTop = feed.scrollHeight;
                        else if (feed.scrollTop !== top) feed.scrollTop = top;
                    } else if (dragging === 'owed') {
                        const shift = unappliedShift(anchor);
                        if (Math.abs(shift) > 0.5) {
                            owedShift += shift;
                            if (!owedArmed) { owedArmed = true; owedEnd(1); }
                        }
                    } else anchors.restore(anchor);
                    if (remoteContent && result && !follow) activity();
                    state.top = feed.scrollTop; state.stick = follow;
                    if (dragging === 'released' && state.top !== top) placed = state.top;
                    // A write (a state poll, a live frame) captures whatever line is passing. A box the reader scrolls
                    // stays the place, moved by as far as the write moved the box itself (not by a lift the feed's limit
                    // may not have applied); so does a row the limit still holds back.
                    const boxKept = !dragging && box?.isConnected && feed.contains(box);
                    if (boxKept) viewportAnchor.offset += topIn(box) - stood;
                    if (boxKept || (!follow && !dragging && heldByLimit(viewportAnchor))) viewportTop = null;
                    else state.remember(relayout);
                    updateButton();
                }
            }
        },
        cancel(keepOwed = false) {
            const busy = state.returning;
            generation++; scheduled = false; intent = null; dragging = ''; dragDecided = false;
            if (!keepOwed) { owedShift = 0; owedArmed = false; }
            const notify = approximate;
            approximate = false; state.remember(true);
            if (notify) changed(); else if (busy) updateButton();
        },
        /** One navigation (↓), including its later history load: the reader's
         * own gesture supersedes it, even one a bounded box absorbs. */
        claim(arrive = false) {
            state.cancel();
            state.ownMove = false;
            const token = claimed = generation;
            arriving = arrive ? token : null;
            return () => token === generation;
        },
        /** ↓ (`current`: its claim) waits while the present is read: drawn busy on ↓ until `wait(current, false)`
         * once that read has landed or failed, or until the reader's own gesture or another navigation supersedes
         * the claim. A superseded claim's end changes nothing of a newer one. */
        wait(current, waiting = true) {
            if (!alive() || !current()) return;
            returning = waiting ? generation : null;
            updateButton();
        },
        bindGestures(navigate) {
            // `held`: pressed element inside the feed (such as a nested box whose own scrollbar is dragged), distinct from
            // `dragging === 'held'` (the outer feed scrollbar).
            let touchY = null, pressed = null, latched = null, wheelAt = -Infinity, boxScrolled = null, gestures = 0, held = null;
            let touchStart = null, axis = null;  // where the touch began, and its axis once decided
            const doc = feed.ownerDocument;
            // Keys scroll the focused control's scroller or, with nothing focused,
            // the one last pressed; neither sends keydown to the feed itself.
            const keyOrigin = ({ target, key }) => feed.contains(target)
                ? (key === ' ' && target.closest?.('button, a[href], summary, [role="button"]') ? null : target)
                : (target === doc.body || target === doc.documentElement ? pressed : null);
            // A bounded box (full output, Review detail, card timeline) that can
            // still move absorbs the gesture; only at its edge does the feed move.
            const nestedScroller = (node, direction) => {
                for (let box = node; box && box !== feed; box = box.parentElement) {
                    if (!(box.scrollHeight > box.clientHeight + 1)
                        || !/auto|scroll|overlay/.test(doc.defaultView?.getComputedStyle?.(box)?.overflowY || '')) continue;
                    if (direction < 0 ? box.scrollTop > 0 : box.scrollTop + box.clientHeight < box.scrollHeight - 1) return box;
                }
                return false;
            };
            const gesture = event => {
                if (event.type === 'pointerup' || event.type === 'pointercancel') {
                    // Released, a feed scrollbar drag decides by where it is; an animated gesture's `'owed'` state
                    // outlives touch `pointercancel`/`pointerup` until its scrolls rest (`owedEnd`).
                    held = null;
                    if (dragging === 'held') {
                        state.stick = feed.scrollHeight > feed.clientHeight && state.nearBottom();
                        dragging = 'released'; releasedBox = feedBox(); placed = null; dragDecided = true;
                    }
                    return;
                }
                if (event.type === 'pointerdown') {
                    pressed = feed.contains(event.target) ? event.target : null;
                    held = pressed !== feed ? pressed : null;
                    if (event.target === feed) { state.cancel(); state.stick = false; state.ownMove = true; dragging = 'held'; }
                    else if (dragging !== 'owed') dragging = '';
                    return; // a scrollbar drag owns position, not archive traversal
                }
                // A second finger makes the touch a pinch, which only magnifies: no reading, paging or keyboard
                // dismissal, until a new one-finger touch begins.
                if (event.type === 'touchstart' || event.type === 'touchmove') {
                    if (event.touches?.length > 1) axis = 'pinch';
                    if (axis === 'pinch' && (event.type === 'touchmove' || event.touches?.length > 1)) return;
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
                        || !origin || !visible())) return;  // a key pressed elsewhere is no reading of a room left
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
                const nested = continuing ? latched : nestedScroller(origin, direction);
                latched = event.type === 'keydown' ? null : nested;
                // The reader's own box scrolling supersedes a pending place or a ↓
                // still awaiting history, which must not override it later; other
                // layout stays. The feed pages nothing, and following can only
                // end: turning back or reading away from the live edge.
                // Cancelling a pending ↓ also ends its follow, including a shared read's later row updates.
                const follow = state.stick && !state.returning;
                const absorb = box => {
                    state.stick = follow && direction > 0 && state.nearBottom();
                    // The outer feed may never send a scroll. Hand off to this scroller's edge once,
                    // even when already away from the live edge; its inner lines can still move.
                    if (!state.stick && !(viewportAnchor?.nested && viewportAnchor.node === box)
                            && box?.isConnected && feed.contains(box)) state.remember(true, box);
                };
                if (nested) {
                    // Box scrolling also ends an earlier feed gesture's queued frame and
                    // the scrolls it still owes, though not a scrollbar drag still held,
                    // and an unread opening still owed after its claim reached the present.
                    gestures++; if (dragging !== 'held') dragging = '';
                    if (intent || approximate || state.returning || claimed === generation || arriving === generation) state.cancel(true);
                    absorb(nested);
                    return;
                }
                state.cancel(true); state.stick = false; state.ownMove = true;
                if (direction > 0 && feed.scrollHeight > feed.clientHeight && !state.nearBottom()) {
                    dragging = 'owed'; owedScrolls = scrolls;
                }
                const token = generation, own = ++gestures, since = event.timeStamp;
                requestAnimationFrame(() => {
                    if (token !== generation || own !== gestures) return;
                    if (!alive() || !visible()) { if (dragging === 'owed') { dragging = ''; owedShift = 0; owedArmed = false; } return; }
                    // A browser may move the box before it reports the gesture: when the
                    // box under it scrolled, the gesture was that box's after all.
                    for (let box = origin; boxScrolled?.at >= since - 50 && box && box !== feed; box = box.parentElement) {
                        if (box === boxScrolled.box) {
                            if (dragging === 'owed') dragging = '';
                            latched = box; absorb(box); return;
                        }
                    }
                    state.stick = direction > 0 && feed.scrollHeight > feed.clientHeight && state.nearBottom();
                    // A key, wheel or swipe the engine animates may still be moving the feed:
                    // like a released drag, its own later scrolls decide until scrollend.
                    if (direction > 0 && !state.stick) {
                        if (dragging !== 'owed') { dragging = 'owed'; owedScrolls = scrolls; }
                    } else if (dragging === 'owed') dragging = '';
                    state.top = feed.scrollTop;
                    navigate(direction);
                });
            };
            const types = ['wheel', 'touchstart', 'touchmove'];
            for (const type of types) feed.addEventListener(type, gesture, { passive: true });
            const noteBoxScroll = ({ target: box, timeStamp }) => {
                if (!box || box === feed) return;
                boxScrolled = { box, at: timeStamp };
                if (viewportAnchor?.nested && viewportAnchor.node === box) noteInner(viewportAnchor, box);
                // A box's own scrollbar dragged, or a box scrolling around a captured inner line: the box is the place.
                else if (!state.stick && !intent && box.isConnected && feed.contains(box)
                    && (box === held || (!mutationDepth && settledBox === feedBox()
                        && box.contains?.(viewportAnchor?.node) && viewportAnchor?.node !== box))) {
                    state.remember(true, box);
                }
            };
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
            dragging = ''; owedShift = 0; owedArmed = false;
            if (!intent) intent = { scrollTop: state.top, stick: state.stick, historyAnchor: anchors.serialize() };
            state.position();
        },
        capture() {
            const saved = intent || { scrollTop: feed.scrollTop, stick: state.stick, historyAnchor: anchors.serialize() };
            return () => { state.cancel(); intent = saved; state.position(); };
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
        /** Follows the newest message across the next layouts. `yielding`: a follow nobody asked for (an opening's saved
         * preview) gives way to a navigation that claimed this generation and left the newest message (a Reply). */
        followAfterLayout(yielding = false) {
            const token = generation;
            if (yielding && claimed === token && !state.stick) return;
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
