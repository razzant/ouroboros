/** The chat's one reading intent. Readiness comes from data/paint owners, not
 * a frame deadline. Frames only settle geometry after those owners are ready.
 * No fetch, pager, message cache or timer lives here.
 */
export function createChatReadingPosition({ visible, alive, ready, feed, anchors, changed, afterWrite, activity, updateButton }) {
    let generation = 0, scheduled = false;
    let mutationDepth = 0;
    let viewportAnchor = null, width = feed.clientWidth;
    // Every room opens following its newest message (owner decision 2026-10-05).
    let intent = null;
    // dragging: 'held'/'released' feed scrollbar, or 'owed' to a downward gesture's animation.
    let approximate = false, dragging = '', claimed = null, scrolls = 0;
    const state = {
        top: 0,
        stick: true,
        get generation() { return generation; },
        get pending() { return Boolean(intent); },
        get target() { return intent; },
        get approximate() { return approximate; },
        remember(force = false) {
            if (intent || !visible() || (!force && width !== feed.clientWidth)) return;
            width = feed.clientWidth; viewportAnchor = anchors.capture();
        },
        reflow() {
            if (dragging === 'released' || dragging === 'owed') dragging = ''; // a resize is no part of either
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
                // it leaves the feed.
                if (dragging) state.stick = feed.scrollHeight > feed.clientHeight && state.nearBottom();
                state.top = feed.scrollTop;
                state.remember();
            }
            updateButton();
        },
        nearBottom: (threshold = 48) => feed.scrollHeight - feed.scrollTop - feed.clientHeight <= threshold,
        mutate(write, { forceFollow = false, remoteContent = false, excludeAnchorNode = null } = {}) {
            if (typeof write !== 'function') return undefined;
            if (!alive()) return false;
            if (mutationDepth) return write();
            if (intent || !visible()) {
                const result = write(); afterWrite();
                if (remoteContent && result && !state.stick) activity();
                return result;
            }
            const follow = forceFollow || (state.stick && state.nearBottom());
            const anchor = follow ? null : anchors.capture(excludeAnchorNode);
            const height = feed.scrollHeight, top = feed.scrollTop;
            mutationDepth++;
            let result;
            try { result = write(); afterWrite(); return result; }
            finally {
                mutationDepth--;
                if (visible()) {
                    if (follow) {
                        if (forceFollow || height !== feed.scrollHeight) feed.scrollTop = feed.scrollHeight;
                        else if (feed.scrollTop !== top) feed.scrollTop = top;
                    } else anchors.restore(anchor);
                    if (remoteContent && result && !follow) activity();
                    state.top = feed.scrollTop; state.stick = follow;
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
        claim() {
            state.cancel();
            const token = claimed = generation;
            return () => token === generation;
        },
        bindGestures(navigate) {
            let touchY = null, pressed = null, latched = null, wheelAt = -Infinity, boxScrolled = null, gestures = 0;
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
                    // scrolls after the release; they decide until the feed's scrollend.
                    if (dragging) { state.stick = feed.scrollHeight > feed.clientHeight && state.nearBottom(); dragging = 'released'; }
                    return;
                }
                if (event.type === 'pointerdown') {
                    pressed = feed.contains(event.target) ? event.target : null;
                    if (event.target === feed) { state.cancel(); state.stick = false; }
                    dragging = event.target === feed ? 'held' : '';
                    return; // a scrollbar drag owns position, not archive traversal
                }
                if (event.type === 'touchstart') { touchY = event.touches?.[0]?.clientY; latched = null; return; }
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
                    // the scrolls it still owes, though not a scrollbar drag still held.
                    gestures++; if (dragging !== 'held') dragging = '';
                    if (intent || approximate || claimed === generation) state.cancel();
                    absorb();
                    return;
                }
                state.cancel(); state.stick = false;
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
                if (dragging === 'released') dragging = '';
                if (dragging !== 'owed') return;
                // WebKit also ends each step of an animated page scroll: the owed
                // scrolls end only once a frame passes without another, and only
                // this gesture's: a newer one's may not have begun to arrive.
                const token = generation, seen = scrolls;
                requestAnimationFrame(() => requestAnimationFrame(() => {
                    if (dragging === 'owed' && token === generation && scrolls === seen) dragging = '';
                }));
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
