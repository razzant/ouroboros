/** Where the owner was inside what stays mounted while hidden (docs/DESIGN.md §10 "Moving between places"; owner
 * 2026-11-03: a place left and shown again keeps its tab, filters, unsaved input and position). A page, its tabs and a
 * kept room hold their own state while hidden; what a hidden box loses is its scroll: an element shown again after
 * `display: none` starts at the top (WebKit, chat.js). Every box the owner scrolled is remembered as it scrolls and put
 * back when its page, Dashboard tab or room is shown again. A room's feed keeps its own reading place
 * (chat_reading_position.js) and the Logs list its own (logs.js), so those two are left alone. No storage: the place
 * lives as long as the page.
 *
 * A repaint keeps the place too (DESIGN §10 "A repaint keeps the reading place"). A page that draws again what it
 * shows (a poll, a reconnect, a saved row) leaves its box where the owner left it, but WebKit does not: when the box
 * is or lies in a size container (`container-type`, as Settings and every Dashboard tab are) and the part drawn again
 * holds an <svg>, the box is held to what remains while that part is inserted, so it falls to the top, or part of the
 * way under whatever stands above the part (plain HTML shows it: 800 becomes 0; without the container or without the
 * <svg> it stays, as it always does in Chromium). An engine is asked once whether it does this (`dropsOnRedraw`), and
 * one that does not is left wholly alone. In one that does, after every change in the document and before the frame is
 * painted, a box the change left above its place goes back, when:
 *   - the change lies in a size container;
 *   - the box lost more scroll than what it holds got shorter (an engine that keeps the reader's row while what stands
 *     above it shrinks moves the box by exactly that, and a number put back would move the row);
 *   - the box holds within a screen of what it held: the same thing drawn again. Something else shown in the box
 *     (another folder, a filtered list) is its page's to place.
 * A page that moves a remembered box itself (`scrollTop`, `scrollTo`, `scrollIntoView`: a new circle's form at the top)
 * has moved the place with it, so a repaint in the same moment never undoes that move. */
export const OWN_PLACE = '.chat-messages, #chat-messages, #page-logs';

// Whether this engine drops a box's scroll when what it holds is drawn again: the plain case, off screen.
export function dropsOnRedraw(doc = document) {
    if (!doc.body || typeof doc.createElement !== 'function') return false;
    const probe = doc.createElement('div');
    probe.style.cssText = 'position:absolute;left:-9999px;top:0;width:200px;height:200px;display:grid;'
        + 'container-type:inline-size;visibility:hidden';
    probe.innerHTML = '<div style="overflow:auto;min-height:0"><div></div></div>';
    doc.body.append(probe);
    const box = probe.firstChild;
    const rows = `${'<div style="height:40px"></div>'.repeat(30)}<svg width="8" height="8"></svg>`;
    box.firstChild.innerHTML = rows;
    box.scrollTop = 600;
    const asked = box.scrollTop;
    box.firstChild.innerHTML = rows;
    const dropped = asked > 0 && box.scrollTop < asked;
    probe.remove();
    return dropped;
}

export function initPagePlaces({ doc = document, win = window, drops = () => dropsOnRedraw(doc) } = {}) {
    const places = new Map();  // a scrolled box → where it was left, and how much it held then
    const remember = (box) => places.set(box, [box.scrollTop, box.scrollLeft, box.scrollHeight]);
    // A page, tab or room shown again: its boxes go back where they were; a box no longer in the page is forgotten.
    const restore = (root) => {
        if (!root) return;
        for (const [box, [top, left]] of places) {
            if (!box.isConnected) { places.delete(box); continue; }
            if (!root.contains(box)) continue;
            if (box.scrollTop !== top) box.scrollTop = top;
            if (box.scrollLeft !== left) box.scrollLeft = left;
        }
    };

    // A page's own move of a remembered box is its new place at once: the box's `scrollTop` and `scrollTo` say so as
    // they move it (the element's own, over the platform's), and `scrollIntoView` says it for every box around what it
    // shows. A box out of sight takes no write and reads 0, and one whose content is not there yet cannot reach what
    // it is asked: neither is a move, so the box keeps the place it had (a hidden Dashboard tab, a closed Project's
    // panel, a page read again while away).
    const native = win.Element?.prototype && Object.getOwnPropertyDescriptor(win.Element.prototype, 'scrollTop');
    const moved = (box, asked) => {
        if (box.getClientRects?.().length !== 0 && !(native.get.call(box) < asked - 1)) remember(box);
    };
    const ownMoves = (box) => {
        if (!native?.set || Object.prototype.hasOwnProperty.call(box, 'scrollTop')) return;
        Object.defineProperty(box, 'scrollTop', { configurable: true,
            get() { return native.get.call(this); },
            set(value) { native.set.call(this, value); moved(this, value); } });
        const scrollTo = box.scrollTo;
        if (typeof scrollTo !== 'function') return;
        box.scrollTo = function scrollToHere(...args) {
            scrollTo.apply(this, args);
            moved(this, typeof args[0] === 'object' ? args[0]?.top : args[1]);
        };
    };
    // The boxes a change touched, each with what changed in it. Only a box the owner left below its top is asked, and
    // its layout is read only when the change lies in a size container: a list outside one (the Projects) costs a
    // repaint nothing. What is a container is asked anew for each change, since a layout switch can change it.
    const repainted = (records) => {
        const touched = new Map();
        for (const [box, [top]] of places) {
            if (top <= 1) continue;
            if (!box.isConnected) { places.delete(box); continue; }
            if (box.closest?.(OWN_PLACE)) continue;
            for (const record of records) {
                if (!box.contains(record.target)) continue;
                if (!touched.has(box)) touched.set(box, []);
                touched.get(box).push(record.target);
            }
        }
        if (!touched.size) return;
        const sized = new Map();  // an element → whether it is a size container
        const inContainer = (node) => {
            for (let el = node?.nodeType === 1 ? node : node?.parentElement; el; el = el.parentElement) {
                if (!sized.has(el)) sized.set(el, (win.getComputedStyle?.(el)?.containerType || 'normal') !== 'normal');
                if (sized.get(el)) return true;
            }
            return false;
        };
        for (const [box, targets] of touched) {
            if (!targets.some(inContainer)) continue;
            const [top, , held] = places.get(box);
            const holds = box.scrollHeight;
            // In place, moved on by the owner, or moved up only as far as what it holds got shorter: where it is now
            // is its place.
            if (top - box.scrollTop <= Math.max(0, held - holds) + 1) { remember(box); continue; }
            if (Math.abs(holds - held) <= box.clientHeight) box.scrollTop = Math.min(top, holds - box.clientHeight);
        }
    };
    // Asked at the first scroll, when the page is laid out for certain; an engine that keeps the place gets none of it.
    let repairs = null;
    const watchRepaints = () => {
        const into = win.Element?.prototype?.scrollIntoView;
        if (native?.set && typeof into === 'function') {
            win.Element.prototype.scrollIntoView = function scrollIntoViewHere(...args) {
                into.apply(this, args);
                for (const box of places.keys()) if (box.contains(this)) moved(box);
            };
        }
        // The whole document: pages, and the sheets and dialogs that lie over them.
        if (doc.body && win.MutationObserver) new win.MutationObserver(repainted).observe(doc.body, { childList: true, subtree: true });
    };
    const onScroll = (event) => {
        const box = event.target;
        if (box?.nodeType !== 1 || box.matches(OWN_PLACE)) return;
        if (repairs === null && (repairs = Boolean(drops()))) watchRepaints();
        if (repairs && !places.has(box)) ownMoves(box);
        remember(box);
    };
    const onPage = (event) => restore(doc.getElementById(`page-${event.detail?.page}`));
    const onTab = (event) => restore(doc.getElementById(`dashboard-panel-${event.detail?.tab}`));
    doc.addEventListener('scroll', onScroll, { capture: true, passive: true });
    win.addEventListener('ouro:page-shown', onPage);
    win.addEventListener('ouro:dashboard-subtab-shown', onTab);
    // The room owner releases its boxes at disposal, even when no later page restore occurs.
    restore.forget = (root) => {
        for (const box of places.keys()) if (!box.isConnected || root?.contains(box)) places.delete(box);
    };
    return restore;
}
