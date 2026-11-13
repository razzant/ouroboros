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
 * holds an <svg>, the box's scroll falls to the top as the part is inserted (ten lines of plain HTML show it: 800
 * becomes 0; without the container or without the <svg> it stays, as it always does in Chromium). So after every
 * change in the document, before the frame is painted, a box found at its top that the owner left lower goes back
 * there, when the change lies in a size container and what the box holds still reaches that far. A page that moves a
 * remembered box itself (`scrollTop`, `scrollTo`: a new circle's form at the top) has moved the place with it, so a
 * repaint in the same moment never undoes that move. */
export const OWN_PLACE = '.chat-messages, #chat-messages, #page-logs';

export function initPagePlaces({ doc = document, win = window } = {}) {
    const places = new Map();  // a scrolled box → where it was left
    const remember = (box) => places.set(box, [box.scrollTop, box.scrollLeft]);
    // A page's own move of a remembered box is its new place at once: the box's `scrollTop` and `scrollTo` say so as
    // they move it (the element's own, over the platform's; nothing else changes).
    const native = win.Element?.prototype && Object.getOwnPropertyDescriptor(win.Element.prototype, 'scrollTop');
    const ownMoves = (box) => {
        if (!native?.set || Object.prototype.hasOwnProperty.call(box, 'scrollTop')) return;
        Object.defineProperty(box, 'scrollTop', { configurable: true,
            get() { return native.get.call(this); },
            set(value) { native.set.call(this, value); remember(this); } });
        const scrollTo = box.scrollTo;
        if (typeof scrollTo === 'function') box.scrollTo = function scrollToHere(...args) { scrollTo.apply(this, args); remember(this); };
    };
    const onScroll = (event) => {
        const box = event.target;
        if (box?.nodeType !== 1 || box.matches(OWN_PLACE)) return;
        if (!places.has(box)) ownMoves(box);
        remember(box);
    };
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
            const [top] = places.get(box);
            if (box.scrollTop > 1 || box.scrollHeight - box.clientHeight < top - 1) continue;
            box.scrollTop = top;
        }
    };
    // The whole document: pages, and the sheets and dialogs that lie over them.
    const Observer = win.MutationObserver;
    const watch = doc.body && Observer ? new Observer(repainted) : null;
    watch?.observe(doc.body, { childList: true, subtree: true });
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
