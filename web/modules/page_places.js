/** Where the owner was inside what stays mounted while hidden (docs/DESIGN.md §10 "Moving between places"; owner
 * 2026-11-03: a place left and shown again keeps its tab, filters, unsaved input and position). A page, its tabs and a
 * kept room hold their own state while hidden; what a hidden box loses is its scroll: an element shown again after
 * `display: none` starts at the top (WebKit, chat.js). Every box the owner scrolled is remembered as it scrolls and put
 * back when its page, Dashboard tab or room is shown again. A room's feed keeps its own reading place
 * (chat_reading_position.js) and the Logs list its own (logs.js), so those two are left alone. No storage: the place
 * lives as long as the page. */
const OWN_PLACE = '.chat-messages, #chat-messages, #page-logs';

export function initPagePlaces({ doc = document, win = window } = {}) {
    const places = new Map();  // a scrolled box → where it was left
    const onScroll = (event) => {
        const box = event.target;
        if (box?.nodeType === 1 && !box.matches(OWN_PLACE)) places.set(box, [box.scrollTop, box.scrollLeft]);
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
