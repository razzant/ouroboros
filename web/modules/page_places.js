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
 * <svg> it stays, as it always does in Chromium). An engine is asked once whether it does this (`dropsOnRedraw`, after
 * the first scroll), and one that does not is left wholly alone. In one that does, after every change in the document
 * and before the frame is painted, a box the change left above its place goes back, when:
 *   - the change lies in a size container;
 *   - the box lost more scroll than what it holds got shorter (an engine that keeps the reader's row while what stands
 *     above it shrinks moves the box by exactly that, and a number put back would move the row);
 *   - the box holds within a screen of what it held: the same thing drawn again. Something else shown in the box
 *     (another folder, a filtered list) is its page's to place.
 * A page that moves a remembered box itself (`scrollTop`, `scrollTo`, `scroll`, `scrollBy`, `scrollIntoView`: a new
 * circle's form at the top) has moved the place with it, so a repaint in the same moment never undoes that move. */
import { revealPastFade } from './scroll_fade.js';

export const OWN_PLACE = '.chat-messages, #chat-messages, #page-logs';
// The row of a section's list that names the page open beside it (Settings' rail, the Dashboard's list): the chosen
// row, while its shell draws the page's column too. The sheets decide that (the Dashboard's automatic view), and a list
// that stands alone names nothing.
const NAMING_ROW = '.section-shell > .section-rail :is([aria-current="page"], [aria-orientation="vertical"] [aria-selected="true"])';
const drawn = (node) => node?.getClientRects?.().length > 0;

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

export function initPagePlaces({ doc = document, win = window, drops = () => dropsOnRedraw(doc),
    later = (run) => (win.setTimeout || setTimeout)(run, 0) } = {}) {
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

    // A page's own move of a remembered box is its new place at once: the box's `scrollTop` and its scroll methods say
    // so as they move it (the element's own, over the platform's), and `scrollIntoView` says it for every box around
    // what it shows. A box out of sight takes no write and reads 0, and one whose content is not there yet cannot
    // reach what it is asked: neither is a move, so the box keeps the place it had (a hidden Dashboard tab, a closed
    // Project's panel, a page read again while away).
    const native = win.Element?.prototype && Object.getOwnPropertyDescriptor(win.Element.prototype, 'scrollTop');
    const moved = (box, asked) => {
        if (box.getClientRects?.().length !== 0 && !(native.get.call(box) < asked - 1)) remember(box);
    };
    const ownMoves = (box) => {
        if (!native?.set || Object.prototype.hasOwnProperty.call(box, 'scrollTop')) return;
        Object.defineProperty(box, 'scrollTop', { configurable: true,
            get() { return native.get.call(this); },
            set(value) { native.set.call(this, value); moved(this, value); } });
        for (const name of ['scrollTo', 'scroll', 'scrollBy']) {
            const platform = box[name];
            if (typeof platform !== 'function') continue;
            box[name] = function ownMove(...args) {
                platform.apply(this, args);
                moved(this, name === 'scrollBy' ? undefined : typeof args[0] === 'object' ? args[0]?.top : args[1]);
            };
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
            // A box out of sight reads 0 for all of it: a page drawing again while away tells nothing of its place.
            if (!targets.some(inContainer) || box.getClientRects?.().length === 0) continue;
            const [top, , held] = places.get(box);
            const holds = box.scrollHeight;
            // In place, moved on by the owner, or moved up only as far as what it holds got shorter: where it is now
            // is its place.
            if (top - box.scrollTop <= Math.max(0, held - holds) + 1) { remember(box); continue; }
            if (Math.abs(holds - held) <= box.clientHeight) box.scrollTop = Math.min(top, holds - box.clientHeight);
        }
    };
    // Asked once, after the first scroll, when the page is laid out for certain; an engine that keeps the place gets
    // none of it. Not inside the scroll event: a document changed there loses the engine's own keeping of the reader's
    // row for the box that scrolled.
    let asked = false;
    let repairs = false;
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
        for (const box of places.keys()) ownMoves(box);
    };
    const ask = () => { if ((repairs = Boolean(drops()))) watchRepaints(); };
    const onScroll = (event) => {
        const box = event.target;
        if (box?.nodeType !== 1 || box.matches(OWN_PLACE)) return;
        if (!asked) { asked = true; later(ask); }
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
    // A forward jump starts the visible destination anew; its captured return still owns the old reading place.
    restore.land = (root) => {
        for (const box of [root, ...(root?.querySelectorAll('*') || [])]) {
            if (!box || box.matches(OWN_PLACE) || !box.getClientRects().length) continue;
            if (!box.scrollTop && !box.scrollLeft) continue;
            box.scrollTop = box.scrollLeft = 0;
            remember(box);
        }
    };
    // A return owns a snapshot of the retained place, not a second navigation stack. The room's feed and Logs
    // still restore through their reading owners. Keep nearby anchors too: a row may disappear while away. A box left
    // within its fade's depth returns to its top, and a list beside its open page shows the row naming that page.
    restore.capture = (root, opener) => {
        const boxes = [root, ...(root?.querySelectorAll('*') || [])].filter((box) => box && !box.matches(OWN_PLACE)
            && (box.scrollHeight > box.clientHeight || box.scrollWidth > box.clientWidth));
        const saved = boxes.map((box) => [box, box.scrollTop, box.scrollLeft]);
        const anchors = [];
        const popup = opener?.closest?.('[role="dialog"], [role="menu"], .ui-popup');
        if (popup?.id) anchors.push(...Array.from(doc.querySelectorAll('[aria-controls]'))
            .filter((node) => node.getAttribute('aria-controls').split(' ').includes(popup.id)));
        if (opener) anchors.unshift(opener);
        for (let at = root?.contains(opener) ? opener : null; at && at !== root; at = at.parentElement) {
            anchors.push(at, at.nextElementSibling, at.previousElementSibling);
        }
        // Within its fade's depth the top is the place: a few pixels, left by the owner or by the returned row's own
        // reveal past the lower fade, would only dim the first row under the upper one.
        const settle = () => {
            for (const [box] of saved) {
                if (!box.isConnected || !(box.scrollTop > 0) || !box.matches('.scroll-fade-y')) continue;
                const fade = parseFloat(win.getComputedStyle?.(box)?.getPropertyValue?.('--scroll-fade-edge')) || 0;
                if (box.scrollTop < fade) { box.scrollTop = 0; remember(box); }
            }
        };
        return ({ keyboard = false } = {}) => {
            for (const [box, top, left] of saved) {
                if (!box.isConnected) continue;
                box.scrollTop = top;
                box.scrollLeft = left;
                remember(box);
            }
            const anchor = anchors.find((el) => el?.isConnected && el.getClientRects().length && !el.closest('[hidden], [inert]')) || root;
            // A list brings the row that names the page beside it in, past its fade, as the list's own reveal does when
            // the row is chosen, so a return does not leave the open page unnamed. The pressed row keeps its view: when
            // both cannot share it, the list stays where it was left.
            const scrolls = (box) => box.scrollHeight > box.clientHeight && /auto|scroll/.test(win.getComputedStyle?.(box)?.overflowY);
            for (const row of root?.querySelectorAll?.(NAMING_ROW) || []) {
                if (!drawn(row) || !drawn(row.closest?.('.section-shell')?.querySelector?.(':scope > .section-pane'))) continue;
                let box = row.parentElement;
                while (box && !scrolls(box)) box = box === root ? null : box.parentElement;
                if (!box) continue;
                const left = box.scrollTop;
                revealPastFade(box, row);
                const shown = box.scrollTop;
                if (anchor !== root && box.contains?.(anchor)) {
                    revealPastFade(box, anchor);
                    if (Math.abs(box.scrollTop - shown) > 1) box.scrollTop = left;  // a row's sub-pixel height is no move
                }
                remember(box);
            }
            if (!anchor || !anchor.isConnected) { settle(); return; }
            if (!anchor.matches('button, a, input, select, textarea, [tabindex]')) anchor.tabIndex = -1;
            anchor.setAttribute?.('data-return-focus', keyboard ? 'keyboard' : 'pointer');
            const clear = () => {
                anchor.removeAttribute?.('data-return-focus');
                anchor.removeEventListener?.('blur', clear);
                anchor.removeEventListener?.('keydown', clear);
            };
            anchor.addEventListener?.('blur', clear);
            anchor.addEventListener?.('keydown', clear);
            anchor.focus?.({ preventScroll: true });
            settle();
        };
    };
    return restore;
}
