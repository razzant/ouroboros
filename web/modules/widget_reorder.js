/* Widgets card arrangement by the owner: the `widget_order` preference applied
   to the card list, the pure key-order move behind a reorder and its merge
   into the stored order (a card not on screen keeps its slot), the move handle
   (`bindWidgetCardReorder`: a pointer or a finger carries a card, its arrow
   keys and the card menu step it), and the card sizes (`createWidgetWidths`:
   the card menu, the right, bottom and corner edges and their keys over
   `ui_preferences.widget_size`, and Show all).
   Nothing here moves an <article>: the masonry (web/modules/masonry.js)
   places the cards by custom properties, so a running frame — retained or
   not — is never reloaded by a reorder or a resize. widgets.js owns persisting
   the order through `/api/ui/preferences` and fitting each card to its height.
   Disclosed residual: the Tab / focus order follows the DOM, so after a visual
   reorder it can differ from the visible order until a window reload rebuilds
   the cards; keyboard reorder through the handle follows the key order. */

import { haptic } from './haptics.js';
import { isCoarsePointer, onLayoutChange } from './layout.js';
import { widgetKey } from './widget_list.js';
import { applyMasonry } from './masonry.js';
import {
    draggedWidgetHeight, heightName, nearestWidthChoice, nextWidgetSize, normalizeWidgetSize, ownerSpans,
    stepWidgetWidth, WIDGET_FULL_SPAN, WIDGET_HEIGHT_KEY_STEP, WIDGET_HEIGHT_MIN, WIDGET_WIDTH_STEPS, widgetHeight,
    widgetWidth, widthChoices,
} from './widget_size.js';

export function normalizeWidgetOrder(value) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    return value
        .map((item) => String(item || '').trim())
        .filter((item) => {
            if (!item || seen.has(item)) return false;
            seen.add(item);
            return true;
        });
}

/**
 * The cards in the owner's order. A card the order does not hold yet keeps the
 * place this window last showed it in (`shown`), and a card new to both joins
 * the end, so a widget that appears never lands at its listing place (the
 * server lists by key) ahead of the cards already on screen.
 */
export function sortTabsByWidgetOrder(tabs, order, shown = []) {
    const known = normalizeWidgetOrder([...normalizeWidgetOrder(order), ...normalizeWidgetOrder(shown)]);
    const rank = new Map(known.map((key, idx) => [key, idx]));
    return tabs.map((tab, originalIndex) => ({ tab, originalIndex })).sort((a, b) => {
        const aRank = rank.has(widgetKey(a.tab)) ? rank.get(widgetKey(a.tab)) : Number.MAX_SAFE_INTEGER;
        const bRank = rank.has(widgetKey(b.tab)) ? rank.get(widgetKey(b.tab)) : Number.MAX_SAFE_INTEGER;
        if (aRank !== bRank) return aRank - bRank;
        return a.originalIndex - b.originalIndex;
    }).map((item) => item.tab);
}

/**
 * The stored order after the owner rearranged the shown cards into `shown`:
 * every slot that holds a shown key takes the next key of `shown`, so a key
 * not on screen (its skill is off) keeps its slot for its return; shown keys
 * the stored order lacks take new slots at its end. Stored `[A, H, B]` with
 * `[B, A]` shown becomes `[B, H, A]`. "Not on screen" is the only signal.
 */
export function mergeWidgetOrder(stored, shown) {
    const visible = normalizeWidgetOrder(shown);
    const onScreen = new Set(visible);
    const kept = normalizeWidgetOrder(stored);
    const keptKeys = new Set(kept);
    let next = 0;
    return [...kept, ...visible.filter((key) => !keptKeys.has(key))]
        .map((key) => (onScreen.has(key) ? visible[next++] : key));
}

/**
 * Pure key-order move: `key` leaves its slot and re-enters at `toIndex`
 * (clamped to the list). A drop onto another card passes that card's index,
 * which lands the dragged key after a target it was before and before a target
 * it was after. Returns the SAME array when nothing changes, so callers test
 * identity for "moved".
 */
export function moveWidgetKey(order, key, toIndex) {
    const from = order.indexOf(key);
    if (from < 0 || !order.length) return order;
    const target = Math.max(0, Math.min(order.length - 1, Math.trunc(Number(toIndex) || 0)));
    if (target === from) return order;
    const next = order.slice();
    next.splice(from, 1);
    next.splice(target, 0, key);
    return next;
}


/** Where the card menu's move `where` (start, earlier, later, end) sends `key` in `order`; -1 when it cannot go there. */
export function widgetMoveIndex(order, key, where) {
    const from = order.indexOf(key);
    const to = { start: 0, earlier: from - 1, later: from + 1, end: order.length - 1 }[where];
    return from < 0 || to === undefined || to < 0 || to >= order.length || to === from ? -1 : to;
}

const titleOf = (card) => card?.querySelector?.('.widgets-card-title strong')?.textContent?.trim() || 'Widget';

/** What the live region says of the card `key` (its node `card`) once moved to the key order `order`, by any path. */
export function widgetPlaceWords(key, card, order) {
    return `${titleOf(card)}: place ${order.indexOf(key) + 1} of ${order.length}`;
}

// A press on a card's head (mouse or pen) or its move handle becomes a carry after this much travel; a press on a
// bottom edge previews a height after this much vertical travel.
const MOVE_THRESHOLD_PX = 5;
// A carried card takes a new place only when that place is this much nearer than the one it holds.
const MOVE_HYSTERESIS_PX = 24;
// Near the top or bottom edge of the board's scroller the board scrolls under a carried card.
const SCROLL_ZONE_PX = 64;
const SCROLL_STEP_PX = 24;
// The drop's glide into place (style.css's transition, and a little more).
const GLIDE_MS = 260;
const boundLists = new WeakMap();

/**
 * Moving cards: `currentOrder()` returns the complete visible key order; a move
 * hands the next order to `onOrderChange` and touches no node. A pointer or a
 * finger on the move handle, or a mouse or pen on the card's head, carries the
 * card: it follows the pointer, the board shows where it would land (the cards
 * after it make room, an outline marks the place) and scrolls near its edges;
 * release puts it there, Escape back. The handle's arrow keys (Home, End) step it.
 * `board` is the list's sizes controller (`createWidgetWidths`): its masonry plans
 * the places, and its live region names the move. Bound once per list; a later
 * call replaces the three inputs.
 */
export function bindWidgetCardReorder(list, currentOrder, onOrderChange, board = null) {
    if (!list) return;
    const known = boundLists.get(list);
    if (known) {
        Object.assign(known, { currentOrder, onOrderChange, board });
        return;
    }
    const inputs = { currentOrder, onOrderChange, board };
    boundLists.set(list, inputs);
    const doc = list.ownerDocument || globalThis.document;
    let pending = null;
    let carry = null;
    let scrollFrame = 0;
    let glideTimer = 0;
    const announce = (text) => inputs.board?.announce?.(text);
    const placeWords = (card, order) => widgetPlaceWords(card.dataset.widgetKey, card, order);

    list.addEventListener('keydown', (event) => {
        const handle = event.target.closest?.('[data-widget-reorder-handle]');
        const card = handle?.closest('[data-widget-key]');
        if (!card || carry || event.altKey || event.ctrlKey || event.metaKey) return;
        const key = card.dataset.widgetKey || '';
        const order = inputs.currentOrder();
        const from = order.indexOf(key);
        const toIndex = { ArrowUp: from - 1, ArrowLeft: from - 1, ArrowDown: from + 1, ArrowRight: from + 1,
            Home: 0, End: Number.MAX_SAFE_INTEGER }[event.key];
        if (from < 0 || toIndex === undefined) return;
        const next = moveWidgetKey(order, key, toIndex);
        if (next === order) return;
        event.preventDefault();
        inputs.onOrderChange(next);
        handle.focus();
        announce(placeWords(card, next));
    });

    // A press holds its pointer from the moment it lands, and the board's frames ignore the pointer meanwhile
    // (`data-widget-pressed`, style.css), so its moves and its release over the card's own frame, or past the window's
    // edge, reach the board. An engine that routes the pointer by what it last painted can still hand a frame what
    // arrives before the next paint; a release lost that way ends the press at the next move. Whichever way the press
    // ends (a release, a cancel, a move with no button held, Escape, leaving the page, the card leaving the board, seen
    // at the next move), the listening, the pointer and the frames' pause go with it; a carried card goes back unsaved.
    const letGo = (el, pointerId) => {
        try { if (el?.hasPointerCapture?.(pointerId)) el.releasePointerCapture(pointerId); } catch { /* gone */ }
    };
    const stopListening = () => {
        list.removeAttribute('data-widget-pressed');
        doc.removeEventListener('pointermove', onPointerMove, true);
        doc.removeEventListener('pointerup', onPointerUp, true);
        doc.removeEventListener('pointercancel', onPointerUp, true);
        doc.removeEventListener('keydown', onEscape, true);
    };
    // A press that ends before it carried anything.
    const dropPress = () => {
        const press = pending;
        pending = null;
        stopListening();
        if (press) letGo(press.el, press.pointerId);
    };
    const listPoint = (event) => {
        const box = list.getBoundingClientRect();
        return { x: event.clientX - box.left, y: event.clientY - box.top };
    };
    // The card follows the pointer; the nearest place the masonry can give it, among every place in the order,
    // becomes its place when it is clearly nearer than the one it holds.
    // Another window may take a card off this board, add one or reorder them while a card is carried here: the carry
    // then goes on over the cards shown now, in their order now (nothing is saved yet), and a drop that comes after
    // such a change, before the pointer moves again, saves nothing.
    const sameOrder = (order) => {
        const live = inputs.currentOrder();
        return live.length === order.length && live.every((key, index) => key === order[index]);
    };
    const follow = (event) => {
        carry.last = event;
        if (!sameOrder(carry.origin)) {
            carry.origin = inputs.currentOrder().slice();
            carry.order = carry.origin;
            carry.index = carry.origin.indexOf(carry.key);
            if (carry.index < 0) {
                end(false);
                return;
            }
            inputs.board.carry(carry.key, carry.order);
        }
        const point = listPoint(event);
        // The lifted card stays over the board: never past its left or right edge, nor above its top.
        const room = Math.max(0, (list.clientWidth || 0) - (carry.card.offsetWidth || 0));
        const x = Math.min(room, Math.max(0, point.x - carry.dx));
        const y = Math.max(0, point.y - carry.dy);
        carry.card.style.setProperty('--masonry-x', `${Math.round(x)}px`);
        carry.card.style.setProperty('--masonry-y', `${Math.round(y)}px`);
        const laid = inputs.board?.laid?.();
        const own = laid ? laid.items.indexOf(carry.card) : -1;
        if (own < 0) return;
        const width = laid.plan.placements[own].width;
        // Compared by the card's horizontal middle and its top edge, where the pointer holds it, so a tall card is not
        // judged by its far end.
        const distanceAt = (index) => {
            const order = moveWidgetKey(carry.origin, carry.key, index);
            const place = laid.planOrder(order).placements[order.indexOf(carry.key)];
            return place ? Math.hypot(place.left + place.width / 2 - (x + width / 2), place.top - y) : Infinity;
        };
        const held = distanceAt(carry.index);
        let best = carry.index;
        let nearest = held;
        for (let index = 0; index < carry.origin.length; index += 1) {
            if (index === carry.index) continue;
            const distance = distanceAt(index);
            if (distance < nearest) { best = index; nearest = distance; }
        }
        if (best === carry.index || nearest > held - MOVE_HYSTERESIS_PX) return;
        carry.index = best;
        carry.order = moveWidgetKey(carry.origin, carry.key, best);
        inputs.board.carry(carry.key, carry.order);
    };
    const scrollWhileCarrying = () => {
        const scroller = list.closest?.('.widgets-scroll') || list.parentElement;
        const tick = () => {
            scrollFrame = 0;
            if (carry && !carry.card.isConnected) end(false);
            if (!carry || !scroller?.getBoundingClientRect) return;
            const box = scroller.getBoundingClientRect();
            const y = carry.last.clientY;
            let step = 0;
            if (y < box.top + SCROLL_ZONE_PX) step = -Math.ceil((box.top + SCROLL_ZONE_PX - y) / 4);
            else if (y > box.bottom - SCROLL_ZONE_PX) step = Math.ceil((y - box.bottom + SCROLL_ZONE_PX) / 4);
            if (step) {
                const before = scroller.scrollTop;
                scroller.scrollTop += Math.max(-SCROLL_STEP_PX, Math.min(SCROLL_STEP_PX, step));
                if (scroller.scrollTop !== before) follow(carry.last);
            }
            scrollFrame = requestAnimationFrame(tick);
        };
        scrollFrame = requestAnimationFrame(tick);
    };
    const begin = (event) => {
        const start = pending;
        const laid = inputs.board?.laid?.();
        const own = laid ? laid.items.indexOf(start.card) : -1;
        if (own < 0 || !start.card.isConnected) {
            dropPress();
            return;
        }
        pending = null;
        const box = list.getBoundingClientRect();
        const place = laid.plan.placements[own];
        const origin = inputs.currentOrder().slice();
        carry = {
            key: start.key, card: start.card, el: start.el, pointerId: start.pointerId, last: event,
            dx: start.x - box.left - place.left, dy: start.y - box.top - place.top,
            origin, order: origin, index: origin.indexOf(start.key),
        };
        doc.getSelection?.()?.removeAllRanges?.();
        clearTimeout(glideTimer);
        list.setAttribute('data-widget-arranging', '');
        start.card.setAttribute('data-widget-lifted', '');
        haptic('pickup');
        inputs.board.carry(start.key, origin);
        follow(event);
        scrollWhileCarrying();
        announce(`Moving ${titleOf(start.card)}: release to put it there, Escape to put it back`);
    };
    const end = (commit) => {
        const done = carry;
        carry = null;
        stopListening();
        if (scrollFrame) cancelAnimationFrame(scrollFrame);
        scrollFrame = 0;
        letGo(done.el, done.pointerId);
        done.card.removeAttribute('data-widget-lifted');
        const moved = commit && done.order !== done.origin && sameOrder(done.origin);
        inputs.board.carry('', null);
        if (moved) inputs.onOrderChange(done.order);
        glideTimer = setTimeout(() => list.removeAttribute('data-widget-arranging'), GLIDE_MS);
        announce(moved ? placeWords(done.card, done.order) : `${titleOf(done.card)} stays in place`);
    };
    // Leaving the page, or the carried card leaving the board, puts the card back: nothing is saved and nothing goes
    // on listening or scrolling (`cancelWidgetCardCarry`).
    inputs.cancel = () => {
        if (carry) end(false);
        else if (pending) dropPress();
    };
    function onPointerMove(event) {
        const press = carry || pending;
        if (press && !press.card.isConnected) {
            if (carry) end(false);
            else dropPress();
            return;
        }
        if (!press || event.pointerId !== press.pointerId) return;
        // A mouse or pen that moves with no button held was released where the board did not hear it: the press is
        // over, the card stays where it was and nothing is saved.
        if ((event.pointerType === 'mouse' || event.pointerType === 'pen') && event.buttons === 0) {
            if (carry) end(false);
            else dropPress();
            return;
        }
        if (carry) follow(event);
        else if (Math.hypot(event.clientX - pending.x, event.clientY - pending.y) >= MOVE_THRESHOLD_PX) begin(event);
    }
    function onPointerUp(event) {
        if (carry && event.pointerId === carry.pointerId) end(event.type === 'pointerup');
        else if (pending && event.pointerId === pending.pointerId) dropPress();
    }
    function onEscape(event) {
        if (event.key !== 'Escape' || (!carry && !pending)) return;
        event.preventDefault();
        event.stopPropagation();
        if (carry) end(false);
        else dropPress();
    }
    list.addEventListener('pointerdown', (event) => {
        if (carry || pending || event.button !== 0 || !inputs.board) return;
        const card = event.target.closest?.('[data-widget-key]');
        if (!card || card.hasAttribute('data-widget-removed') || inputs.currentOrder().length < 2) return;
        const grip = event.target.closest('[data-widget-reorder-handle]');
        const head = grip ? null : event.target.closest('.widgets-card-head');
        // The head, its title included, is the card's handle, as a window's title bar is: the title is its label, not
        // content (DESIGN: a control may keep selection off its own label), so a press there moves the card.
        const control = event.target.closest('button, a, input, select, textarea, label, dialog, [role="menu"], [contenteditable="true"]');
        if (!grip && !(head && !control && event.pointerType !== 'touch')) return;
        // No text selection starts under a card about to be carried; the handle keeps its keyboard focus path.
        event.preventDefault();
        pending = { card, key: card.dataset.widgetKey || '', pointerId: event.pointerId, x: event.clientX, y: event.clientY,
            el: grip || head };
        doc.addEventListener('pointermove', onPointerMove, true);
        doc.addEventListener('pointerup', onPointerUp, true);
        doc.addEventListener('pointercancel', onPointerUp, true);
        doc.addEventListener('keydown', onEscape, true);
        list.setAttribute('data-widget-pressed', '');
        try { pending.el.setPointerCapture?.(event.pointerId); } catch { /* the pointer may already be gone */ }
    });
}

/** Puts back a card being carried on `list` (or forgets a press about to carry one), saving nothing. */
export function cancelWidgetCardCarry(list) {
    boundLists.get(list)?.cancel?.();
}

// A card narrower than this keeps only ⋮ and the move handle beside its title; its lifecycle row goes under it (style.css).
const NARROW_CARD_PX = 360;
const WIDTH_NAMES = new Map(WIDGET_WIDTH_STEPS.map(({ w, label }) => [w, label.toLowerCase()]));
const widthName = (w) => WIDTH_NAMES.get(w) || `${w} columns`;

/**
 * The card sizes of one Widgets list. `options.tabs()` — the shown cards in
 * key order; `options.prefs()` — the page's current `ui_preferences`;
 * `options.adopt(sizes)` replaces its `widget_size`; `options.save(payload)`
 * POSTs; `options.resized(key, picked)` fits the card to its height now
 * (widgets.js; `picked` is `{hadHeight}` when the owner chose Natural height or
 * Reset size, so its fit starts over); `options.heightInfo(key)` answers
 * `{content, natural}`: the card's content height on screen and the height its
 * content would take (0 when unknown or when its frame is fixed: a taller fixed
 * frame is a bigger working area). The live region is the list's
 * `[data-widget-arrange-status]` sibling.
 *
 * `relayout()` binds the edges of new cards and hands the masonry the key order
 * (or a carried card's preview, `carry`) and the owner's spans; each plan it
 * reports sets the list's `data-widget-layout` (`stack` when the list is too
 * narrow for two columns: widths do not apply there), marks
 * `data-widget-width-fixed` on a card no step can widen or narrow (the only card
 * on the board), whose right edge CSS hides, and places the outline of a carried
 * card's place. The card menu sets a width step, a height or Reset size
 * (`setWidth`, `setHeight`, `resetSize`), each keeping the other axis except
 * Reset. The right edge drags a card between the widths its steps can give it,
 * the bottom edge its height (snapped, taller or shorter than its content, the
 * content's height again within the catch of it), the corner both, each with a
 * live preview (Escape cancels); the right edge's arrow keys step widths (Home /
 * End: one column / full width), the bottom edge's step heights (Home: the
 * least, End: the content's). Show all (`setExpanded`) lifts a height in this window only and is
 * never saved. Every change is named in the live region, shows at once and is
 * saved one write at a time: changes landing meanwhile merge into the next
 * write, so a card's last size is the one stored whatever order the replies
 * arrive in. A list read takes its reader from `beginRead()` as it begins, so
 * its reply never undoes a size written while it was out. A failed write stays
 * on screen with its notice (no change announced meanwhile replaces it) and
 * rides along with the next change's write until one succeeds; nothing retries
 * on its own. Under a finger only the one card being adjusted (`adjust(key)`,
 * the menu's Resize) wears `data-widget-adjusting`: style.css gives it alone the
 * foot with Done and the grip. Done, Escape, a tap outside it (in another
 * card's frame only where the window's blur tells it), leaving Widgets (`cancel`), the
 * card leaving the board and a change of the primary input put it away, a drag
 * on it cancelled first; scrolling keeps it, and neither way writes anything.
 */
export function createWidgetWidths(list, options) {
    const boundHandles = new WeakSet();
    const status = list.parentElement?.querySelector('[data-widget-arrange-status]') || null;
    let drag = null;
    let sizing = null;
    let saving = null;
    let queued = null;
    let unsaved = null;
    let failed = false;
    let disposeBoard = null;
    let laid = null;
    // The one card a finger is adjusting: `{key, card, stop}` (stop removes the listeners that end it), or null.
    let adjusting = null;
    // Each card's width in the last plan.
    const laidWidths = new WeakMap();
    let held = '';
    let preview = null;
    const expanded = new Set();
    // Completed writes, counted, and each key's last written size stamped with
    // that count: a read that began at count `since` shows what was written after.
    let writes = 0;
    const written = new Map();

    const sizes = () => options.prefs().widget_size || {};
    const tabOf = (key) => options.tabs().find((tab) => widgetKey(tab) === key) || null;
    const widthOf = (key) => {
        const tab = tabOf(key);
        return tab ? widgetWidth(tab, sizes()) : 0;
    };
    // A height dragged on screen and not dropped yet is the card's height meanwhile, and ends its Show all meanwhile.
    const previewing = (key) => Boolean(sizing && sizing.key === key && sizing.h !== null);
    const heightOf = (key) => (previewing(key) ? sizing.h : widgetHeight(key, sizes()));
    const announce = (text, tone = 'neutral') => {
        if (!status || (failed && tone !== 'error')) return;
        status.textContent = text;
        status.dataset.tone = tone;
    };
    const resized = (key) => options.resized?.(key);
    // Where a card sits in a plan: its width and its share of the row.
    const placeAt = (plan, index) => ({
        width: plan.placements[index].width, share: plan.placements[index].span / plan.columnCount,
    });
    // The widths the steps can give the card at `index` of the last plan, asked of
    // the masonry for the board as it is (an owner width can change its columns).
    const choicesAt = (index) => widthChoices(
        (w) => placeAt(laid.replan(index, w), index), widthOf(laid.items[index].dataset.widgetKey || ''),
    );
    // Each masonry plan: kept for the edges and a carried card; the list's mode for CSS and the
    // menu; on each card whether any step could change its width, and whether it is narrow
    // (style.css then puts its lifecycle row under its title); and a carried card's place.
    const onLayout = (plan, items, replan, planOrder) => {
        laid = { plan, items, replan, planOrder };
        const mode = plan.availableColumns > 1 ? 'columns' : 'stack';
        if (list.dataset.widgetLayout !== mode) list.dataset.widgetLayout = mode;
        const rewrapped = [];
        items.forEach((item, index) => {
            const fixed = choicesAt(index).length < 2;
            if (item.hasAttribute('data-widget-width-fixed') !== fixed) item.toggleAttribute('data-widget-width-fixed', fixed);
            const { width } = plan.placements[index];
            const narrow = width < NARROW_CARD_PX;
            if (item.hasAttribute('data-widget-narrow') !== narrow) item.toggleAttribute('data-widget-narrow', narrow);
            if (laidWidths.get(item) !== width) rewrapped.push(item.dataset.widgetKey || '');
            laidWidths.set(item, width);
        });
        // A card placed at a new width (the window, a column count, its own width) rewraps its content, which can
        // change what an owner height keeps out of view: it is fitted again in this layout's frame.
        rewrapped.forEach(resized);
        const at = held ? items.findIndex((item) => item.dataset.widgetKey === held) : -1;
        if (at < 0) {
            delete list.dataset.widgetMoving;
            return;
        }
        const place = plan.placements[at];
        list.style.setProperty('--widget-slot-x', `${place.left}px`);
        list.style.setProperty('--widget-slot-y', `${place.top}px`);
        list.style.setProperty('--widget-slot-w', `${place.width}px`);
        list.style.setProperty('--widget-slot-h', `${items[at].offsetHeight}px`);
        list.dataset.widgetMoving = 'true';
    };
    const relayout = () => {
        // The card being adjusted replaced or removed (a new revision, its skill turned off) puts its foot away.
        if (adjusting && (!adjusting.card.isConnected || adjusting.card.hasAttribute('data-widget-removed'))) endAdjust();
        // A card replaced or removed while one of its edges is dragged (a new revision, its skill turned off) ends that
        // drag: its handle hears nothing more.
        if ((drag || sizing) && !(drag || sizing).card.isConnected) {
            cancelDrag();
            return;
        }
        for (const selector of ['[data-widget-resize-handle]', '[data-widget-height-handle]', '[data-widget-corner-handle]']) {
            list.querySelectorAll(selector).forEach(bindHandle);
        }
        const spans = ownerSpans(sizes());
        if (drag?.width) spans[drag.key] = drag.width;
        disposeBoard = applyMasonry(list, { order: preview || options.tabs().map(widgetKey), spans, held, onLayout });
    };

    const flush = () => {
        saving = { ...unsaved, ...queued };
        unsaved = null;
        queued = null;
        Promise.resolve()
            .then(() => options.save({ widget_size: saving }))
            .then(() => {
                writes += 1;
                for (const [key, size] of Object.entries(saving)) written.set(key, { size, at: writes });
                if (!failed) return;
                failed = false;
                announce('');
            })
            .catch((err) => {
                console.warn('Failed to save widget size', err);
                unsaved = saving;
                failed = true;
                announce(`Size not saved: ${err?.message || err}`, 'error');
            })
            .finally(() => {
                saving = null;
                if (queued) flush();
            });
    };

    // One card's size as the owner left it (null: none): shown at once, fitted, saved one write at a time.
    // `picked`: Natural height or Reset size, whose fit starts over (`resized`).
    function store(key, size, picked = false) {
        const hadHeight = widgetHeight(key, sizes()) > 0;
        const next = { ...sizes() };
        if (size) next[key] = size;
        else delete next[key];
        options.adopt(next);
        relayout();
        queued = { ...queued, [key]: size };
        if (!saving) flush();
        options.resized?.(key, picked ? { hadHeight } : null);
    }

    /** A card's width in columns (its height stays); `null` is Reset size, as before heights existed. */
    function setWidth(key, w) {
        if (w === null) {
            resetSize(key);
            return;
        }
        store(key, nextWidgetSize(sizes()[key], { w }));
        announce(`Width: ${widthName(widthOf(key))}`);
    }

    /** A card's content height in px, 0 for its content's own (its width stays); Show all ends. */
    function setHeight(key, h) {
        expanded.delete(key);
        store(key, nextWidgetSize(sizes()[key], { h }), h === 0);
        announce(`Height: ${heightName(widgetHeight(key, sizes()))}`);
    }

    /** Reset size: the author's width and the content's own height. */
    function resetSize(key) {
        expanded.delete(key);
        store(key, null, true);
        announce(`Size reset: ${widthName(widthOf(key))}, natural height`);
    }

    /** Show all (`on`) lifts the card's height in this window only; Back returns to it. Never saved. */
    function setExpanded(key, on) {
        if (Boolean(on) === expanded.has(key)) return;
        if (on) expanded.add(key);
        else expanded.delete(key);
        resized(key);
        relayout();
        announce(on ? 'Showing all of it' : `Back to ${heightName(widgetHeight(key, sizes()))}`);
    }

    /** A carried card (`key`, '' for none) and the order it previews (null: the shown order). */
    function carry(key, order) {
        held = key || '';
        preview = order || null;
        relayout();
    }

    /**
     * A stored map as this window shows it, for a read that began after `since`
     * completed writes: sizes written later, and sizes not written yet, stay on
     * top of the reply.
     */
    function readSizes(stored, since = writes) {
        const next = normalizeWidgetSize(stored);
        const late = {};
        written.forEach(({ size, at }, key) => { if (at > since) late[key] = size; });
        for (const [key, size] of Object.entries({ ...late, ...unsaved, ...saving, ...queued })) {
            if (size) next[key] = size;
            else delete next[key];
        }
        return next;
    }

    /** Called as a list read begins; returns the reader of that read's reply. */
    function beginRead() {
        const since = writes;
        return (stored) => readSizes(stored, since);
    }

    function onWidthKey(event, card) {
        if (event.altKey || event.ctrlKey || event.metaKey) return;
        const key = card.dataset.widgetKey || '';
        const w = widthOf(key);
        const next = {
            ArrowLeft: stepWidgetWidth(w, -1), ArrowRight: stepWidgetWidth(w, 1),
            Home: WIDGET_WIDTH_STEPS[0].w, End: WIDGET_FULL_SPAN,
        }[event.key];
        if (!w || next === undefined) return;
        event.preventDefault();
        // A card with no owner width may be shown wider than its author's span
        // (the masonry widens a lone narrow card): a key that names that span is
        // then a choice to store, not a repeat of what is already on screen. An
        // owner height alone is no owner width.
        if (next !== w || !sizes()[key]?.w) setWidth(key, next);
        else announce(`Width: ${widthName(next)}`);
    }

    function onHeightKey(event, card) {
        if (event.altKey || event.ctrlKey || event.metaKey) return;
        const key = card.dataset.widgetKey || '';
        const info = options.heightInfo?.(key);
        if (!info) return;
        const h = heightOf(key);
        // From the height on screen, as a drag: under Show all that is more than the owner's height.
        const from = info.content || h;
        const next = {
            ArrowUp: () => draggedWidgetHeight(from, -WIDGET_HEIGHT_KEY_STEP, info.natural),
            ArrowDown: () => draggedWidgetHeight(from, WIDGET_HEIGHT_KEY_STEP, info.natural),
            Home: () => WIDGET_HEIGHT_MIN,
            End: () => 0,
        }[event.key];
        if (!next) return;
        event.preventDefault();
        const height = next();
        // Under Show all even the same height is a choice: it ends Show all.
        if (height !== h || expanded.has(key)) setHeight(key, height);
        else announce(`Height: ${heightName(height)}`);
    }

    // What the edge being dragged shows on its card: the width and height it would leave.
    const showChip = (card, text) => {
        if (text) card.dataset.widgetSizeChip = text;
        else delete card.dataset.widgetSizeChip;
    };
    const chipText = (key) => [drag?.key === key && drag.width ? `Width: ${widthName(drag.width)}` : '',
        sizing?.key === key && sizing.h !== null ? `Height: ${heightName(sizing.h)}` : ''].filter(Boolean).join(' · ');

    // The right edge (or the corner) offers the widths the steps can give this card on the board as it is: the
    // pointer's travel moves the card's right edge, and the nearest of those widths is previewed. A width the card
    // already has previews nothing.
    function beginWidth(event, card) {
        const at = laid ? laid.items.indexOf(card) : -1;
        if (drag || at < 0) return false;
        const choices = choicesAt(at);
        // No step changes this card (the only card, or a stack): nothing to drag.
        if (choices.length < 2) return false;
        drag = { key: card.dataset.widgetKey || '', card, choices, from: placeAt(laid.plan, at), width: null, x: event.clientX };
        return true;
    }

    // The bottom edge (or the corner) moves the content's height on the snap grid from the height on screen, above the
    // content's own height as below it; within the catch of it, it is the content's height again. A pointer that has
    // hardly moved, or a height the card already shows (as an owner height: Show all shows what the content's own
    // height shows), previews nothing, as a width the card already has: the card stays as it was, Show all included.
    function beginHeight(event, card) {
        const key = card.dataset.widgetKey || '';
        const info = options.heightInfo?.(key);
        if (sizing || !info) return false;
        const saved = widgetHeight(key, sizes());
        sizing = { key, card, y: event.clientY, from: info.content, natural: info.natural, h: null, saved,
            shown: expanded.has(key) ? 0 : saved };
        return true;
    }

    function begin(event, card, kind) {
        if (drag || sizing || event.button !== 0) return;
        const width = kind !== 'height' && beginWidth(event, card);
        const height = kind !== 'width' && beginHeight(event, card);
        if (!width && !height) return;
        event.preventDefault();
        const handle = event.currentTarget;
        const grip = { handle, pointerId: event.pointerId, card };
        if (drag) Object.assign(drag, grip);
        if (sizing) Object.assign(sizing, grip);
        handle.setPointerCapture?.(event.pointerId);
        list.classList.add('resizing');
        card.classList.add('resizing');
        list.ownerDocument.addEventListener('keydown', onDragKey, true);
    }

    function onDragMove(event) {
        const active = drag || sizing;
        if (!active || event.pointerId !== active.pointerId) return;
        if (!active.card.isConnected) {
            cancelDrag();
            return;
        }
        let changed = false;
        if (drag) {
            const choice = nearestWidthChoice(drag.choices, drag.from.width + event.clientX - drag.x);
            const next = choice.share === drag.from.share ? null : choice.w;
            if (next !== drag.width) {
                drag.width = next;
                changed = true;
            }
        }
        if (sizing) {
            const travel = event.clientY - sizing.y;
            const dragged = Math.abs(travel) < MOVE_THRESHOLD_PX ? null
                : draggedWidgetHeight(sizing.from, travel, sizing.natural);
            const next = dragged === sizing.shown ? null : dragged;
            if (next !== sizing.h) {
                sizing.h = next;
                resized(sizing.key);
                changed = true;
            }
        }
        if (!changed) return;
        showChip(active.card, chipText(active.card.dataset.widgetKey || ''));
        relayout();
    }

    function finishDrag() {
        const done = { width: drag, height: sizing };
        const active = drag || sizing;
        drag = null;
        sizing = null;
        list.classList.remove('resizing');
        active.card.classList.remove('resizing');
        showChip(active.card, '');
        list.ownerDocument.removeEventListener('keydown', onDragKey, true);
        if (active.handle.hasPointerCapture?.(active.pointerId)) active.handle.releasePointerCapture(active.pointerId);
        return done;
    }

    function cancelDrag() {
        if (!drag && !sizing) return;
        const { height } = finishDrag();
        if (height) resized(height.key);
        relayout();
    }

    function onDragKey(event) {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        event.stopPropagation();
        cancelDrag();
    }

    function onDragEnd(event) {
        const active = drag || sizing;
        if (!active || event.pointerId !== active.pointerId) return;
        const { width, height } = finishDrag();
        const newWidth = Boolean(width) && width.width !== null;
        // A previewed height is dropped, so Show all ends, even on the height already saved (no write then).
        const dropped = Boolean(height) && height.h !== null;
        if (dropped) expanded.delete(height.key);
        const newHeight = dropped && height.h !== height.saved;
        if (!newWidth && !newHeight) {
            if (height) resized(height.key);
            relayout();
            if (dropped) announce(`Height: ${heightName(height.h)}`);
            return;
        }
        // One stored size and one sentence for the corner's two axes.
        const key = (width || height).key;
        store(key, nextWidgetSize(sizes()[key], { ...(newWidth ? { w: width.width } : {}), ...(newHeight ? { h: height.h } : {}) }));
        announce([newWidth ? `Width: ${widthName(widthOf(key))}` : '',
            newHeight ? `Height: ${heightName(widgetHeight(key, sizes()))}` : ''].filter(Boolean).join(' · '));
    }

    // Where the primary input is a finger the bottom edge's button is the grip in the card's own foot (style.css), the
    // only one a finger has: it sets the height and, sideways, the width where widths apply, as the corner does.
    const fingerGrip = () => isCoarsePointer({ root: list.ownerDocument?.documentElement ?? null });

    /** Under a finger, `key` alone shows its foot, its grip focused and brought into view; '' ends it (Done). */
    function adjust(key) {
        if (!key) {
            endAdjust({ focus: true });
            return;
        }
        if (!fingerGrip()) return;
        const card = laid?.items.find((item) => item.dataset.widgetKey === key && item.isConnected
            && !item.hasAttribute('data-widget-removed'));
        if (!card) return;
        if (adjusting?.card !== card) {
            endAdjust();
            const doc = list.ownerDocument;
            const win = doc.defaultView;
            // This card's own menu, portalled to the body while it is open, is inside it too.
            const inside = (node) => Boolean(card.contains(node) || (node?.closest?.('[role="menu"]')
                && card.querySelector('[data-widget-menu-trigger][aria-expanded="true"]')));
            // A tap, never a press: a finger that scrolls the board ends in no click.
            const onClick = (event) => {
                if (!inside(event.target)) endAdjust();
                else if (event.target.closest?.('[data-widget-adjust-done]')) endAdjust({ focus: true });
            };
            // A tap in a frame reaches no click here. Where the browser moves the focus to that frame, the window's blur
            // says so; not every such tap does (a frame's own text in WebKit, a second frame in Chromium): Done stays.
            const onBlur = () => {
                const active = doc.activeElement;
                if (active?.tagName === 'IFRAME' && !card.contains(active)) endAdjust();
            };
            // While the focus is on the board, after a menu's or a drag's own Escape, which take it first.
            const onKey = (event) => {
                if (event.key !== 'Escape' || event.defaultPrevented || event.isComposing || !list.contains(doc.activeElement)) return;
                event.preventDefault();
                endAdjust({ focus: card.contains(doc.activeElement) });
            };
            // A mouse or a trackpad has the edges again; a rotation alone changes nothing.
            const stopLayout = onLayoutChange(() => { if (!fingerGrip()) endAdjust(); }, { w: win });
            // WebKit sends a tap's click only where something listens for it: the page hears its own empty margins.
            const page = list.closest?.('.page') || null;
            page?.addEventListener('click', onClick, true);
            doc.addEventListener('click', onClick, true);
            doc.addEventListener('keydown', onKey);
            win?.addEventListener('blur', onBlur);
            adjusting = { key, card, stop() {
                page?.removeEventListener('click', onClick, true);
                doc.removeEventListener('click', onClick, true);
                doc.removeEventListener('keydown', onKey);
                win?.removeEventListener('blur', onBlur);
                stopLayout();
            } };
            card.toggleAttribute('data-widget-adjusting', true);
        }
        const grip = card.querySelector('[data-widget-height-handle]');
        grip?.focus({ preventScroll: true });
        grip?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    // `focus`: back to the card's menu button (Done, Escape), without scrolling the board.
    function endAdjust({ focus = false } = {}) {
        if (!adjusting) return;
        const { card, stop } = adjusting;
        if (drag || sizing) cancelDrag();
        adjusting = null;
        stop();
        card.toggleAttribute('data-widget-adjusting', false);
        if (focus && card.isConnected) card.querySelector('[data-widget-menu-trigger]')?.focus({ preventScroll: true });
    }

    function bindHandle(handle) {
        const card = handle.closest('[data-widget-key]');
        if (!card || boundHandles.has(handle)) return;
        boundHandles.add(handle);
        const kind = handle.hasAttribute?.('data-widget-height-handle') ? 'height'
            : handle.hasAttribute?.('data-widget-corner-handle') ? 'corner' : 'width';
        handle.addEventListener('pointerdown', (event) => begin(event, card, kind === 'height' && fingerGrip() ? 'corner' : kind));
        handle.addEventListener('pointermove', onDragMove);
        handle.addEventListener('pointerup', onDragEnd);
        handle.addEventListener('pointercancel', cancelDrag);
        handle.addEventListener('lostpointercapture', cancelDrag);
        if (kind === 'width') handle.addEventListener('keydown', (event) => onWidthKey(event, card));
        if (kind === 'height') handle.addEventListener('keydown', (event) => onHeightKey(event, card));
    }

    return {
        relayout,
        setWidth,
        setHeight,
        resetSize,
        setExpanded,
        isExpanded: (key) => expanded.has(key) && !previewing(key),
        widthOf,
        heightOf,
        readSizes,
        beginRead,
        carry,
        announce,
        laid: () => laid,
        adjust,
        adjusting: () => adjusting?.key || '',
        canAdjust: fingerGrip,
        /** The owner left the page: an edge being dragged goes back to where it was, saving nothing; a foot goes away. */
        cancel() {
            cancelDrag();
            endAdjust();
        },
        dispose() {
            cancelDrag();
            endAdjust();
            disposeBoard?.();
        },
    };
}
