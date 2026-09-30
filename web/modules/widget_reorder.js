/* Widgets card arrangement: the owner's `widget_order` and `widget_layout`
   preferences applied to the card list, the pure key-order move behind a
   stacked reorder, and the move / resize handles on the cards. Nothing here
   moves an <article>: the desktop grid places each card at its saved cell
   (web/modules/widget_grid.js) and the narrow stack orders the cards by key,
   both through custom properties, so a running frame — retained or not — is
   never reloaded by a move, a resize or a reorder. `createWidgetArrangement`
   owns the relayout, the live drag preview and the one-at-a-time preference
   write; widgets.js owns the card list and the preferences it read.
   Disclosed residual: the Tab / focus order follows the DOM, so after an
   arrangement it can differ from the visible order until a window reload
   rebuilds the cards; the handle keys follow the visible arrangement. */

import { widgetKey } from './widget_list.js';
import {
    applyWidgetGrid,
    arrangeWidgetSlot,
    defaultWidgetSize,
    planWidgetGrid,
    WIDGET_GRID_COLUMNS,
    WIDGET_LAYOUT_MAX_ITEMS,
    WIDGET_GRID_GAP_PX,
    WIDGET_GRID_ROW_PX,
    widgetLayoutFromPlacements,
    widgetReadingOrder,
} from './widget_grid.js';

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

// Reorder only visible slots. A disabled skill's key keeps its place in the
// owner's order and reappears there when enabled again.
export function mergeVisibleWidgetOrder(fullOrder, visibleOrder) {
    const visible = normalizeWidgetOrder(visibleOrder);
    const visibleKeys = new Set(visible);
    let index = 0;
    const merged = normalizeWidgetOrder(fullOrder).map((key) => (
        visibleKeys.has(key) ? visible[index++] : key
    ));
    return [...merged, ...visible.slice(index)];
}

export function sortTabsByWidgetOrder(tabs, order) {
    const rank = new Map(normalizeWidgetOrder(order).map((key, idx) => [key, idx]));
    return tabs.map((tab, originalIndex) => ({ tab, originalIndex })).sort((a, b) => {
        const aRank = rank.has(widgetKey(a.tab)) ? rank.get(widgetKey(a.tab)) : Number.MAX_SAFE_INTEGER;
        const bRank = rank.has(widgetKey(b.tab)) ? rank.get(widgetKey(b.tab)) : Number.MAX_SAFE_INTEGER;
        if (aRank !== bRank) return aRank - bRank;
        return a.originalIndex - b.originalIndex;
    }).map((item) => item.tab);
}

/**
 * Pure key-order move: `key` leaves its slot and re-enters at `toIndex`
 * (clamped to the list). Returns the SAME array when nothing changes, so
 * callers test identity for "moved".
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

const STEPS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
// A drag near the scroll container's top or bottom edge scrolls it, so a card
// can travel past the visible rows on touch too (no wheel there).
const EDGE_PX = 48;
const EDGE_STEP_PX = 24;

/**
 * The card arrangement of one Widgets list.
 *
 * `options.tabs()` — the shown cards in key order; `options.prefs()` — the
 * page's current `ui_preferences`; `options.commit(next)` adopts a changed
 * `{ widget_order?, widget_layout? }` in the page; `options.save(next)` POSTs
 * it; `options.status` — an optional live region for keyboard moves.
 *
 * Desktop grid: the move handle drags a card cell by cell (arrow keys: one
 * cell; Home: top-left; End: below every other card), the corner handle
 * resizes it (arrows: one column / row), and a card in the way is pushed
 * straight down. Any grid change pins every shown card at its cell and
 * re-derives `widget_order` from the grid's reading order, so the narrow
 * stack reads like the desktop grid. Narrow stack: the move handle reorders
 * the key order (arrows, Home, End), the corner handle changes the height.
 */
export function createWidgetArrangement(list, options) {
    const boundHandles = new WeakSet();
    let drag = null;
    let disposeGrid = null;
    let revision = 0;
    let inFlight = null;
    let queued = null;

    const order = () => options.tabs().map(widgetKey);
    const plan = () => planWidgetGrid(
        options.tabs().map((tab) => ({ key: widgetKey(tab), ...defaultWidgetSize(tab) })),
        options.prefs().widget_layout,
    );
    const stacked = () => list.dataset.widgetLayout === 'stack';
    const relayout = () => {
        if (!drag) disposeGrid = applyWidgetGrid(list, { placements: plan(), order: order() });
    };
    const announce = (text) => {
        if (!options.status) return;
        options.status.textContent = text;
        options.status.dataset.tone = text.startsWith('Layout not saved') ? 'error' : 'neutral';
    };

    // One write in flight; changes landing meanwhile merge into the next one,
    // so the last arrangement is the one stored whatever order replies arrive in.
    const flush = () => {
        const payload = queued;
        queued = null;
        inFlight = new Promise((resolve) => resolve(options.save(payload)))
            .catch((err) => {
                console.warn('Failed to save widget arrangement', err);
                announce('Layout not saved. Try moving a card again when the connection recovers.');
            })
            .finally(() => {
                inFlight = null;
                if (queued) flush();
            });
    };
    const persist = (next) => {
        if (options.canEdit?.() === false) return;
        revision += 1;
        options.commit(next);
        relayout();
        queued = { ...queued, ...next };
        if (!inFlight) flush();
    };
    const commitOrder = (next) => persist({ widget_order: mergeVisibleWidgetOrder(options.prefs().widget_order, next) });
    const commitPlacements = (placements, reorder) => persist({
        widget_layout: widgetLayoutFromPlacements(placements, options.prefs().widget_layout),
        ...(reorder ? { widget_order: mergeVisibleWidgetOrder(options.prefs().widget_order, widgetReadingOrder(placements)) } : {}),
    });

    function onMoveKey(event, card) {
        if (options.canEdit?.() === false || event.altKey || event.ctrlKey || event.metaKey) return;
        const key = card.dataset.widgetKey || '';
        if (stacked()) {
            const current = order();
            const from = current.indexOf(key);
            const to = { ArrowUp: from - 1, ArrowLeft: from - 1, ArrowDown: from + 1, ArrowRight: from + 1, Home: 0, End: current.length - 1 }[event.key];
            if (from < 0 || to === undefined) return;
            const next = moveWidgetKey(current, key, to);
            if (next === current) return;
            event.preventDefault();
            commitOrder(next);
            announce(`Moved to position ${next.indexOf(key) + 1} of ${next.length}`);
        } else {
            const placements = plan();
            const slot = placements.get(key);
            const step = STEPS[event.key];
            let target = null;
            if (!slot) return;
            if (step) target = { ...slot, x: slot.x + step[0], y: slot.y + step[1] };
            else if (event.key === 'Home') target = { ...slot, x: 0, y: 0 };
            else if (event.key === 'End') {
                const others = [...placements].filter(([other]) => other !== key);
                target = { ...slot, x: 0, y: Math.max(0, ...others.map(([, cell]) => cell.y + cell.h)) };
            }
            const next = target ? arrangeWidgetSlot(placements, key, target) : placements;
            if (next === placements) return;
            event.preventDefault();
            commitPlacements(next, true);
            const cell = next.get(key);
            announce(`Moved to column ${cell.x + 1}, row ${cell.y + 1}`);
        }
        card.scrollIntoView?.({ block: 'nearest' });
    }

    function onResizeKey(event, card) {
        if (options.canEdit?.() === false || event.altKey || event.ctrlKey || event.metaKey) return;
        const key = card.dataset.widgetKey || '';
        const placements = plan();
        const slot = placements.get(key);
        const step = STEPS[event.key];
        // The stacked column is always full width: only the height changes there.
        if (!slot || !step || (stacked() && step[0])) return;
        const next = arrangeWidgetSlot(placements, key, {
            ...slot, w: Math.min(slot.w + step[0], WIDGET_GRID_COLUMNS - slot.x), h: slot.h + step[1],
        });
        if (next === placements) return;
        event.preventDefault();
        commitPlacements(next, !stacked());
        const cell = next.get(key);
        announce(stacked() ? `Resized to ${cell.h} rows` : `Resized to ${cell.w} columns by ${cell.h} rows`);
        card.scrollIntoView?.({ block: 'nearest' });
    }

    const listPoint = (event) => {
        const box = list.getBoundingClientRect();
        return { x: event.clientX - box.left, y: event.clientY - box.top };
    };

    function beginDrag(event, card, kind) {
        if (options.canEdit?.() === false || drag || event.button !== 0) return;
        const key = card.dataset.widgetKey || '';
        const placements = plan();
        if (!placements.has(key)) return;
        event.preventDefault();
        const style = getComputedStyle(list);
        const columnGap = parseFloat(style.columnGap) || WIDGET_GRID_GAP_PX;
        drag = {
            kind,
            key,
            card,
            handle: event.currentTarget,
            pointerId: event.pointerId,
            stacked: stacked(),
            start: placements,
            placements,
            order: order(),
            nextOrder: null,
            cards: new Map(Array.from(list.querySelectorAll('[data-widget-key]'))
                .filter((node) => !node.hasAttribute('data-widget-removed'))
                .map((node) => [node.dataset.widgetKey, node])),
            columnPitch: (list.clientWidth + columnGap) / WIDGET_GRID_COLUMNS,
            rowPitch: (parseFloat(style.gridAutoRows) || WIDGET_GRID_ROW_PX) + (parseFloat(style.rowGap) || WIDGET_GRID_GAP_PX),
            origin: listPoint(event),
        };
        drag.handle.setPointerCapture?.(event.pointerId);
        list.classList.add('arranging');
        card.classList.add('arranging');
        list.ownerDocument.addEventListener('keydown', onDragKey, true);
    }

    function finishDrag() {
        const done = drag;
        drag = null;
        list.classList.remove('arranging');
        done.card.classList.remove('arranging');
        list.ownerDocument.removeEventListener('keydown', onDragKey, true);
        if (done.handle.hasPointerCapture?.(done.pointerId)) done.handle.releasePointerCapture(done.pointerId);
        return done;
    }

    function cancelDrag() {
        if (!drag) return;
        finishDrag();
        relayout();
    }

    function onDragKey(event) {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        event.stopPropagation();
        cancelDrag();
    }

    function edgeScroll(event) {
        const scroller = list.closest?.('.widgets-scroll');
        const box = scroller?.getBoundingClientRect();
        if (!box) return;
        if (event.clientY > box.bottom - EDGE_PX) scroller.scrollTop += EDGE_STEP_PX;
        else if (event.clientY < box.top + EDGE_PX) scroller.scrollTop -= EDGE_STEP_PX;
    }

    // Live preview: the arrangement the drop would commit, recomputed from the
    // drag's START each time, so a card pushed aside returns when the dragged
    // card leaves its way again.
    function onDragMove(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;
        if (!drag.card.isConnected) {
            cancelDrag();
            return;
        }
        edgeScroll(event);
        if (drag.kind === 'move' && drag.stacked) {
            const others = drag.order.filter((key) => key !== drag.key);
            const above = others.filter((key) => {
                const box = drag.cards.get(key)?.getBoundingClientRect();
                return box && box.top + box.height / 2 < event.clientY;
            }).length;
            drag.nextOrder = [...others.slice(0, above), drag.key, ...others.slice(above)];
            applyWidgetGrid(list, { placements: drag.start, order: drag.nextOrder });
            return;
        }
        const at = listPoint(event);
        const dx = Math.round((at.x - drag.origin.x) / drag.columnPitch);
        const dy = Math.round((at.y - drag.origin.y) / drag.rowPitch);
        const slot = drag.start.get(drag.key);
        const target = drag.kind === 'move'
            ? { ...slot, x: slot.x + dx, y: slot.y + dy }
            : { ...slot, w: drag.stacked ? slot.w : Math.min(slot.w + dx, WIDGET_GRID_COLUMNS - slot.x), h: slot.h + dy };
        drag.placements = arrangeWidgetSlot(drag.start, drag.key, target);
        applyWidgetGrid(list, { placements: drag.placements, order: drag.order });
    }

    function onDragEnd(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;
        const done = finishDrag();
        if (done.nextOrder && done.nextOrder.join('\n') !== done.order.join('\n')) commitOrder(done.nextOrder);
        else if (done.placements !== done.start) commitPlacements(done.placements, !done.stacked);
        else relayout();
    }

    function bindHandle(handle, kind) {
        const card = handle.closest('[data-widget-key]');
        if (!card || boundHandles.has(handle)) return;
        boundHandles.add(handle);
        handle.addEventListener('pointerdown', (event) => beginDrag(event, card, kind));
        handle.addEventListener('pointermove', onDragMove);
        handle.addEventListener('pointerup', onDragEnd);
        handle.addEventListener('pointercancel', cancelDrag);
        handle.addEventListener('lostpointercapture', cancelDrag);
        handle.addEventListener('keydown', (event) => (kind === 'move' ? onMoveKey : onResizeKey)(event, card));
    }

    return {
        relayout,
        // The first successful preferences read pins every newly shown card.
        // Before this, removing a sibling would repack the default positions.
        // A failed preferences read never authors an empty replacement map.
        pinDefaults() {
            const stored = options.prefs().widget_layout || {};
            const missing = order().filter((key) => !Object.prototype.hasOwnProperty.call(stored, key));
            if (!missing.length || Object.keys(stored).length + missing.length > WIDGET_LAYOUT_MAX_ITEMS) return;
            persist({ widget_layout: widgetLayoutFromPlacements(plan(), stored) });
        },
        /** Bind the handles of cards added since the last call (each once). */
        bind() {
            list.querySelectorAll('[data-widget-move-handle]').forEach((handle) => bindHandle(handle, 'move'));
            list.querySelectorAll('[data-widget-resize-handle]').forEach((handle) => bindHandle(handle, 'resize'));
        },
        // A preferences read begun at revision `since` is older than an
        // arrangement made — or still being written — after it; the page keeps
        // its own order and layout over such a read.
        revision: () => revision,
        settled: (since) => since === revision && !inFlight && !queued,
        dispose() {
            cancelDrag();
            disposeGrid?.();
        },
    };
}
