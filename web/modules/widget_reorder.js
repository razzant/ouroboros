/* Widgets card order: the owner's `widget_order` preference applied to the card
   list, the pure key-order move behind a reorder, and the drag / keyboard
   reorder handles on the cards. A reorder never moves an <article>: the visual
   order is the masonry key order (web/modules/masonry.js), so a running frame —
   retained or not — is never reloaded by it. widgets.js owns persisting the
   order through `/api/ui/preferences` and relayouting with it. Disclosed
   residual: the Tab / focus order follows the DOM, so after a visual reorder it
   can differ from the visible order until a window reload rebuilds the cards;
   keyboard reorder through the handle follows the key order. */

import { widgetKey } from './widget_list.js';

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

// Cards keep their DOM node across list patches, so binding is per card, once;
// the drag source is shared by every binding pass over the one Widgets list.
const reorderBoundCards = new WeakSet();
let draggedKey = '';

/**
 * Drag and keyboard reorder on the card handles. `currentOrder()` returns the
 * complete visible key order; a move hands the next order to `onOrderChange`
 * and touches no node.
 */
export function bindWidgetCardReorder(list, currentOrder, onOrderChange) {
    if (!list) return;
    const clearDragState = () => {
        list.querySelectorAll('.widgets-card.dragging, .widgets-card.drag-over').forEach((card) => {
            card.classList.remove('dragging', 'drag-over');
        });
        draggedKey = '';
    };
    const move = (key, toIndex) => {
        const order = currentOrder();
        const next = moveWidgetKey(order, key, toIndex);
        if (next === order) return false;
        onOrderChange(next);
        return true;
    };
    list.querySelectorAll('[data-widget-reorder-handle]').forEach((handle) => {
        const card = handle.closest('[data-widget-key]');
        if (!card || reorderBoundCards.has(card)) return;
        handle.setAttribute('draggable', 'true');
        handle.addEventListener('dragstart', (event) => {
            draggedKey = card.dataset.widgetKey || '';
            if (!draggedKey) return;
            card.classList.add('dragging');
            if (event.dataTransfer) {
                event.dataTransfer.effectAllowed = 'move';
                event.dataTransfer.setData('text/plain', draggedKey);
            }
        });
        handle.addEventListener('dragend', clearDragState);
        handle.addEventListener('keydown', (event) => {
            const key = card.dataset.widgetKey || '';
            const from = currentOrder().indexOf(key);
            if (from < 0) return;
            let toIndex = from;
            if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') toIndex = from - 1;
            else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') toIndex = from + 1;
            else if (event.key === 'Home') toIndex = 0;
            else if (event.key === 'End') toIndex = Number.MAX_SAFE_INTEGER;
            else return;
            if (!move(key, toIndex)) return;
            event.preventDefault();
            clearDragState();
            handle.focus();
        });
    });
    list.querySelectorAll('.widgets-card').forEach((card) => {
        if (reorderBoundCards.has(card)) return;
        reorderBoundCards.add(card);
        card.addEventListener('dragover', (event) => {
            if (!draggedKey || card.dataset.widgetKey === draggedKey) return;
            event.preventDefault();
            card.classList.add('drag-over');
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        });
        card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
        card.addEventListener('drop', (event) => {
            if (!draggedKey || card.dataset.widgetKey === draggedKey) return;
            event.preventDefault();
            const key = draggedKey;
            const targetIndex = currentOrder().indexOf(card.dataset.widgetKey || '');
            clearDragState();
            if (targetIndex >= 0) move(key, targetIndex);
        });
    });
}
