/* Widgets card arrangement by the owner: the `widget_order` preference applied
   to the card list, the pure key-order move behind a reorder and its merge
   into the stored order (a card not on screen keeps its slot), the drag /
   keyboard reorder handles, and the card widths (`createWidgetWidths`: the
   card menu, the edge handle and its keys over `ui_preferences.widget_size`).
   Nothing here moves an <article>: the masonry (web/modules/masonry.js)
   places the cards by custom properties, so a running frame — retained or
   not — is never reloaded by a reorder or a resize. widgets.js owns persisting
   the order through `/api/ui/preferences`. Disclosed residual: the Tab / focus
   order follows the DOM, so after a visual reorder it can differ from the
   visible order until a window reload rebuilds the cards; keyboard reorder
   through the handle follows the key order. */

import { widgetKey } from './widget_list.js';
import { applyMasonry } from './masonry.js';
import {
    nearestWidthChoice, normalizeWidgetSize, ownerSpans, stepWidgetWidth, WIDGET_FULL_SPAN, WIDGET_WIDTH_STEPS,
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

const WIDTH_NAMES = new Map(WIDGET_WIDTH_STEPS.map(({ w, label }) => [w, label.toLowerCase()]));
const widthName = (w) => WIDTH_NAMES.get(w) || `${w} columns`;

/**
 * The card widths of one Widgets list. `options.tabs()` — the shown cards in
 * key order; `options.prefs()` — the page's current `ui_preferences`;
 * `options.adopt(sizes)` replaces its `widget_size`; `options.save(payload)`
 * POSTs. The live region is the list's `[data-widget-arrange-status]` sibling.
 *
 * `relayout()` binds the edge handles of new cards and hands the masonry the
 * key order and the owner's spans; each plan it reports sets the list's
 * `data-widget-layout` (`stack` when the list is too narrow for two columns:
 * widths do not apply there) and marks `data-widget-width-fixed` on a card no
 * step can widen or narrow (the only card on the board), whose edge CSS hides.
 * The card menu sets a width step or `null` (the author default) through
 * `setWidth`, on every card. The edge handle drags a card between the widths
 * its steps can give it, which the masonry answers for the current board,
 * with a live preview (Escape cancels), and its arrow keys step it (Home /
 * End: one column / full width). Every
 * change, the menu's included, is named in the live region, shows at once and
 * is saved one write at a time: changes landing meanwhile merge into the next
 * write, so a card's last width is the one stored whatever order the replies
 * arrive in. A list read takes its reader from `beginRead()` as it begins, so
 * its reply never undoes a width written while it was out. A failed write
 * stays on screen with its notice (no step announced meanwhile replaces it)
 * and rides along with the next change's write until one succeeds; nothing
 * retries on its own.
 */
export function createWidgetWidths(list, options) {
    const boundHandles = new WeakSet();
    const status = list.parentElement?.querySelector('[data-widget-arrange-status]') || null;
    let drag = null;
    let saving = null;
    let queued = null;
    let unsaved = null;
    let failed = false;
    let disposeBoard = null;
    let laid = null;
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
    const announce = (text, tone = 'neutral') => {
        if (!status || (failed && tone !== 'error')) return;
        status.textContent = text;
        status.dataset.tone = tone;
    };
    // Where a card sits in a plan: its width and its share of the row.
    const placeAt = (plan, index) => ({
        width: plan.placements[index].width, share: plan.placements[index].span / plan.columnCount,
    });
    // The widths the steps can give the card at `index` of the last plan, asked of
    // the masonry for the board as it is (an owner width can change its columns).
    const choicesAt = (index) => widthChoices(
        (w) => placeAt(laid.replan(index, w), index), widthOf(laid.items[index].dataset.widgetKey || ''),
    );
    // Each masonry plan: kept for the edge drag; the list's mode for CSS and the
    // menu; and on each card whether any step could change its width.
    const onLayout = (plan, items, replan) => {
        laid = { plan, items, replan };
        const mode = plan.availableColumns > 1 ? 'columns' : 'stack';
        if (list.dataset.widgetLayout !== mode) list.dataset.widgetLayout = mode;
        items.forEach((item, index) => {
            const fixed = choicesAt(index).length < 2;
            if (item.hasAttribute('data-widget-width-fixed') !== fixed) item.toggleAttribute('data-widget-width-fixed', fixed);
        });
    };
    const relayout = () => {
        list.querySelectorAll('[data-widget-resize-handle]').forEach(bindHandle);
        const spans = ownerSpans(sizes());
        if (drag?.width) spans[drag.key] = drag.width;
        disposeBoard = applyMasonry(list, { order: options.tabs().map(widgetKey), spans, onLayout });
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

    /** A card's width in columns, or `null` for its author default; the live region names it. */
    function setWidth(key, w) {
        const size = w === null ? null : { w, h: 0 };
        const next = { ...sizes() };
        if (size) next[key] = size;
        else delete next[key];
        options.adopt(next);
        relayout();
        queued = { ...queued, [key]: size };
        if (!saving) flush();
        announce(`Width: ${widthName(widthOf(key))}`);
    }

    /**
     * A stored map as this window shows it, for a read that began after `since`
     * completed writes: widths written later, and widths not written yet, stay on
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

    function onKey(event, card) {
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
        // then a choice to store, not a repeat of what is already on screen.
        if (next !== w || !Object.hasOwn(sizes(), key)) setWidth(key, next);
        else announce(`Width: ${widthName(next)}`);
    }

    // The drag offers the widths the steps can give this card on the board as it
    // is: the pointer's travel moves the card's right edge, and the nearest of
    // those widths is previewed. A width the card already has previews nothing.
    function beginDrag(event, card) {
        const at = laid ? laid.items.indexOf(card) : -1;
        if (drag || at < 0 || event.button !== 0) return;
        const choices = choicesAt(at);
        // No step changes this card (the only card, or a stack): nothing to drag.
        if (choices.length < 2) return;
        event.preventDefault();
        drag = {
            key: card.dataset.widgetKey || '', card, choices, from: placeAt(laid.plan, at), width: null,
            handle: event.currentTarget, pointerId: event.pointerId, x: event.clientX,
        };
        drag.handle.setPointerCapture?.(event.pointerId);
        list.classList.add('resizing');
        card.classList.add('resizing');
        list.ownerDocument.addEventListener('keydown', onDragKey, true);
    }

    function onDragMove(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;
        if (!drag.card.isConnected) {
            cancelDrag();
            return;
        }
        const choice = nearestWidthChoice(drag.choices, drag.from.width + event.clientX - drag.x);
        const next = choice.share === drag.from.share ? null : choice.w;
        if (next === drag.width) return;
        drag.width = next;
        relayout();
    }

    function finishDrag() {
        const done = drag;
        drag = null;
        list.classList.remove('resizing');
        done.card.classList.remove('resizing');
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

    function onDragEnd(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;
        const done = finishDrag();
        if (done.width === null) relayout();
        else setWidth(done.key, done.width);
    }

    function bindHandle(handle) {
        const card = handle.closest('[data-widget-key]');
        if (!card || boundHandles.has(handle)) return;
        boundHandles.add(handle);
        handle.addEventListener('pointerdown', (event) => beginDrag(event, card));
        handle.addEventListener('pointermove', onDragMove);
        handle.addEventListener('pointerup', onDragEnd);
        handle.addEventListener('pointercancel', cancelDrag);
        handle.addEventListener('lostpointercapture', cancelDrag);
        handle.addEventListener('keydown', (event) => onKey(event, card));
    }

    return {
        relayout,
        setWidth,
        widthOf,
        readSizes,
        beginRead,
        dispose() {
            cancelDrag();
            disposeBoard?.();
        },
    };
}
