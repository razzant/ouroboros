/* Widgets card grid: the owner's desktop arrangement of the cards as stable
   cells of a 12-column grid — each card at a fixed column and row with a
   fixed width and height in grid units (`ui_preferences.widget_layout`, keyed
   like `widget_order`) — and the one stacked column a narrow list falls back
   to. Nothing here measures a card: a placement is a pure function of the
   saved slots, the key order and each card's declared default size, so
   content that grows, shrinks or starts never moves or resizes a card; a card
   whose content is taller than its cell scrolls inside its body.
   `applyWidgetGrid` writes a placement ONLY as custom properties —
   `--widget-col/-row/-w/-h` and `--widget-order` on each card — plus the
   `data-widget-layout` mode on the list, which one static rule set in
   web/style.css turns into grid lines. No node is ever moved, so a running
   <iframe> is never reloaded by a move, a resize or a mode switch.
   The bounds mirror ouroboros/gateway/ui_preferences.py. */

import { frameHeight } from './widget_module.js';

export const WIDGET_GRID_COLUMNS = 12;
export const WIDGET_GRID_MIN_W = 3;
export const WIDGET_GRID_MIN_H = 4;
export const WIDGET_GRID_MAX_H = 48;
export const WIDGET_GRID_MAX_Y = 10000;
export const WIDGET_LAYOUT_MAX_ITEMS = 200;
const WIDGET_KEY_MAX_LENGTH = 200;
// One grid row and the gap between rows / columns (`.widgets-list` in
// web/style.css), and the card chrome — padding, border, head — that a default
// height adds above the declared body.
export const WIDGET_GRID_ROW_PX = 40;
export const WIDGET_GRID_GAP_PX = 14;
const WIDGET_CARD_CHROME_PX = 72;
// Below this list width the grid falls back to one stacked column. The band
// around it keeps a scrollbar that appears or leaves with the other mode's
// height from flipping the mode straight back.
export const WIDGET_GRID_STACK_BELOW_PX = 720;
const WIDGET_GRID_MODE_BAND_PX = 24;

const bound = new WeakMap();

function boundedInt(value, lo, hi) {
    return Math.max(lo, Math.min(hi, Math.trunc(Number(value) || 0)));
}

/** One saved slot clamped into the grid; null for anything but four integers. */
export function normalizeWidgetSlot(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    if (!['x', 'y', 'w', 'h'].every((name) => Number.isInteger(value[name]))) return null;
    const w = boundedInt(value.w, WIDGET_GRID_MIN_W, WIDGET_GRID_COLUMNS);
    return {
        x: boundedInt(value.x, 0, WIDGET_GRID_COLUMNS - w),
        y: boundedInt(value.y, 0, WIDGET_GRID_MAX_Y),
        w,
        h: boundedInt(value.h, WIDGET_GRID_MIN_H, WIDGET_GRID_MAX_H),
    };
}

/** The saved `widget_layout` map, bounded the way the server stores it. */
export function normalizeWidgetLayout(value) {
    const layout = {};
    if (!value || typeof value !== 'object' || Array.isArray(value)) return layout;
    for (const [rawKey, rawSlot] of Object.entries(value).slice(0, WIDGET_LAYOUT_MAX_ITEMS)) {
        const key = String(rawKey || '').trim();
        const slot = normalizeWidgetSlot(rawSlot);
        if (key && key.length <= WIDGET_KEY_MAX_LENGTH && slot) layout[key] = slot;
    }
    return layout;
}

/**
 * A card's size before the owner sets one: a third of the grid, two thirds for
 * a `span: 2` card, and enough rows for its declared frame height (the frame
 * floor for a declarative card or an auto-height module) under the card head.
 */
export function defaultWidgetSize(tab) {
    const span = Number(tab?.span || tab?.grid_span || 1);
    const body = frameHeight(tab?.render || {});
    const pitch = WIDGET_GRID_ROW_PX + WIDGET_GRID_GAP_PX;
    return {
        w: span >= 2 ? 8 : 4,
        h: boundedInt(Math.ceil((body + WIDGET_CARD_CHROME_PX + WIDGET_GRID_GAP_PX) / pitch), WIDGET_GRID_MIN_H, WIDGET_GRID_MAX_H),
    };
}

// Row occupancy as one column bitmask per row, so a fit test or a placement
// costs the card's height, not the number of cards already placed.
function occupancy() {
    const rows = [];
    const mask = (slot) => ((1 << slot.w) - 1) << slot.x;
    return {
        fits(slot) {
            for (let row = slot.y; row < slot.y + slot.h; row += 1) {
                if ((rows[row] || 0) & mask(slot)) return false;
            }
            return true;
        },
        take(slot) {
            for (let row = slot.y; row < slot.y + slot.h; row += 1) rows[row] = (rows[row] || 0) | mask(slot);
            return slot;
        },
    };
}

// The slot in its own columns, at its own row or pushed straight down to the
// first row where it overlaps nothing placed before it. Never up, never sideways.
function settle(grid, slot) {
    let y = slot.y;
    while (!grid.fits({ ...slot, y })) y += 1;
    return grid.take({ ...slot, y });
}

function byCell([aKey, a], [bKey, b]) {
    return a.y - b.y || a.x - b.x || (aKey < bKey ? -1 : aKey > bKey ? 1 : 0);
}

/**
 * Every card's cell. Saved slots first, in reading order, each at its own cell
 * or pushed straight down past one it would overlap; then the cards without a
 * saved slot, in key order, packed first-fit BELOW the saved arrangement, so a
 * new card never lands inside the owner's composition. `cards` is
 * `[{ key, w, h }]` in key order (w/h: the default size); returns a Map
 * key → { x, y, w, h } in the same order. No measured size is an input.
 */
export function planWidgetGrid(cards, layout = {}) {
    const grid = occupancy();
    const placements = new Map();
    const saved = cards
        .map((card) => [card.key, Object.prototype.hasOwnProperty.call(layout || {}, card.key) ? normalizeWidgetSlot(layout[card.key]) : null])
        .filter(([, slot]) => slot)
        .sort(byCell);
    let floor = 0;
    for (const [key, slot] of saved) {
        const placed = settle(grid, slot);
        placements.set(key, placed);
        floor = Math.max(floor, placed.y + placed.h);
    }
    for (const card of cards) {
        if (placements.has(card.key)) continue;
        const w = boundedInt(card.w, WIDGET_GRID_MIN_W, WIDGET_GRID_COLUMNS);
        const h = boundedInt(card.h, WIDGET_GRID_MIN_H, WIDGET_GRID_MAX_H);
        let placed = null;
        for (let y = floor; !placed; y += 1) {
            for (let x = 0; x + w <= WIDGET_GRID_COLUMNS && !placed; x += 1) {
                if (grid.fits({ x, y, w, h })) placed = grid.take({ x, y, w, h });
            }
        }
        placements.set(card.key, placed);
    }
    return new Map(cards.map((card) => [card.key, placements.get(card.key)]));
}

function sameSlot(a, b) {
    return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

/**
 * One owner move or resize: `key` takes `slot` (clamped into the grid) and
 * every other card keeps its cell unless it would overlap, in which case it
 * is pushed straight down — in reading order, so a pushed card pushes the
 * cards below it in turn. Nothing is compacted upwards. Returns the SAME Map
 * when the slot does not change, so callers test identity for "changed".
 */
export function arrangeWidgetSlot(placements, key, slot) {
    const current = placements.get(key);
    const next = normalizeWidgetSlot(slot);
    if (!current || !next || sameSlot(current, next)) return placements;
    const grid = occupancy();
    const result = new Map([[key, grid.take(next)]]);
    [...placements].filter(([other]) => other !== key).sort(byCell)
        .forEach(([other, cell]) => result.set(other, settle(grid, cell)));
    return new Map([...placements.keys()].map((other) => [other, result.get(other)]));
}

/** Keys by the cell they start in: top to bottom, then left to right. */
export function widgetReadingOrder(placements) {
    return [...placements].sort(byCell).map(([key]) => key);
}

/**
 * The `widget_layout` write for an arrangement: every card shown now, pinned
 * at its cell, then the saved slots of cards not shown — a disabled skill
 * keeps its place, the `widget_start_mode` rule — within the stored bound.
 */
export function widgetLayoutFromPlacements(placements, previous = {}) {
    const entries = [...placements].map(([key, { x, y, w, h }]) => [key, { x, y, w, h }]);
    for (const [key, slot] of Object.entries(normalizeWidgetLayout(previous))) {
        if (!placements.has(key)) entries.push([key, slot]);
    }
    return Object.fromEntries(entries.slice(0, WIDGET_LAYOUT_MAX_ITEMS));
}

/** `grid` or `stack` for a list this wide; a zero width (a hidden page) keeps the mode. */
export function widgetGridMode(width, previous = 'grid') {
    if (!(width > 0)) return previous;
    const half = WIDGET_GRID_MODE_BAND_PX / 2;
    if (previous === 'stack') return width >= WIDGET_GRID_STACK_BELOW_PX + half ? 'grid' : 'stack';
    return width < WIDGET_GRID_STACK_BELOW_PX - half ? 'stack' : 'grid';
}

// Only a changed value is written, so an unchanged plan touches no style attribute.
function writeCell(style, slot, rank) {
    const [col, row, w, h, order] = [slot.x + 1, slot.y + 1, slot.w, slot.h, rank].map(String);
    if (style.getPropertyValue('--widget-col') !== col) style.setProperty('--widget-col', col);
    if (style.getPropertyValue('--widget-row') !== row) style.setProperty('--widget-row', row);
    if (style.getPropertyValue('--widget-w') !== w) style.setProperty('--widget-w', w);
    if (style.getPropertyValue('--widget-h') !== h) style.setProperty('--widget-h', h);
    if (style.getPropertyValue('--widget-order') !== order) style.setProperty('--widget-order', order);
}

/**
 * Bind (once per list) and write a placement: grid lines per card, and the
 * card's rank in `order` for the stacked column. A card marked
 * `data-widget-removed` (its frame still stopping in order) keeps the cell it
 * had. The list's one ResizeObserver only switches `data-widget-layout`
 * between `grid` and `stack` by the list's own width — no card size feeds
 * back into it — one frame later, because the switch changes the observed
 * list's height and a change inside the callback would be an observer loop.
 * Every call returns the list's one idempotent disposer.
 */
export function applyWidgetGrid(container, { placements = new Map(), order = [] } = {}) {
    if (!container) return () => {};
    let entry = bound.get(container);
    if (!entry) {
        let frame = 0;
        const syncMode = () => {
            const mode = widgetGridMode(container.clientWidth, container.dataset.widgetLayout || 'grid');
            if (container.dataset.widgetLayout !== mode) container.dataset.widgetLayout = mode;
        };
        const observer = new ResizeObserver(() => {
            if (!frame) frame = requestAnimationFrame(() => {
                frame = 0;
                syncMode();
            });
        });
        entry = {
            dispose() {
                if (bound.get(container) !== entry) return;
                bound.delete(container);
                observer.disconnect();
                if (frame) cancelAnimationFrame(frame);
                frame = 0;
            },
        };
        bound.set(container, entry);
        observer.observe(container);
        syncMode();
    }
    const rank = new Map(order.map((key, index) => [key, index]));
    container.querySelectorAll('[data-widget-key]').forEach((card) => {
        const key = card.dataset.widgetKey || '';
        const slot = placements.get(key);
        if (!slot || card.hasAttribute('data-widget-removed')) return;
        writeCell(card.style, slot, rank.has(key) ? rank.get(key) : order.length);
    });
    return entry.dispose;
}
