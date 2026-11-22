/* The owner's card sizes on the Widgets board (docs/DESIGN.md "Widgets board").
   The masonry (web/modules/masonry.js) packs the cards in the owner's order;
   a card spans as many of its columns as the owner chose
   (`ui_preferences.widget_size`, `{w, h}`) or, until the owner picks one, the
   author's `span` (1 or 2). A width is a column count; WIDGET_FULL_SPAN means
   every column the board has, 0 the author's span. A height is the content's
   viewport in px (a framed card's frame, a host-drawn card's body), 0 as tall
   as its content; the two axes are independent, so changing one keeps the other.
   Nothing here measures or writes the DOM. The bounds mirror
   ouroboros/gateway/ui_preferences.py. */

import { widgetKey } from './widget_list.js';

export const WIDGET_FULL_SPAN = 12;
export const WIDGET_SIZE_MAX_ITEMS = 200;
export const WIDGET_HEIGHT_MIN = 120;
export const WIDGET_HEIGHT_MAX = 8192;
// A dragged height lands on this grid; arrow keys on the bottom edge step by WIDGET_HEIGHT_KEY_STEP.
export const WIDGET_HEIGHT_SNAP = 8;
export const WIDGET_HEIGHT_KEY_STEP = 40;
// A drag that ends this close to the content's own height is the content's height: the owner's height goes.
export const WIDGET_HEIGHT_CATCH = 20;
// The card menu's heights, beside Natural height.
export const WIDGET_HEIGHT_PRESETS = Object.freeze([360, 560, 760]);
const WIDGET_KEY_MAX_LENGTH = 200;
// The owner's width steps, in columns of the board; the masonry decides how
// many columns fit, and a step wider than that fills the row.
export const WIDGET_WIDTH_STEPS = Object.freeze([
    { w: 1, label: '1 column' },
    { w: 2, label: '2 columns' },
    { w: 3, label: '3 columns' },
    { w: WIDGET_FULL_SPAN, label: 'Full width' },
]);

/** An owner height clamped to its bounds; anything not a positive number is 0 (as tall as the content). */
export function clampWidgetHeight(value) {
    const height = Math.round(Number(value));
    return Number.isFinite(height) && height > 0 ? Math.max(WIDGET_HEIGHT_MIN, Math.min(WIDGET_HEIGHT_MAX, height)) : 0;
}

/** The saved `widget_size` map, bounded the way the server stores it: a size with neither axis set is none. */
export function normalizeWidgetSize(value) {
    const sizes = {};
    if (!value || typeof value !== 'object' || Array.isArray(value)) return sizes;
    for (const [rawKey, size] of Object.entries(value).slice(0, WIDGET_SIZE_MAX_ITEMS)) {
        const key = String(rawKey || '').trim();
        if (!key || key.length > WIDGET_KEY_MAX_LENGTH || !Number.isInteger(size?.w)) continue;
        const w = Math.max(0, Math.min(WIDGET_FULL_SPAN, size.w));
        const h = Number.isInteger(size.h) ? clampWidgetHeight(size.h) : 0;
        if (w || h) sizes[key] = { w, h };
    }
    return sizes;
}

/**
 * The size a card keeps when the owner changes one axis: `change.w` or `change.h` (0 or null: the author's width,
 * the content's height) replaces that axis and keeps the other, so a height never pins the width the board chose.
 * `null` when neither is set (the card has no owner size).
 */
export function nextWidgetSize(current, change) {
    const w = Object.hasOwn(change, 'w') ? Math.max(0, Number(change.w) || 0) : (current?.w || 0);
    const h = Object.hasOwn(change, 'h') ? clampWidgetHeight(change.h) : (current?.h || 0);
    return w || h ? { w, h } : null;
}

/** The author's default width: two columns for `span: 2`, else one. */
export function defaultWidgetWidth(tab) {
    return Number(tab?.span || tab?.grid_span || 1) >= 2 ? 2 : 1;
}

/** A card's width setting in columns: the owner's over the author's default. */
export function widgetWidth(tab, sizes) {
    return sizes?.[widgetKey(tab)]?.w || defaultWidgetWidth(tab);
}

/** The owner's height of the card `key` in px, 0 when its content decides. */
export function widgetHeight(key, sizes) {
    return sizes?.[key]?.h || 0;
}

/** The owner's widths by card key, the `spans` the masonry takes. */
export function ownerSpans(sizes) {
    return Object.fromEntries(Object.entries(sizes || {}).filter(([, size]) => size?.w).map(([key, size]) => [key, size.w]));
}

/**
 * What the steps can make of one card: `placeOf(w)` answers the card's width
 * (px) and its share of the row (columns spanned / the board's columns) under
 * step `w`; the masonry plans it, and an owner width can change the number of
 * columns. Steps with the same share are one choice (their pixel widths differ
 * by rounding at most), kept as the card's `current` step when it is among them
 * and as the narrower step otherwise. One choice means no step changes the card.
 */
export function widthChoices(placeOf, current = 0) {
    const choices = [];
    for (const { w } of WIDGET_WIDTH_STEPS) {
        const { width, share } = placeOf(w);
        const same = choices.find((choice) => choice.share === share);
        if (!same) choices.push({ w, width, share });
        else if (w === current) same.w = w;
    }
    return choices;
}

/** The choice whose width is nearest `width` (px), a tie keeping the narrower. */
export function nearestWidthChoice(choices, width) {
    return choices.reduce((best, choice) => {
        const nearer = Math.abs(choice.width - width) - Math.abs(best.width - width);
        return nearer < 0 || (nearer === 0 && choice.width < best.width) ? choice : best;
    });
}

/** The next step wider (`delta` > 0) or narrower than `w`, held at the first and last step. */
export function stepWidgetWidth(w, delta) {
    const steps = WIDGET_WIDTH_STEPS.map((step) => step.w);
    if (delta > 0) return steps.find((step) => step > w) ?? steps[steps.length - 1];
    return steps.slice().reverse().find((step) => step < w) ?? steps[0];
}

/**
 * The height a drag of the bottom edge asks for: `from` px moved by `delta`, on the snap grid and within bounds.
 * `natural` is the content's own height when the board knows it (0 when it does not): a height within
 * WIDGET_HEIGHT_CATCH of it, on either side, is the content's height, answered as 0; any other height is taken as
 * set, above the content as below it.
 */
export function draggedWidgetHeight(from, delta, natural = 0) {
    const raw = Math.max(WIDGET_HEIGHT_MIN, Math.round((Number(from) + Number(delta)) / WIDGET_HEIGHT_SNAP) * WIDGET_HEIGHT_SNAP);
    if (natural > 0 && Math.abs(raw - natural) <= WIDGET_HEIGHT_CATCH) return 0;
    return clampWidgetHeight(raw);
}

/** How a height reads in the live region and the menu: `432 px`, or `natural` for the content's own height. */
export function heightName(h) {
    return h ? `${Math.round(h)} px` : 'natural';
}
