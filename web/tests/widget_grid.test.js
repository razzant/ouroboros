import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    applyWidgetGrid,
    arrangeWidgetSlot,
    defaultWidgetSize,
    normalizeWidgetLayout,
    normalizeWidgetSlot,
    planWidgetGrid,
    WIDGET_GRID_COLUMNS,
    WIDGET_GRID_MAX_H,
    WIDGET_GRID_MAX_Y,
    WIDGET_GRID_MIN_H,
    WIDGET_GRID_MIN_W,
    WIDGET_LAYOUT_MAX_ITEMS,
    widgetGridMode,
    widgetLayoutFromPlacements,
    widgetReadingOrder,
} from '../modules/widget_grid.js';

// Widgets grid: every card keeps a stable cell and a fixed size in grid units
// (`ui_preferences.widget_layout`); nothing measured is ever an input, and the
// plan reaches the DOM only as custom properties — no node is moved.

const slots = (placements) => Object.fromEntries(placements);
const card = (key, w = 4, h = 8) => ({ key, w, h });

test('a saved slot is four integers clamped into the grid; anything else is no slot', () => {
    assert.deepEqual(normalizeWidgetSlot({ x: 2, y: 5, w: 6, h: 10 }), { x: 2, y: 5, w: 6, h: 10 });
    // Width first, then x follows the clamped width; y and h clamp to their bounds.
    assert.deepEqual(normalizeWidgetSlot({ x: 11, y: -3, w: 40, h: 999 }), { x: 0, y: 0, w: WIDGET_GRID_COLUMNS, h: WIDGET_GRID_MAX_H });
    assert.deepEqual(normalizeWidgetSlot({ x: 11, y: WIDGET_GRID_MAX_Y + 5, w: 1, h: 0 }), { x: WIDGET_GRID_COLUMNS - WIDGET_GRID_MIN_W, y: WIDGET_GRID_MAX_Y, w: WIDGET_GRID_MIN_W, h: WIDGET_GRID_MIN_H });
    for (const bad of [null, [], 'x', { x: 0, y: 0, w: 4 }, { x: 0, y: 0, w: 4, h: 1.5 }, { x: '0', y: 0, w: 4, h: 8 }, { x: true, y: 0, w: 4, h: 8 }]) {
        assert.equal(normalizeWidgetSlot(bad), null, JSON.stringify(bad));
    }
});

test('the saved layout map is bounded like the server stores it', () => {
    assert.deepEqual(normalizeWidgetLayout(null), {});
    assert.deepEqual(normalizeWidgetLayout([]), {});
    assert.deepEqual(normalizeWidgetLayout({
        ' demo:a ': { x: 0, y: 0, w: 4, h: 8 },
        '': { x: 0, y: 0, w: 4, h: 8 },
        ['x'.repeat(201)]: { x: 0, y: 0, w: 4, h: 8 },
        'demo:bad': { x: 'no' },
    }), { 'demo:a': { x: 0, y: 0, w: 4, h: 8 } });
    const many = Object.fromEntries(Array.from({ length: 250 }, (_, i) => [`demo:${i}`, { x: 0, y: i, w: 4, h: 4 }]));
    assert.equal(Object.keys(normalizeWidgetLayout(many)).length, WIDGET_LAYOUT_MAX_ITEMS);
});

test('a card without a saved slot is sized from its span and declared frame height, never its content', () => {
    // 320 px frame floor + card chrome → 8 rows of 40 px with 14 px gaps.
    assert.deepEqual(defaultWidgetSize({ render: { kind: 'module', entry: 'w.js' } }), { w: 4, h: 8 });
    assert.deepEqual(defaultWidgetSize({ span: 2, render: { kind: 'module', entry: 'w.js' } }), { w: 8, h: 8 });
    assert.deepEqual(defaultWidgetSize({ grid_span: 2, render: { kind: 'iframe', route: 'v', height: 480 } }), { w: 8, h: 11 });
    assert.deepEqual(defaultWidgetSize({ render: { kind: 'declarative', components: [] } }), { w: 4, h: 8 });
    assert.equal(defaultWidgetSize({ render: { kind: 'module', entry: 'w.js', height: 8192 } }).h, WIDGET_GRID_MAX_H);
});

test('cards without saved slots pack first-fit in key order', () => {
    const plan = planWidgetGrid([card('a', 8), card('b'), card('c'), card('d', 8, 4)]);
    assert.deepEqual([...plan.keys()], ['a', 'b', 'c', 'd']);
    assert.deepEqual(slots(plan), {
        a: { x: 0, y: 0, w: 8, h: 8 },
        b: { x: 8, y: 0, w: 4, h: 8 },
        c: { x: 0, y: 8, w: 4, h: 8 },
        d: { x: 4, y: 8, w: 8, h: 4 },
    });
    // Deterministic: the same inputs give the same cells on every revisit.
    assert.deepEqual(slots(planWidgetGrid([card('a', 8), card('b'), card('c'), card('d', 8, 4)])), slots(plan));
});

test('saved slots win exactly; an overlap is pushed straight down; unsaved cards go below the saved arrangement', () => {
    const layout = {
        a: { x: 6, y: 3, w: 6, h: 5 },
        b: { x: 0, y: 0, w: 3, h: 4 },
        // Overlaps a (stale or clamped slot): keeps its columns, moves below a.
        c: { x: 8, y: 4, w: 4, h: 4 },
    };
    const plan = planWidgetGrid([card('a'), card('b'), card('c'), card('new')], layout);
    assert.deepEqual(slots(plan), {
        a: { x: 6, y: 3, w: 6, h: 5 },
        b: { x: 0, y: 0, w: 3, h: 4 },
        c: { x: 8, y: 8, w: 4, h: 4 },
        // Not inside the owner's composition (the free cells at x 3..5 stay free).
        new: { x: 0, y: 12, w: 4, h: 8 },
    });
});

test('adding, removing or updating other cards never moves a saved card', () => {
    const layout = {
        a: { x: 0, y: 0, w: 6, h: 6 },
        b: { x: 6, y: 0, w: 6, h: 10 },
        c: { x: 0, y: 6, w: 6, h: 4 },
    };
    const base = slots(planWidgetGrid([card('a'), card('b'), card('c')], layout));
    const added = slots(planWidgetGrid([card('a'), card('z', 12, 20), card('b'), card('c')], layout));
    assert.deepEqual({ a: added.a, b: added.b, c: added.c }, base);
    assert.deepEqual(added.z, { x: 0, y: 10, w: 12, h: 20 });
    // A removed card leaves its cell empty; nothing is compacted into it.
    const removed = slots(planWidgetGrid([card('b'), card('c')], layout));
    assert.deepEqual(removed, { b: base.b, c: base.c });
    // A changed declaration (span, declared height) only changes the DEFAULT size.
    const updated = slots(planWidgetGrid([card('a', 12, 40), card('b', 3, 4), card('c')], layout));
    assert.deepEqual(updated, base);
    // A saved slot of a card that is not shown occupies nothing.
    assert.deepEqual(slots(planWidgetGrid([card('c')], layout)), { c: base.c });
});

test('a move pushes the cards in its way straight down, in cascade, and compacts nothing', () => {
    const plan = planWidgetGrid([card('a'), card('b'), card('c')], {
        a: { x: 0, y: 0, w: 4, h: 4 },
        b: { x: 4, y: 0, w: 4, h: 4 },
        c: { x: 4, y: 4, w: 4, h: 4 },
    });
    const moved = arrangeWidgetSlot(plan, 'a', { x: 3, y: 1, w: 4, h: 4 });
    assert.deepEqual(slots(moved), {
        a: { x: 3, y: 1, w: 4, h: 4 },
        b: { x: 4, y: 5, w: 4, h: 4 },
        c: { x: 4, y: 9, w: 4, h: 4 },
    });
    assert.deepEqual(slots(plan).a, { x: 0, y: 0, w: 4, h: 4 }, 'the input map is never mutated');
    // Moving down leaves the old cells empty; the cards above do not follow.
    const down = arrangeWidgetSlot(plan, 'b', { x: 4, y: 20, w: 4, h: 4 });
    assert.deepEqual(slots(down), { a: slots(plan).a, b: { x: 4, y: 20, w: 4, h: 4 }, c: slots(plan).c });
});

test('an unchanged, clamped-to-unchanged or unknown move returns the same map', () => {
    const plan = planWidgetGrid([card('a'), card('b')]);
    assert.equal(arrangeWidgetSlot(plan, 'a', { ...plan.get('a') }), plan);
    assert.equal(arrangeWidgetSlot(plan, 'a', { ...plan.get('a'), x: -4, y: -1 }), plan);
    assert.equal(arrangeWidgetSlot(plan, 'zzz', { x: 0, y: 0, w: 4, h: 4 }), plan);
    assert.equal(arrangeWidgetSlot(plan, 'a', { x: 0, y: 0 }), plan);
});

test('a resize is bounded and pushes what it grows into', () => {
    const plan = planWidgetGrid([card('a'), card('b')], { a: { x: 0, y: 0, w: 4, h: 4 }, b: { x: 0, y: 4, w: 4, h: 4 } });
    const taller = arrangeWidgetSlot(plan, 'a', { x: 0, y: 0, w: 4, h: 6 });
    assert.deepEqual(slots(taller), { a: { x: 0, y: 0, w: 4, h: 6 }, b: { x: 0, y: 6, w: 4, h: 4 } });
    const tiny = arrangeWidgetSlot(plan, 'a', { x: 0, y: 0, w: 1, h: 1 });
    assert.deepEqual(tiny.get('a'), { x: 0, y: 0, w: WIDGET_GRID_MIN_W, h: WIDGET_GRID_MIN_H });
});

test('reading order and the pinned layout write', () => {
    const plan = planWidgetGrid([card('a'), card('b'), card('c')], {
        a: { x: 6, y: 4, w: 4, h: 4 },
        b: { x: 0, y: 4, w: 4, h: 4 },
        c: { x: 8, y: 0, w: 4, h: 4 },
    });
    assert.deepEqual(widgetReadingOrder(plan), ['c', 'b', 'a']);
    const previous = { gone: { x: 0, y: 0, w: 12, h: 4 }, a: { x: 0, y: 40, w: 4, h: 4 } };
    const layout = widgetLayoutFromPlacements(plan, previous);
    // Every shown card pinned at its cell first, then the hidden card's slot kept.
    assert.deepEqual(Object.keys(layout), ['a', 'b', 'c', 'gone']);
    assert.deepEqual(layout.a, { x: 6, y: 4, w: 4, h: 4 });
    assert.deepEqual(layout.gone, previous.gone);
    const crowded = Object.fromEntries(Array.from({ length: 250 }, (_, i) => [`old:${i}`, { x: 0, y: i, w: 4, h: 4 }]));
    const bounded = widgetLayoutFromPlacements(plan, crowded);
    assert.equal(Object.keys(bounded).length, WIDGET_LAYOUT_MAX_ITEMS);
    assert.deepEqual(Object.keys(bounded).slice(0, 3), ['a', 'b', 'c'], 'shown cards survive the bound');
});

test('a reload restores the same cells from the stored JSON', () => {
    const cards = [card('a'), card('b', 8), card('c')];
    const arranged = arrangeWidgetSlot(planWidgetGrid(cards), 'c', { x: 2, y: 3, w: 6, h: 10 });
    const stored = JSON.parse(JSON.stringify({ widget_layout: widgetLayoutFromPlacements(arranged, {}) }));
    const revisit = planWidgetGrid(cards, normalizeWidgetLayout(stored.widget_layout));
    assert.deepEqual(slots(revisit), slots(arranged));
    // …and again with the cards arriving in another key order.
    assert.deepEqual(slots(planWidgetGrid(cards.slice().reverse(), stored.widget_layout)), slots(arranged));
});

test('the narrow fallback switches by list width with a band against scrollbar flapping', () => {
    assert.equal(widgetGridMode(1200), 'grid');
    assert.equal(widgetGridMode(500), 'stack');
    assert.equal(widgetGridMode(0, 'stack'), 'stack', 'a hidden page keeps its mode');
    assert.equal(widgetGridMode(0), 'grid');
    // Inside the band, the previous mode holds.
    assert.equal(widgetGridMode(715, 'grid'), 'grid');
    assert.equal(widgetGridMode(715, 'stack'), 'stack');
    assert.equal(widgetGridMode(725, 'stack'), 'stack');
    assert.equal(widgetGridMode(707, 'grid'), 'stack');
    assert.equal(widgetGridMode(732, 'stack'), 'grid');
});

// --- DOM writer ------------------------------------------------------------

function fakeCard(key, { removed = false } = {}) {
    const props = new Map();
    const writes = [];
    const node = {
        dataset: { widgetKey: key },
        props,
        writes,
        style: {
            getPropertyValue: (name) => props.get(name) || '',
            setProperty: (name, value) => { writes.push(name); props.set(name, value); },
            removeProperty: () => { throw new Error('the grid never removes a placement'); },
        },
        hasAttribute: (name) => removed && name === 'data-widget-removed',
        get offsetHeight() { throw new Error('the grid must never measure a card'); },
        getBoundingClientRect() { throw new Error('the grid must never measure a card'); },
    };
    for (const name of ['before', 'after', 'append', 'prepend', 'remove', 'replaceWith', 'insertBefore', 'appendChild']) {
        node[name] = () => { throw new Error(`the grid must not call ${name}`); };
    }
    return node;
}

function fakeList(cards, width = 1200) {
    return {
        cards,
        clientWidth: width,
        dataset: {},
        querySelectorAll(selector) {
            assert.equal(selector, '[data-widget-key]');
            return this.cards.slice();
        },
    };
}

function installObserver() {
    const observers = [];
    const frames = new Map();
    let nextFrame = 1;
    globalThis.requestAnimationFrame = (callback) => {
        frames.set(nextFrame, callback);
        return nextFrame++;
    };
    globalThis.cancelAnimationFrame = (id) => frames.delete(id);
    observers.flush = () => {
        const callbacks = [...frames.values()];
        frames.clear();
        callbacks.forEach((callback) => callback());
    };
    observers.pending = () => frames.size;
    globalThis.ResizeObserver = class {
        constructor(callback) {
            this.callback = callback;
            this.disconnected = false;
            observers.push(this);
        }
        observe(target) { this.target = target; }
        disconnect() { this.disconnected = true; }
    };
    return observers;
}

test('applyWidgetGrid writes cells only as custom properties and never measures or moves a card', () => {
    installObserver();
    const a = fakeCard('demo:a');
    const b = fakeCard('demo:b');
    const retiring = fakeCard('demo:gone', { removed: true });
    const list = fakeList([b, retiring, a]);
    const placements = planWidgetGrid([card('demo:a', 8), card('demo:b')]);
    const dispose = applyWidgetGrid(list, { placements, order: ['demo:a', 'demo:b'] });
    assert.deepEqual(Object.fromEntries(a.props), {
        '--widget-col': '1', '--widget-row': '1', '--widget-w': '8', '--widget-h': '8', '--widget-order': '0',
    });
    assert.deepEqual(Object.fromEntries(b.props), {
        '--widget-col': '9', '--widget-row': '1', '--widget-w': '4', '--widget-h': '8', '--widget-order': '1',
    });
    assert.equal(retiring.props.size, 0, 'a card whose frame is still stopping keeps the cell it had');
    assert.equal(list.dataset.widgetLayout, 'grid');
    // Unchanged placement: no property is written again.
    const before = a.writes.length + b.writes.length;
    applyWidgetGrid(list, { placements, order: ['demo:a', 'demo:b'] });
    assert.equal(a.writes.length + b.writes.length, before);
    dispose();
});

test('the list width alone switches grid and stack; one observer per list, one idempotent disposer', () => {
    const observers = installObserver();
    const a = fakeCard('demo:a');
    const list = fakeList([a], 1200);
    const placements = planWidgetGrid([card('demo:a')]);
    const dispose = applyWidgetGrid(list, { placements, order: ['demo:a'] });
    assert.equal(applyWidgetGrid(list, { placements, order: ['demo:a'] }), dispose);
    assert.equal(observers.length, 1);
    assert.equal(observers[0].target, list);
    list.clientWidth = 480;
    observers[0].callback();
    observers[0].callback();
    assert.equal(list.dataset.widgetLayout, 'grid', 'the switch waits one frame: never inside the observer callback');
    assert.equal(observers.pending(), 1, 'triggers before the frame coalesce');
    observers.flush();
    assert.equal(list.dataset.widgetLayout, 'stack');
    // The card's cell is untouched by the switch: the stack reads it from CSS.
    assert.equal(a.props.get('--widget-col'), '1');
    list.clientWidth = 0;
    observers[0].callback();
    observers.flush();
    assert.equal(list.dataset.widgetLayout, 'stack', 'a hidden list keeps its mode');
    list.clientWidth = 1000;
    observers[0].callback();
    observers.flush();
    assert.equal(list.dataset.widgetLayout, 'grid');
    observers[0].callback();
    dispose();
    dispose();
    assert.equal(observers[0].disconnected, true);
    assert.equal(observers.pending(), 0, 'the disposer cancels a pending switch');
    // Forgotten: the next call binds afresh.
    const next = applyWidgetGrid(list, { placements, order: ['demo:a'] });
    assert.notEqual(next, dispose);
    assert.equal(observers.length, 2);
    next();
    assert.equal(typeof applyWidgetGrid(null), 'function');
});

test('the grid modules measure nothing and have no node insertion or move API', () => {
    for (const file of ['widget_grid.js', 'widget_reorder.js']) {
        const source = readFileSync(new URL(`../modules/${file}`, import.meta.url), 'utf8');
        for (const forbidden of ['.before(', '.after(', '.prepend(', '.append(', 'insertBefore', 'appendChild', 'replaceWith', '.remove()', 'offsetHeight', 'scrollHeight', "from './masonry.js'"]) {
            assert.equal(source.includes(forbidden), false, `${file} must not use ${forbidden}`);
        }
    }
    const grid = readFileSync(new URL('../modules/widget_grid.js', import.meta.url), 'utf8');
    assert.equal(grid.includes('getBoundingClientRect'), false, 'placement never measures');
});
