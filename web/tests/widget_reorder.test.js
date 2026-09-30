import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    createWidgetArrangement,
    mergeVisibleWidgetOrder,
    moveWidgetKey,
    normalizeWidgetOrder,
    sortTabsByWidgetOrder,
} from '../modules/widget_reorder.js';

// Widgets arrangement: a stacked reorder is a pure move in the KEY order, a
// grid move or resize a pure change of cells; the handles never move an
// <article> (a moved <iframe> reloads) and write only custom properties.

test('moveWidgetKey moves a key to a clamped index and returns the same array when nothing changes', () => {
    const order = ['a', 'b', 'c', 'd'];
    assert.deepEqual(moveWidgetKey(order, 'b', 2), ['a', 'c', 'b', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'b', 0), ['b', 'a', 'c', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'c', 0), ['c', 'a', 'b', 'd']);
    assert.deepEqual(moveWidgetKey(order, 'a', Number.MAX_SAFE_INTEGER), ['b', 'c', 'd', 'a']);
    assert.deepEqual(moveWidgetKey(order, 'd', -5), ['d', 'a', 'b', 'c']);
    // Identity means "not moved": same slot, first card moved up, unknown key.
    assert.equal(moveWidgetKey(order, 'b', 1), order);
    assert.equal(moveWidgetKey(order, 'a', -1), order);
    assert.equal(moveWidgetKey(order, 'zzz', 0), order);
    assert.equal(moveWidgetKey([], 'a', 0).length, 0);
    // The input is never mutated.
    assert.deepEqual(order, ['a', 'b', 'c', 'd']);
});

test('a drop onto a target lands after a target the key was before, before a target it was after', () => {
    const order = ['a', 'b', 'c', 'd'];
    // Drag a onto c: a was before c → a lands after c.
    assert.deepEqual(moveWidgetKey(order, 'a', order.indexOf('c')), ['b', 'c', 'a', 'd']);
    // Drag d onto b: d was after b → d lands before b.
    assert.deepEqual(moveWidgetKey(order, 'd', order.indexOf('b')), ['a', 'd', 'b', 'c']);
});

test('normalizeWidgetOrder and sortTabsByWidgetOrder keep the phase-2 contract', () => {
    assert.deepEqual(normalizeWidgetOrder([' a ', '', 'b', 'a', null]), ['a', 'b']);
    assert.deepEqual(normalizeWidgetOrder('nope'), []);
    const tabs = [{ key: 'x' }, { key: 'y' }, { key: 'z' }];
    assert.deepEqual(sortTabsByWidgetOrder(tabs, ['z']).map((tab) => tab.key), ['z', 'x', 'y']);
});

test('reordering visible cards preserves disabled keys in the owner order', () => {
    assert.deepEqual(mergeVisibleWidgetOrder(['a', 'disabled', 'b'], ['b', 'a']), ['b', 'disabled', 'a']);
    assert.deepEqual(mergeVisibleWidgetOrder([], ['b', 'a']), ['b', 'a']);
    assert.deepEqual(mergeVisibleWidgetOrder(['a', 'gone', 'b'], ['b', 'new', 'a']), ['b', 'gone', 'new', 'a']);
});

// --- Arrangement controller over a fake list -------------------------------

const DOM_MOVES = ['before', 'after', 'append', 'prepend', 'remove', 'replaceWith', 'insertBefore', 'appendChild'];

function fakeTarget() {
    const listeners = new Map();
    return {
        addEventListener(type, fn) {
            if (!listeners.has(type)) listeners.set(type, new Set());
            listeners.get(type).add(fn);
        },
        removeEventListener(type, fn) { listeners.get(type)?.delete(fn); },
        dispatch(type, event = {}) {
            const full = {
                currentTarget: this,
                defaultPrevented: false,
                preventDefault() { this.defaultPrevented = true; },
                stopPropagation() {},
                ...event,
            };
            [...(listeners.get(type) || [])].forEach((fn) => fn(full));
            return full;
        },
        count(type) { return listeners.get(type)?.size || 0; },
    };
}

function fakeClassList() {
    const names = new Set();
    return { add: (name) => names.add(name), remove: (name) => names.delete(name), contains: (name) => names.has(name) };
}

function fakeCard(key) {
    const props = new Map();
    const card = {
        ...fakeTarget(),
        dataset: { widgetKey: key },
        isConnected: true,
        props,
        classList: fakeClassList(),
        rect: { top: 0, height: 100 },
        scrolled: 0,
        frame: { id: `frame:${key}` },
        style: {
            getPropertyValue: (name) => props.get(name) || '',
            setProperty: (name, value) => props.set(name, value),
        },
        hasAttribute: () => false,
        getBoundingClientRect() { return this.rect; },
        scrollIntoView() { this.scrolled += 1; },
    };
    for (const name of DOM_MOVES) card[name] = () => { throw new Error(`arrangement must not call card.${name}`); };
    const handle = () => {
        const node = { ...fakeTarget(), captured: null };
        node.closest = (selector) => (selector === '[data-widget-key]' ? card : null);
        node.setPointerCapture = (id) => { node.captured = id; };
        node.hasPointerCapture = (id) => node.captured === id;
        node.releasePointerCapture = (id) => {
            node.captured = null;
            node.dispatch('lostpointercapture', { pointerId: id });
        };
        return node;
    };
    card.move = handle();
    card.resize = handle();
    return card;
}

function fakeList(cards, width = 1200) {
    const doc = fakeTarget();
    return {
        cards,
        clientWidth: width,
        dataset: {},
        classList: fakeClassList(),
        ownerDocument: doc,
        querySelectorAll(selector) {
            if (selector === '[data-widget-key]') return this.cards.slice();
            if (selector === '[data-widget-move-handle]') return this.cards.map((node) => node.move);
            if (selector === '[data-widget-resize-handle]') return this.cards.map((node) => node.resize);
            throw new Error(`unexpected selector ${selector}`);
        },
        getBoundingClientRect: () => ({ left: 0, top: 0 }),
        closest: () => null,
    };
}

function installGlobals() {
    const observers = [];
    globalThis.ResizeObserver = class {
        constructor(callback) { this.callback = callback; observers.push(this); }
        observe() {}
        disconnect() {}
    };
    globalThis.getComputedStyle = () => ({ columnGap: '14px', rowGap: '14px', gridAutoRows: '40px' });
    globalThis.requestAnimationFrame = () => 1;
    globalThis.cancelAnimationFrame = () => {};
    return observers;
}

const moduleTab = (key, extra = {}) => ({ key, render: { kind: 'module', entry: 'w.js' }, ...extra });

// A page double: the shown tabs in key order, its preferences, and every POST.
function harness({ tabs, layout = {}, width = 1200, holdSaves = false, save = null, canEdit = true } = {}) {
    installGlobals();
    const cards = tabs.map((tab) => fakeCard(tab.key));
    const list = fakeList(cards, width);
    const state = { tabs: tabs.slice(), prefs: { widget_order: tabs.map((tab) => tab.key), widget_layout: layout } };
    const saves = [];
    const pending = [];
    const status = { textContent: '', dataset: {} };
    const arrangement = createWidgetArrangement(list, {
        tabs: () => state.tabs,
        prefs: () => state.prefs,
        canEdit: () => canEdit,
        commit(next) {
            state.prefs = { ...state.prefs, ...next };
            if (next.widget_order) state.tabs = sortTabsByWidgetOrder(state.tabs, next.widget_order);
        },
        save(payload) {
            saves.push(JSON.parse(JSON.stringify(payload)));
            if (save) return save(payload);
            if (!holdSaves) return Promise.resolve({ ok: true });
            return new Promise((resolve) => pending.push(resolve));
        },
        status,
    });
    arrangement.bind();
    arrangement.relayout();
    const cell = (key) => {
        const node = cards.find((item) => item.dataset.widgetKey === key);
        return ['--widget-col', '--widget-row', '--widget-w', '--widget-h'].map((name) => Number(node.props.get(name)));
    };
    const byKey = (key) => cards.find((item) => item.dataset.widgetKey === key);
    return { arrangement, cards, list, state, saves, pending, status, cell, byKey };
}

const flushMicrotasks = () => new Promise((resolve) => setTimeout(resolve, 0));

test('a new list is placed without any write; binding twice adds no second listener', () => {
    const page = harness({ tabs: [moduleTab('demo:a', { span: 2 }), moduleTab('demo:b'), moduleTab('demo:c')] });
    assert.deepEqual(page.cell('demo:a'), [1, 1, 8, 8]);
    assert.deepEqual(page.cell('demo:b'), [9, 1, 4, 8]);
    assert.deepEqual(page.cell('demo:c'), [1, 9, 4, 8]);
    assert.equal(page.saves.length, 0, 'showing cards never writes the preference');
    page.arrangement.bind();
    assert.equal(page.byKey('demo:a').move.count('keydown'), 1);
    assert.equal(page.byKey('demo:a').resize.count('pointerdown'), 1);
});

test('an unavailable preferences read cannot overwrite hidden saved cells', () => {
    const page = harness({ tabs: [moduleTab('visible')], canEdit: false });
    page.arrangement.pinDefaults();
    page.byKey('visible').move.dispatch('keydown', { key: 'ArrowDown' });
    assert.deepEqual(page.saves, []);
    assert.deepEqual(page.state.prefs.widget_layout, {});
});

test('first successful preferences read pins defaults; removing a sibling keeps saved cells', async () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b'), moduleTab('demo:c')] });
    page.arrangement.pinDefaults();
    assert.equal(page.saves.length, 1);
    assert.deepEqual(Object.keys(page.saves[0].widget_layout), ['demo:a', 'demo:b', 'demo:c']);
    const original = page.arrangement.relayout;
    page.state.tabs = page.state.tabs.filter((tab) => tab.key !== 'demo:a');
    original();
    assert.deepEqual(page.cell('demo:b'), [5, 1, 4, 8]);
    assert.deepEqual(page.cell('demo:c'), [9, 1, 4, 8]);
    page.arrangement.pinDefaults();
    assert.equal(page.saves.length, 1, 'an unchanged list is not saved again');
    await flushMicrotasks();
});

test('grid keys move a card one cell, pin every shown card and re-derive the reading order', async () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b'), moduleTab('demo:c')] });
    const event = page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowDown' });
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(page.cell('demo:a'), [1, 2, 4, 8]);
    assert.deepEqual(page.saves, [{
        widget_layout: {
            'demo:a': { x: 0, y: 1, w: 4, h: 8 },
            'demo:b': { x: 4, y: 0, w: 4, h: 8 },
            'demo:c': { x: 8, y: 0, w: 4, h: 8 },
        },
        widget_order: ['demo:b', 'demo:c', 'demo:a'],
    }]);
    assert.equal(page.status.textContent, 'Moved to column 1, row 2');
    assert.equal(page.byKey('demo:a').scrolled, 1, 'the moved card is kept in view');
    await flushMicrotasks();
    // Home: top-left, pushing what is there straight down; End: below every other card.
    page.byKey('demo:c').move.dispatch('keydown', { key: 'Home' });
    assert.deepEqual(page.cell('demo:c'), [1, 1, 4, 8]);
    assert.deepEqual(page.cell('demo:a'), [1, 9, 4, 8]);
    assert.deepEqual(page.state.prefs.widget_order, ['demo:c', 'demo:b', 'demo:a']);
    await flushMicrotasks();
    page.byKey('demo:c').move.dispatch('keydown', { key: 'End' });
    assert.deepEqual(page.cell('demo:c'), [1, 17, 4, 8]);
    assert.equal(page.state.prefs.widget_order.at(-1), 'demo:c');
});

test('a key that changes nothing writes nothing and leaves the key to the page', () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b')] });
    for (const key of ['ArrowUp', 'ArrowLeft', 'Home', 'Tab', 'Enter']) {
        const event = page.byKey('demo:a').move.dispatch('keydown', { key });
        assert.equal(event.defaultPrevented, false, key);
    }
    assert.equal(page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowRight', altKey: true }).defaultPrevented, false);
    assert.equal(page.saves.length, 0);
});

test('the corner handle resizes by one column or row and pushes the card below', () => {
    const page = harness({
        tabs: [moduleTab('demo:a'), moduleTab('demo:b')],
        layout: { 'demo:a': { x: 0, y: 0, w: 4, h: 4 }, 'demo:b': { x: 0, y: 4, w: 4, h: 4 } },
    });
    page.byKey('demo:a').resize.dispatch('keydown', { key: 'ArrowRight' });
    page.byKey('demo:a').resize.dispatch('keydown', { key: 'ArrowDown' });
    assert.deepEqual(page.cell('demo:a'), [1, 1, 5, 5]);
    assert.deepEqual(page.cell('demo:b'), [1, 6, 4, 4]);
    assert.equal(page.status.textContent, 'Resized to 5 columns by 5 rows');
    assert.deepEqual(page.state.prefs.widget_layout['demo:a'], { x: 0, y: 0, w: 5, h: 5 });
    // At the minimum the key is not consumed.
    assert.equal(page.byKey('demo:b').resize.dispatch('keydown', { key: 'ArrowUp' }).defaultPrevented, false);
});

test('the narrow stack reorders the key order and resizes the height only', async () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b'), moduleTab('demo:c')], width: 480 });
    assert.equal(page.list.dataset.widgetLayout, 'stack');
    page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowDown' });
    assert.deepEqual(page.saves.at(-1), { widget_order: ['demo:b', 'demo:a', 'demo:c'] });
    assert.deepEqual(['demo:a', 'demo:b', 'demo:c'].map((key) => page.byKey(key).props.get('--widget-order')), ['1', '0', '2']);
    assert.equal(page.status.textContent, 'Moved to position 2 of 3');
    await flushMicrotasks();
    page.byKey('demo:c').move.dispatch('keydown', { key: 'Home' });
    assert.deepEqual(page.state.prefs.widget_order, ['demo:c', 'demo:b', 'demo:a']);
    await flushMicrotasks();
    assert.equal(page.byKey('demo:b').resize.dispatch('keydown', { key: 'ArrowRight' }).defaultPrevented, false);
    page.byKey('demo:b').resize.dispatch('keydown', { key: 'ArrowDown' });
    const saved = page.saves.at(-1);
    assert.deepEqual(Object.keys(saved), ['widget_layout'], 'a stacked resize keeps the owner\'s stacked order');
    assert.equal(saved.widget_layout['demo:b'].h, 9);
    assert.equal(page.status.textContent, 'Resized to 9 rows');
});

test('a pointer drag previews cell by cell, pushes what is in the way, and commits once on release', () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b'), moduleTab('demo:c')] });
    const a = page.byKey('demo:a');
    const frame = a.frame;
    a.move.dispatch('pointerdown', { button: 0, pointerId: 7, clientX: 10, clientY: 10 });
    assert.equal(page.list.classList.contains('arranging'), true, 'frames ignore the pointer while arranging');
    assert.equal(a.classList.contains('arranging'), true);
    assert.equal(a.move.captured, 7);
    // (1200 + 14) / 12 px per column, 54 px per row: four columns right, two rows down.
    a.move.dispatch('pointermove', { pointerId: 7, clientX: 10 + 405, clientY: 10 + 108 });
    assert.deepEqual(page.cell('demo:a'), [5, 3, 4, 8]);
    assert.deepEqual(page.cell('demo:b'), [5, 11, 4, 8], 'b is pushed below a');
    // A pointer that is not the dragging one changes nothing.
    a.move.dispatch('pointermove', { pointerId: 9, clientX: 900, clientY: 900 });
    assert.deepEqual(page.cell('demo:a'), [5, 3, 4, 8]);
    assert.equal(page.saves.length, 0, 'the preview is not written');
    a.move.dispatch('pointerup', { pointerId: 7, clientX: 10 + 405, clientY: 10 + 108 });
    assert.equal(page.list.classList.contains('arranging'), false);
    assert.equal(a.move.captured, null);
    assert.equal(page.saves.length, 1);
    assert.deepEqual(page.saves[0].widget_layout['demo:a'], { x: 4, y: 2, w: 4, h: 8 });
    assert.deepEqual(page.saves[0].widget_order, ['demo:c', 'demo:a', 'demo:b']);
    assert.equal(a.frame, frame, 'the frame object in the card is the same one');
});

test('a drag that returns to its cell, Escape, or a lost capture restores the arrangement and writes nothing', () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b')] });
    const a = page.byKey('demo:a');
    a.move.dispatch('pointerdown', { button: 0, pointerId: 1, clientX: 0, clientY: 0 });
    a.move.dispatch('pointermove', { pointerId: 1, clientX: 400, clientY: 0 });
    a.move.dispatch('pointermove', { pointerId: 1, clientX: 20, clientY: 10 });
    a.move.dispatch('pointerup', { pointerId: 1 });
    assert.equal(page.saves.length, 0);

    a.move.dispatch('pointerdown', { button: 0, pointerId: 2, clientX: 0, clientY: 0 });
    a.move.dispatch('pointermove', { pointerId: 2, clientX: 0, clientY: 500 });
    assert.notDeepEqual(page.cell('demo:a'), [1, 1, 4, 8]);
    const escape = page.list.ownerDocument.dispatch('keydown', { key: 'Escape' });
    assert.equal(escape.defaultPrevented, true);
    assert.deepEqual(page.cell('demo:a'), [1, 1, 4, 8]);
    assert.equal(page.list.ownerDocument.count('keydown'), 0, 'the drag listener goes with the drag');

    a.resize.dispatch('pointerdown', { button: 0, pointerId: 3, clientX: 0, clientY: 0 });
    a.resize.dispatch('pointermove', { pointerId: 3, clientX: 300, clientY: 300 });
    a.resize.dispatch('lostpointercapture', { pointerId: 3 });
    assert.deepEqual(page.cell('demo:a'), [1, 1, 4, 8]);
    // A secondary button never starts a drag.
    a.move.dispatch('pointerdown', { button: 2, pointerId: 4, clientX: 0, clientY: 0 });
    assert.equal(page.list.classList.contains('arranging'), false);
    assert.equal(page.saves.length, 0);
});

test('a pointer resize is bounded by the grid and commits on release', () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b')] });
    const b = page.byKey('demo:b');
    b.resize.dispatch('pointerdown', { button: 0, pointerId: 1, clientX: 500, clientY: 400 });
    b.resize.dispatch('pointermove', { pointerId: 1, clientX: 5000, clientY: 400 + 54 * 3 });
    assert.deepEqual(page.cell('demo:b'), [5, 1, 8, 11], 'the width stops at the right edge');
    b.resize.dispatch('pointerup', { pointerId: 1 });
    assert.deepEqual(page.saves.at(-1).widget_layout['demo:b'], { x: 4, y: 0, w: 8, h: 11 });
});

test('a stacked pointer drag reorders by the other cards\' midpoints', () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b'), moduleTab('demo:c')], width: 480 });
    page.byKey('demo:a').rect = { top: 0, height: 400 };
    page.byKey('demo:b').rect = { top: 414, height: 200 };
    page.byKey('demo:c').rect = { top: 628, height: 200 };
    const a = page.byKey('demo:a');
    a.move.dispatch('pointerdown', { button: 0, pointerId: 1, clientX: 10, clientY: 10 });
    a.move.dispatch('pointermove', { pointerId: 1, clientX: 10, clientY: 600 });
    assert.deepEqual(['demo:a', 'demo:b', 'demo:c'].map((key) => page.byKey(key).props.get('--widget-order')), ['1', '0', '2']);
    a.move.dispatch('pointerup', { pointerId: 1 });
    assert.deepEqual(page.saves, [{ widget_order: ['demo:b', 'demo:a', 'demo:c'] }]);
});

test('writes go one at a time, the last arrangement wins, and a read begun before it is stale', async () => {
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b')], holdSaves: true });
    const since = page.arrangement.revision();
    assert.equal(page.arrangement.settled(since), true);
    page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowDown' });
    await flushMicrotasks();
    page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowDown' });
    page.byKey('demo:a').move.dispatch('keydown', { key: 'ArrowDown' });
    await flushMicrotasks();
    assert.equal(page.saves.length, 1, 'the second write waits for the first');
    assert.equal(page.arrangement.settled(since), false);
    page.pending.shift()({ ok: true });
    await flushMicrotasks();
    assert.equal(page.saves.length, 2, 'the queued changes go as one write');
    assert.deepEqual(page.saves[1].widget_layout['demo:a'], { x: 0, y: 3, w: 4, h: 8 });
    page.pending.shift()({ ok: true });
    await flushMicrotasks();
    assert.equal(page.arrangement.settled(since), false, 'a read begun before the arrangement stays stale');
    assert.equal(page.arrangement.settled(page.arrangement.revision()), true);
});

test('a failed write is reported and does not block the next one', async () => {
    const outcomes = [Promise.reject(new Error('offline')), Promise.resolve({ ok: true })];
    const page = harness({ tabs: [moduleTab('demo:a'), moduleTab('demo:b')], save: () => outcomes.shift() });
    const warnings = [];
    const warn = console.warn;
    console.warn = (...args) => warnings.push(args);
    try {
        page.byKey('demo:b').move.dispatch('keydown', { key: 'ArrowDown' });
        await flushMicrotasks();
        page.byKey('demo:b').move.dispatch('keydown', { key: 'ArrowDown' });
        await flushMicrotasks();
    } finally {
        console.warn = warn;
    }
    assert.equal(page.saves.length, 2);
    assert.equal(warnings.length, 1);
    assert.deepEqual(page.state.prefs.widget_layout['demo:b'], { x: 4, y: 2, w: 4, h: 8 }, 'the page keeps the owner\'s arrangement');
    assert.equal(page.arrangement.settled(page.arrangement.revision()), true);
});

test('the arrangement module moves no node and keeps the static contract', () => {
    const source = readFileSync(new URL('../modules/widget_reorder.js', import.meta.url), 'utf8');
    for (const forbidden of ['.before(', '.after(', '.prepend(', '.append(', 'insertBefore', 'appendChild', 'replaceWith', 'draggable', 'dataTransfer', '.style.transform', '.style.left', '.style.top']) {
        assert.equal(source.includes(forbidden), false, `widget_reorder.js must not use ${forbidden}`);
    }
    assert.match(source, /export function createWidgetArrangement\(list, options\)/);
});
