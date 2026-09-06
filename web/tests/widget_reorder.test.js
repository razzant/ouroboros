import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { moveWidgetKey, normalizeWidgetOrder, sortTabsByWidgetOrder } from '../modules/widget_reorder.js';

// Widgets lifecycle phase 3: a reorder is a pure move in the KEY order; the
// handles never move an <article> (a moved <iframe> reloads).

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

test('the reorder module moves keys only: no node insertion or move API, no masonry import', () => {
    const source = readFileSync(new URL('../modules/widget_reorder.js', import.meta.url), 'utf8');
    for (const forbidden of ['.before(', '.after(', '.prepend(', '.append(', 'insertBefore', 'appendChild', 'replaceWith', "from './masonry.js'"]) {
        assert.equal(source.includes(forbidden), false, `widget_reorder.js must not use ${forbidden}`);
    }
    assert.match(source, /export function bindWidgetCardReorder\(list, currentOrder, onOrderChange\)/);
});
