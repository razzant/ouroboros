import test from 'node:test';
import assert from 'node:assert/strict';

import { applyMasonry, planMasonryLayout } from '../modules/masonry.js';

// Widgets lifecycle phase 3: masonry packs the cards in the caller's KEY order
// and writes the plan back only as custom properties — no generated <style>,
// no DOM moves, one idempotent disposer per container.

function fakeItem(key, { height = 100, span = 1 } = {}) {
    const props = new Map();
    return {
        dataset: { widgetKey: key },
        offsetHeight: height,
        classList: { contains: (name) => span === 2 && name === 'widgets-card-span-2' },
        style: {
            setProperty: (name, value) => props.set(name, value),
            removeProperty: (name) => props.delete(name),
        },
        props,
    };
}

function fakeContainer(items, width = 600) {
    const props = new Map();
    return {
        items,
        clientWidth: width,
        dataset: {},
        querySelectorAll(selector) {
            assert.equal(selector, '.widgets-card');
            return this.items.slice();
        },
        contains(item) { return this.items.includes(item); },
        style: {
            setProperty: (name, value) => props.set(name, value),
            removeProperty: (name) => props.delete(name),
        },
        props,
    };
}

// Browser globals the module reaches for at call time (never at import time).
function installGlobals() {
    const frames = new Map();
    let nextFrame = 1;
    const counters = { resizeObservers: 0, mutationObservers: 0, disconnects: 0, cancelled: 0 };
    class FakeResizeObserver {
        constructor() { counters.resizeObservers += 1; }
        observe() {}
        unobserve() {}
        disconnect() { counters.disconnects += 1; }
    }
    class FakeMutationObserver {
        constructor() { counters.mutationObservers += 1; }
        observe() {}
        disconnect() { counters.disconnects += 1; }
    }
    globalThis.ResizeObserver = FakeResizeObserver;
    globalThis.MutationObserver = FakeMutationObserver;
    globalThis.requestAnimationFrame = (callback) => {
        frames.set(nextFrame, callback);
        return nextFrame++;
    };
    globalThis.cancelAnimationFrame = (id) => {
        if (frames.delete(id)) counters.cancelled += 1;
    };
    // Any reach for the document is the old generated-stylesheet shape.
    globalThis.document = {
        head: { appendChild() { throw new Error('masonry must not append a <style> element'); } },
        createElement() { throw new Error('masonry must not create elements'); },
        getElementById() { throw new Error('masonry must not look up a style element'); },
    };
    const flush = () => {
        const callbacks = Array.from(frames.values());
        frames.clear();
        callbacks.forEach((callback) => callback());
    };
    return { counters, flush, pending: () => frames.size };
}

test('layout follows the key order, not the DOM order, and writes only custom properties', () => {
    const { flush } = installGlobals();
    const a = fakeItem('demo:a');
    const b = fakeItem('demo:b');
    const c = fakeItem('demo:c');
    const container = fakeContainer([a, b, c]);
    const dispose = applyMasonry(container, { order: ['demo:c', 'demo:a', 'demo:b'] });
    assert.equal(typeof dispose, 'function');
    flush();
    // 600px, 280px min column, 14px gap → two 293px columns. Sorted: c, a, b.
    assert.deepEqual(Object.fromEntries(c.props), { '--masonry-w': '293px', '--masonry-x': '0px', '--masonry-y': '0px' });
    assert.deepEqual(Object.fromEntries(a.props), { '--masonry-w': '293px', '--masonry-x': '307px', '--masonry-y': '0px' });
    assert.deepEqual(Object.fromEntries(b.props), { '--masonry-w': '293px', '--masonry-x': '0px', '--masonry-y': '114px' });
    assert.deepEqual(Object.fromEntries(container.props), { '--masonry-h': '214px' });
    // Nothing else was written: no generated id on the container, no attribute.
    assert.deepEqual(container.dataset, {});
    dispose();
});

test('keys the order does not name keep their DOM order after the named ones', () => {
    const { flush } = installGlobals();
    const a = fakeItem('demo:a');
    const b = fakeItem('demo:b');
    const c = fakeItem('demo:c');
    const container = fakeContainer([a, b, c], 320);
    const dispose = applyMasonry(container, { order: ['demo:c'] });
    flush();
    // One column: c first (named), then a, b in DOM order.
    assert.equal(c.props.get('--masonry-y'), '0px');
    assert.equal(a.props.get('--masonry-y'), '114px');
    assert.equal(b.props.get('--masonry-y'), '228px');
    dispose();
});

test('a later call with a new order relayouts in place without rebinding; an empty list clears the height', () => {
    const { counters, flush } = installGlobals();
    const a = fakeItem('demo:a');
    const b = fakeItem('demo:b');
    const container = fakeContainer([a, b], 320);
    const dispose = applyMasonry(container, { order: ['demo:a', 'demo:b'] });
    flush();
    assert.equal(a.props.get('--masonry-y'), '0px');
    assert.equal(b.props.get('--masonry-y'), '114px');
    const bound = { ...counters };
    const again = applyMasonry(container, { order: ['demo:b', 'demo:a'] });
    assert.equal(again, dispose, 'every call returns the one disposer of the container');
    flush();
    assert.equal(b.props.get('--masonry-y'), '0px');
    assert.equal(a.props.get('--masonry-y'), '114px');
    assert.deepEqual({ ...counters }, bound, 'no observer is created twice for one container');
    container.items = [];
    applyMasonry(container);
    flush();
    assert.equal(container.props.has('--masonry-h'), false);
    dispose();
});

test('triggers before a frame coalesce into one layout; the disposer cancels the pending frame', () => {
    const { counters, flush, pending } = installGlobals();
    let layouts = 0;
    const item = fakeItem('demo:a');
    item.style.setProperty = (name) => { if (name === '--masonry-y') layouts += 1; };
    const container = fakeContainer([item], 320);
    const dispose = applyMasonry(container, { order: ['demo:a'] });
    applyMasonry(container);
    applyMasonry(container);
    assert.equal(pending(), 1);
    flush();
    assert.equal(layouts, 1);
    applyMasonry(container);
    assert.equal(pending(), 1);
    dispose();
    assert.equal(pending(), 0);
    assert.equal(counters.cancelled >= 1, true);
    flush();
    assert.equal(layouts, 1, 'a cancelled frame never lays out');
});

test('the disposer disconnects both ResizeObservers and the MutationObserver, once, and forgets the container', () => {
    const { counters } = installGlobals();
    const container = fakeContainer([fakeItem('demo:a')], 320);
    const dispose = applyMasonry(container, { order: ['demo:a'] });
    assert.equal(counters.resizeObservers, 2);
    assert.equal(counters.mutationObservers, 1);
    dispose();
    assert.equal(counters.disconnects, 3);
    dispose();
    assert.equal(counters.disconnects, 3, 'idempotent');
    // Forgotten: the next call binds afresh (new observers, a new disposer).
    const next = applyMasonry(container, { order: ['demo:a'] });
    assert.notEqual(next, dispose);
    assert.equal(counters.resizeObservers, 4);
    assert.equal(counters.mutationObservers, 2);
    next();
    assert.equal(counters.disconnects, 6);
});

test('applyMasonry without a container is a no-op that still returns a disposer', () => {
    installGlobals();
    const dispose = applyMasonry(null);
    assert.equal(typeof dispose, 'function');
    dispose();
});

test('planMasonryLayout is unchanged: columns from width, wide cards span two tracks', () => {
    const plan = planMasonryLayout(600, [{ span: 1, height: 100 }, { span: 2, height: 50 }, { span: 1, height: 100 }]);
    assert.equal(plan.columnCount, 2);
    assert.equal(plan.columnWidth, 293);
    assert.deepEqual(plan.placements.map((placement) => [placement.span, placement.left, placement.top]), [
        [1, 0, 0], [2, 0, 114], [1, 0, 178],
    ]);
});
