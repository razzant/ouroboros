import test from 'node:test';
import assert from 'node:assert/strict';

import { applyMasonry, planMasonryLayout } from '../modules/masonry.js';

// docs/DESIGN.md "Widgets board": the owner's column span of a card (`spans` by
// key, the `owner` of a planned item) replaces the author's span class. A board
// without owner spans is planned exactly as before (web/tests/masonry.test.js is
// the target's file, unchanged).

const placed = (plan) => plan.placements.map(({ span, left, top, width }) => [span, left, top, width]);

// The seven real widgets in the owner's order: author span, and the height and
// geometry measured on the owner-approved masonry at three window widths
// (list width 952 / 1112 / 1400 px).
const REAL = [
    // key, span, {listWidth: [height, x, y, width]}
    ['claudexor_quotas:quotas', 2, { 952: [543, 0, 0, 630], 1112: [553, 0, 0, 548], 1400: [535, 0, 0, 550] }],
    ['cache_efficiency_snapshot:cache_efficiency', 2, { 952: [1152, 0, 557, 630], 1112: [1319, 562, 0, 548], 1400: [1319, 564, 0, 550] }],
    ['context-lens:lens', 2, { 952: [834, 0, 1723, 630], 1112: [834, 0, 567, 548], 1400: [834, 0, 549, 550] }],
    ['issue-observatory:issues', 2, { 952: [794, 0, 2571, 630], 1112: [794, 562, 1333, 548], 1400: [794, 564, 1333, 550] }],
    ['keenable:keenable', 1, { 952: [748, 644, 0, 308], 1112: [694, 0, 1415, 548], 1400: [748, 1128, 0, 268] }],
    ['memory-atlas:atlas', 2, { 952: [654, 0, 3379, 630], 1112: [654, 0, 2123, 548], 1400: [654, 0, 1397, 550] }],
    ['token-usage:observatory', 2, { 952: [634, 0, 4047, 630], 1112: [634, 562, 2141, 548], 1400: [634, 0, 2065, 550] }],
];
const realSpecs = (listWidth, owners = {}) => REAL.map(([key, span, geometry]) => ({
    span, height: geometry[listWidth][0], owner: owners[key],
}));

test('a board without owner spans plans identically with and without the size argument', () => {
    let seed = 20261003;
    const random = () => {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        return seed / 2147483648;
    };
    for (let round = 0; round < 400; round += 1) {
        const specs = Array.from({ length: 1 + Math.floor(random() * 10) }, () => ({
            span: random() < 0.5 ? 2 : 1, height: Math.floor(random() * 1200),
        }));
        const width = 240 + Math.floor(random() * 2000);
        const plain = planMasonryLayout(width, specs);
        for (const owner of [undefined, null, 0, '', 'wide']) {
            assert.deepEqual(planMasonryLayout(width, specs.map((item) => ({ ...item, owner }))), plain);
        }
    }
});

test('the untouched real widgets plan exactly as the owner-approved masonry', () => {
    for (const listWidth of [952, 1112, 1400]) {
        const plan = planMasonryLayout(listWidth, realSpecs(listWidth));
        const expected = REAL.map(([, , geometry]) => geometry[listWidth].slice(1));
        assert.deepEqual(plan.placements.map(({ left, top, width }) => [left, top, width]), expected, `list ${listWidth}px`);
    }
});

test("an owner span replaces the author's, sets the track count with the others and is clamped to it", () => {
    // Two one-column cards share a wide list in halves.
    const pair = [{ span: 1, height: 100 }, { span: 1, height: 100 }];
    assert.deepEqual(placed(planMasonryLayout(1400, pair)), [[1, 0, 0, 693], [1, 707, 0, 693]]);
    // Two columns for the first: three tracks, the row stays full.
    assert.deepEqual(placed(planMasonryLayout(1400, [{ ...pair[0], owner: 2 }, pair[1]])), [[2, 0, 0, 928], [1, 942, 0, 457]]);
    // Three columns: four tracks.
    assert.deepEqual(placed(planMasonryLayout(1400, [{ ...pair[0], owner: 3 }, pair[1]])), [[3, 0, 0, 1045], [1, 1059, 0, 339]]);
    // A list with room for two tracks clamps three columns to the whole row.
    assert.deepEqual(placed(planMasonryLayout(600, [{ ...pair[0], owner: 3 }, pair[1]])), [[2, 0, 0, 600], [1, 0, 114, 293]]);
    // One column for an author-wide card.
    assert.deepEqual(placed(planMasonryLayout(1400, [{ span: 2, height: 100, owner: 1 }, pair[1]])), [[1, 0, 0, 693], [1, 707, 0, 693]]);
});

test('the lone-narrow widening reshapes author defaults only: an owner\'s one column stays one column', () => {
    // On a 1112px list the author-narrow keenable is widened to two of four tracks.
    const untouched = planMasonryLayout(1112, realSpecs(1112));
    assert.equal(untouched.columnCount, 4);
    assert.equal(untouched.placements[4].span, 2);
    const owned = planMasonryLayout(1112, realSpecs(1112, { 'keenable:keenable': 1 }));
    assert.equal(owned.columnCount, 4);
    assert.equal(owned.placements[4].span, 1);
    assert.equal(owned.placements[4].width, untouched.columnWidth);
});

test('Full width takes every track and leaves the track count to its author span', () => {
    const cards = [{ span: 2, height: 100 }, { span: 1, height: 80 }];
    const untouched = planMasonryLayout(1400, cards);
    assert.deepEqual([untouched.columnCount, untouched.placements[1].width], [3, 457]);
    const full = planMasonryLayout(1400, [{ ...cards[0], owner: 12 }, cards[1]]);
    assert.equal(full.columnCount, 3, 'the other card keeps its width');
    assert.deepEqual(placed(full), [[3, 0, 0, 1399], [1, 0, 114, 457]]);
    // Every value from Full width up means every track; a stack is one track whatever the owner chose.
    assert.deepEqual(placed(planMasonryLayout(1400, [{ ...cards[0], owner: 99 }, cards[1]])), placed(full));
    const narrow = planMasonryLayout(500, [{ ...cards[0], owner: 12 }, { ...cards[1], owner: 3 }]);
    assert.equal(narrow.availableColumns, 1);
    assert.deepEqual(narrow.placements.map(({ span, width }) => [span, width]), [[1, 500], [1, 500]]);
});

function fakeItem(key, { height = 100, span = 1 } = {}) {
    const props = new Map();
    return {
        dataset: { widgetKey: key },
        offsetHeight: height,
        classList: { contains: (name) => span === 2 && name === 'widgets-card-span-2' },
        style: { setProperty: (name, value) => props.set(name, value), removeProperty: (name) => props.delete(name) },
        props,
    };
}

function installFrames() {
    const frames = new Map();
    let next = 1;
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
    globalThis.MutationObserver = class { observe() {} disconnect() {} };
    globalThis.requestAnimationFrame = (callback) => { frames.set(next, callback); return next++; };
    globalThis.cancelAnimationFrame = (id) => frames.delete(id);
    return () => {
        const callbacks = [...frames.values()];
        frames.clear();
        callbacks.forEach((callback) => callback());
    };
}

test('applyMasonry plans with the owner spans by key and hands onLayout each plan, its items and a replan', () => {
    const flush = installFrames();
    const a = fakeItem('demo:a');
    const b = fakeItem('demo:b');
    const props = new Map();
    const container = {
        clientWidth: 600, dataset: {}, items: [a, b],
        querySelectorAll() { return this.items.slice(); },
        contains(item) { return this.items.includes(item); },
        style: { setProperty: (name, value) => props.set(name, value), removeProperty: (name) => props.delete(name) },
    };
    const plans = [];
    const dispose = applyMasonry(container, {
        order: ['demo:b', 'demo:a'], spans: { 'demo:a': 12 }, onLayout: (...args) => plans.push(args),
    });
    flush();
    assert.equal(a.props.get('--masonry-w'), '600px', 'Full width spans both tracks');
    assert.equal(a.props.get('--masonry-y'), '114px');
    assert.equal(plans.length, 1);
    const [plan, items, replan] = plans[0];
    assert.deepEqual(items, [b, a], 'the items in the planned (key) order');
    assert.deepEqual([plan.columnCount, plan.availableColumns], [2, 2]);
    // What a card would be under another owner span: the same answer a plan of
    // that board gives, with nothing measured again.
    const specs = [{ span: 1, height: 100 }, { span: 1, height: 100, owner: 12 }];
    for (const owner of [undefined, 1, 2, 3, 12]) {
        const asked = specs.map((spec, i) => (i === 0 ? { ...spec, owner } : spec));
        assert.deepEqual(replan(0, owner), planMasonryLayout(600, asked), `owner ${owner}`);
    }
    assert.deepEqual(replan(0, undefined), plan, 'no other span: the plan itself');
    assert.deepEqual(container.dataset, {}, 'masonry writes no attribute; the caller decides what a plan means');
    // A later call replaces the spans; without them the card is back to its author span.
    applyMasonry(container, { spans: {} });
    flush();
    assert.equal(a.props.get('--masonry-w'), '293px');
    assert.equal(plans.length, 2);
    dispose();
});
