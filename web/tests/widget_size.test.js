import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    defaultWidgetWidth,
    nearestWidthChoice,
    normalizeWidgetSize,
    ownerSpans,
    stepWidgetWidth,
    WIDGET_FULL_SPAN,
    WIDGET_SIZE_MAX_ITEMS,
    WIDGET_WIDTH_STEPS,
    widgetWidth,
    widthChoices,
} from '../modules/widget_size.js';

// The owner's card widths (docs/DESIGN.md "Widgets board"): a width is a column
// count of the masonry board, Full width every column it has; until the owner
// picks one, the author's `span` (1 or 2) decides.

const tab = (key, span) => ({ key, skill: key.split(':')[0], tab_id: key.split(':')[1], span });

test('a saved width clamps to 1..Full width, h stays 0, anything but an integer w is no size', () => {
    assert.deepEqual(normalizeWidgetSize({
        'a:main': { w: 2, h: 0 },
        ' b:main ': { w: 99, h: 640 },
        'c:main': { w: -3 },
        'd:main': { w: '2' },
        'e:main': { w: 1.5 },
        'f:main': null,
        '': { w: 1 },
        [`${'x'.repeat(201)}`]: { w: 1 },
    }), { 'a:main': { w: 2, h: 0 }, 'b:main': { w: WIDGET_FULL_SPAN, h: 0 }, 'c:main': { w: 1, h: 0 } });
    for (const bad of [null, undefined, [], 'wide', 4]) assert.deepEqual(normalizeWidgetSize(bad), {});
    const many = Object.fromEntries(Array.from({ length: 250 }, (_, i) => [`s:${i}`, { w: 1 }]));
    assert.equal(Object.keys(normalizeWidgetSize(many)).length, WIDGET_SIZE_MAX_ITEMS);
});

test("the author span is the default width in columns and the owner's width wins over it", () => {
    assert.equal(defaultWidgetWidth(tab('a:main', 1)), 1);
    assert.equal(defaultWidgetWidth(tab('a:main', 2)), 2);
    assert.equal(defaultWidgetWidth({ grid_span: 2 }), 2);
    assert.equal(defaultWidgetWidth({}), 1);
    assert.equal(defaultWidgetWidth(null), 1);
    const sizes = { 'a:main': { w: WIDGET_FULL_SPAN, h: 0 } };
    assert.equal(widgetWidth(tab('a:main', 1), sizes), WIDGET_FULL_SPAN);
    assert.equal(widgetWidth(tab('b:main', 2), sizes), 2, 'a size for another card never leaks');
    assert.equal(widgetWidth({ skill: 'a', tab_id: 'main' }, sizes), WIDGET_FULL_SPAN, 'skill:tab_id is the key fallback');
    // The masonry takes the owner's widths by key, nothing else.
    assert.deepEqual(ownerSpans({ 'a:main': { w: 3, h: 0 }, 'b:main': null, 'c:main': {} }), { 'a:main': 3 });
    assert.deepEqual(ownerSpans(undefined), {});
});

test('width steps: 1, 2, 3 columns and Full width; keys walk them, held at both ends', () => {
    assert.deepEqual(WIDGET_WIDTH_STEPS.map((step) => step.w), [1, 2, 3, WIDGET_FULL_SPAN]);
    assert.deepEqual(WIDGET_WIDTH_STEPS.map((step) => step.label), ['1 column', '2 columns', '3 columns', 'Full width']);
    assert.equal(WIDGET_FULL_SPAN, 12);
    assert.equal(stepWidgetWidth(1, 1), 2);
    assert.equal(stepWidgetWidth(3, 1), WIDGET_FULL_SPAN);
    assert.equal(stepWidgetWidth(WIDGET_FULL_SPAN, 1), WIDGET_FULL_SPAN, 'held at the last step');
    assert.equal(stepWidgetWidth(WIDGET_FULL_SPAN, -1), 3);
    assert.equal(stepWidgetWidth(1, -1), 1, 'held at the first step');
    // A width between steps (a hand-edited file) steps to its neighbours.
    assert.equal(stepWidgetWidth(5, 1), WIDGET_FULL_SPAN);
    assert.equal(stepWidgetWidth(5, -1), 3);
});

test('the steps\' widths on a board are its choices: equal shares are one choice, one choice means a fixed card', () => {
    // What the masonry makes of each step for one card (width px, share of the row).
    const place = (table) => (w) => table[w];
    const twoCards = { 1: { width: 693, share: 1 / 2 }, 2: { width: 928, share: 2 / 3 }, 3: { width: 1045, share: 3 / 4 }, 12: { width: 1400, share: 1 } };
    assert.deepEqual(widthChoices(place(twoCards), 1).map(({ w, width }) => [w, width]), [[1, 693], [2, 928], [3, 1045], [12, 1400]]);
    // Three columns fill a three-column row like Full width does (rounding aside): one choice,
    // the narrower step unless the card's current step is the other one.
    const threeColumns = { 1: { width: 308, share: 1 / 3 }, 2: { width: 630, share: 2 / 3 }, 3: { width: 952, share: 1 }, 12: { width: 951, share: 3 / 3 } };
    assert.deepEqual(widthChoices(place(threeColumns), 1).map(({ w }) => w), [1, 2, 3]);
    assert.deepEqual(widthChoices(place(threeColumns), WIDGET_FULL_SPAN).map(({ w }) => w), [1, 2, WIDGET_FULL_SPAN]);
    // The only card on a board: every step is the whole row, however it rounds.
    const alone = { 1: { width: 1400, share: 1 }, 2: { width: 1400, share: 1 }, 3: { width: 1399, share: 1 }, 12: { width: 1400, share: 1 } };
    assert.equal(widthChoices(place(alone), 1).length, 1);
});

test('a drag lands on the choice nearest the edge, a tie keeping the narrower', () => {
    const choices = [{ w: 1, width: 693 }, { w: 2, width: 928 }, { w: 3, width: 1045 }, { w: 12, width: 1400 }];
    assert.equal(nearestWidthChoice(choices, 0).w, 1);
    assert.equal(nearestWidthChoice(choices, 900).w, 2);
    assert.equal(nearestWidthChoice(choices, 1100).w, 3);
    assert.equal(nearestWidthChoice(choices, 9000).w, 12);
    assert.equal(nearestWidthChoice(choices, (928 + 1045) / 2).w, 2, 'a tie keeps the narrower');
    assert.equal(nearestWidthChoice([{ w: 12, width: 900 }, { w: 1, width: 300 }], 600).w, 1, 'narrower by width, not by order');
});

test('the size module measures nothing and writes no DOM', () => {
    const source = readFileSync(new URL('../modules/widget_size.js', import.meta.url), 'utf8');
    for (const forbidden of ['document', 'querySelector', 'style.', 'offsetHeight', 'getBoundingClientRect', 'ResizeObserver']) {
        assert.equal(source.includes(forbidden), false, `widget_size.js must not use ${forbidden}`);
    }
});
