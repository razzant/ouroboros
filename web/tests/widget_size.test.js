import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    clampWidgetHeight,
    defaultWidgetWidth,
    draggedWidgetHeight,
    heightName,
    nearestWidthChoice,
    nextWidgetSize,
    normalizeWidgetSize,
    ownerSpans,
    stepWidgetWidth,
    WIDGET_FULL_SPAN,
    WIDGET_HEIGHT_MAX,
    WIDGET_HEIGHT_MIN,
    WIDGET_SIZE_MAX_ITEMS,
    WIDGET_WIDTH_STEPS,
    widgetHeight,
    widgetWidth,
    widthChoices,
} from '../modules/widget_size.js';
import { boardFrameGeometry, WIDGET_FRAME_DEFAULT_HEIGHT, WIDGET_FRAME_MAX_HEIGHT } from '../modules/widget_module.js';

// The owner's card widths (docs/DESIGN.md "Widgets board"): a width is a column
// count of the masonry board, Full width every column it has; until the owner
// picks one, the author's `span` (1 or 2) decides.

const tab = (key, span) => ({ key, skill: key.split(':')[0], tab_id: key.split(':')[1], span });

test('a saved size: w clamps to 0 (the author) ..Full width, h to 0 (natural) or its bounds; neither is no size', () => {
    assert.deepEqual(normalizeWidgetSize({
        'a:main': { w: 2, h: 0 },
        ' b:main ': { w: 99, h: 640 },
        'c:main': { w: -3 },
        'd:main': { w: '2' },
        'e:main': { w: 1.5 },
        'f:main': null,
        'g:main': { w: 0, h: 50 },
        'h:main': { w: 0, h: 99999 },
        'i:main': { w: 3, h: 2.5 },
        'j:main': { w: 0, h: 0 },
        '': { w: 1 },
        [`${'x'.repeat(201)}`]: { w: 1 },
    }), {
        'a:main': { w: 2, h: 0 }, 'b:main': { w: WIDGET_FULL_SPAN, h: 640 },
        'g:main': { w: 0, h: WIDGET_HEIGHT_MIN }, 'h:main': { w: 0, h: WIDGET_HEIGHT_MAX }, 'i:main': { w: 3, h: 0 },
    });
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

test('the two axes are independent: a height keeps the width the board chose, a width keeps the height', () => {
    // A height alone never pins a width: the author's span (w 0) stays the author's.
    assert.deepEqual(nextWidgetSize(undefined, { h: 432 }), { w: 0, h: 432 });
    assert.deepEqual(nextWidgetSize({ w: 3, h: 0 }, { h: 432 }), { w: 3, h: 432 });
    assert.deepEqual(nextWidgetSize({ w: 0, h: 432 }, { w: 2 }), { w: 2, h: 432 });
    // Natural height resets h alone; with no width left the card has no size.
    assert.deepEqual(nextWidgetSize({ w: 2, h: 432 }, { h: 0 }), { w: 2, h: 0 });
    assert.equal(nextWidgetSize({ w: 0, h: 432 }, { h: 0 }), null);
    assert.equal(nextWidgetSize({ w: 2, h: 0 }, { w: 0 }), null);
    // Heights clamp to their bounds.
    assert.deepEqual(nextWidgetSize(null, { h: 10 }), { w: 0, h: WIDGET_HEIGHT_MIN });
    assert.equal(clampWidgetHeight(-5), 0);
    assert.equal(clampWidgetHeight(Number.NaN), 0);
    assert.equal(clampWidgetHeight(1e9), WIDGET_HEIGHT_MAX);
    // A card with only an owner height is no owner width: widths and spans read the author's.
    const sizes = { 'a:main': { w: 0, h: 432 } };
    assert.equal(widgetWidth(tab('a:main', 2), sizes), 2);
    assert.equal(widgetHeight('a:main', sizes), 432);
    assert.equal(widgetHeight('b:main', sizes), 0);
    assert.deepEqual(ownerSpans(sizes), {});
});

test('a dragged height snaps, keeps its floor and catches the content height as natural from either side', () => {
    assert.equal(draggedWidgetHeight(400, 13), 416, 'snapped to the 8 px grid');
    assert.equal(draggedWidgetHeight(400, -1000), WIDGET_HEIGHT_MIN);
    assert.equal(draggedWidgetHeight(400, 395, 800), 0, 'within the catch of the content height');
    assert.equal(draggedWidgetHeight(400, 405, 800), 0, 'within the catch above it too');
    assert.equal(draggedWidgetHeight(400, 360, 800), 760, 'short of the catch stays a height');
    assert.equal(draggedWidgetHeight(400, 440, 800), 840, 'past the catch is a height taller than its content');
    assert.equal(draggedWidgetHeight(400, 2000, 800), 2400, 'as far as the owner drags it');
    assert.equal(draggedWidgetHeight(400, 2000, 0), 2400, 'a fixed frame (no content height) just grows');
    assert.equal(heightName(0), 'natural');
    assert.equal(heightName(432), '432 px');
});

test('a board frame: its content under the author\'s ceiling, or exactly the owner\'s height; Show all lifts it', () => {
    const auto = { kind: 'module', entry: 'w.js' };
    // Before its first report the frame stands at the loading height; after it, at its content (no 320 floor).
    assert.deepEqual(boardFrameGeometry(auto), { auto: true, height: WIDGET_FRAME_DEFAULT_HEIGHT, ceiling: WIDGET_FRAME_MAX_HEIGHT, floor: 0, hidden: 0, expandable: false });
    assert.equal(boardFrameGeometry(auto, { natural: 90 }).height, 90);
    // The owner's height is the frame's height: shorter than the content, with what it keeps out of view for Show all,
    const bounded = boardFrameGeometry(auto, { h: 400, natural: 1500 });
    assert.deepEqual([bounded.height, bounded.ceiling, bounded.hidden, bounded.expandable], [400, 400, 1100, true]);
    // or taller, with nothing to show: the room below the content is the owner's.
    assert.deepEqual(boardFrameGeometry(auto, { h: 400, natural: 300 }), { auto: true, height: 400, ceiling: 400, floor: 0, hidden: 0, expandable: false });
    // Expanded: the content under the author's ceiling again; the owner's height stays the owner's.
    assert.deepEqual([boardFrameGeometry(auto, { h: 400, natural: 1500, expanded: true }).height,
        boardFrameGeometry(auto, { h: 400, natural: 1500, expanded: true }).ceiling], [1500, WIDGET_FRAME_MAX_HEIGHT]);
    // An author-capped frame: the owner's height wins over the cap too, and Show all promises no more than the cap.
    const capped = { ...auto, max_height: 600 };
    assert.equal(boardFrameGeometry(capped, { natural: 2000 }).height, 600);
    assert.equal(boardFrameGeometry(capped, { h: 400, natural: 2000 }).hidden, 200);
    const pastCap = boardFrameGeometry(capped, { h: 800, natural: 2000 });
    assert.deepEqual([pastCap.height, pastCap.ceiling, pastCap.expandable], [800, 800, false]);
    assert.equal(boardFrameGeometry(capped, { h: 800, natural: 2000, expanded: true }).height, 600, 'shown all, the author\'s ceiling again');
    // Before its first report a card with an owner height already stands at it.
    assert.equal(boardFrameGeometry(auto, { h: 200 }).height, 200);
    // Fixed module and route frame: h is the frame; Show all returns the declared height.
    const fixed = { kind: 'module', entry: 'w.js', height: 720 };
    assert.deepEqual(boardFrameGeometry(fixed, { h: 400 }), { auto: false, height: 400, ceiling: 400, floor: 400, hidden: 0, expandable: true });
    assert.equal(boardFrameGeometry(fixed, { h: 400, expanded: true }).height, 720);
    assert.equal(boardFrameGeometry(fixed, { h: 900 }).expandable, false);
    const route = { kind: 'iframe', route: 'page' };
    assert.equal(boardFrameGeometry(route).height, WIDGET_FRAME_DEFAULT_HEIGHT);
    assert.equal(boardFrameGeometry(route, { h: 640 }).height, 640);
});
