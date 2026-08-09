/**
 * Selection → line-range mapping (plan §5.1) plus the chip the Files page builds
 * from it, and the two capture DECISIONS around it (which range a capture reads,
 * whether it may inline bytes). All four are PURE functions over plain shapes, so
 * every pinned case runs without a DOM: the DOM layer's only job is resolving
 * each boundary to a `data-line-number` row — including whether an ELEMENT
 * boundary sits before that row's code text — and handing the shape over.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { captureInlinesContent, resolveCaptureRange, selectionLineRange } from '../modules/files.js';
import { makeChipPart, serializeParts } from '../modules/composer_parts.js';

const range = (startLine, startOffset, endLine, endOffset) =>
    selectionLineRange({ startLine, startOffset, endLine, endOffset });

test('a forward selection spans both boundary lines inclusively', () => {
    assert.deepEqual(range(10, 4, 14, 7), { lineStart: 10, lineEnd: 14 });
});

test('a backward selection is ordered, not rejected', () => {
    // Dragging upward puts focus before anchor; both directions must name the
    // same range.
    assert.deepEqual(range(14, 7, 10, 4), { lineStart: 10, lineEnd: 14 });
    assert.deepEqual(range(14, 7, 10, 4), range(10, 4, 14, 7));
});

test('an end boundary at offset 0 excludes that line', () => {
    // The caret sits BEFORE line 15's first character, so line 15 is not selected.
    assert.deepEqual(range(10, 0, 15, 0), { lineStart: 10, lineEnd: 14 });
    // One character into the line puts it back in.
    assert.deepEqual(range(10, 0, 15, 1), { lineStart: 10, lineEnd: 15 });
    // Backward selection ending at offset 0 of the later line: same rule.
    assert.deepEqual(range(15, 0, 10, 2), { lineStart: 10, lineEnd: 14 });
});

test('a single line stays a single line', () => {
    assert.deepEqual(range(7, 0, 7, 12), { lineStart: 7, lineEnd: 7 });
    assert.deepEqual(range(7, 12, 7, 0), { lineStart: 7, lineEnd: 7 });
    // Offset 0 on the SAME line never empties the range.
    assert.deepEqual(range(7, 0, 7, 1), { lineStart: 7, lineEnd: 7 });
});

test('a collapsed selection captures nothing', () => {
    assert.equal(range(7, 5, 7, 5), null);
    assert.equal(range(1, 0, 1, 0), null);
});

test('excluding the boundary line can never fall below the start line', () => {
    // Two adjacent rows, the end boundary at offset 0 of the later one: the
    // exclusion takes the range down to exactly its start line and stops there —
    // it is never allowed to invert into "nothing".
    assert.deepEqual(range(9, 0, 10, 0), { lineStart: 9, lineEnd: 9 });
    assert.deepEqual(range(9, 4, 10, 0), { lineStart: 9, lineEnd: 9 });
    // Backward drag, same shape.
    assert.deepEqual(range(10, 0, 9, 4), { lineStart: 9, lineEnd: 9 });
});

test('non-resolving boundaries are refused rather than guessed', () => {
    assert.equal(selectionLineRange(), null);
    assert.equal(selectionLineRange({}), null);
    assert.equal(range(null, 0, 4, 2), null);
    assert.equal(range(0, 0, 4, 2), null);
    assert.equal(range(-3, 0, 4, 2), null);
    assert.equal(range(2.5, 0, 4, 2), null);
    assert.equal(range(2, 0, undefined, 2), null);
});

test('negative or non-numeric offsets read as the line start', () => {
    assert.deepEqual(range(4, -5, 6, 3), { lineStart: 4, lineEnd: 6 });
    assert.deepEqual(range(4, 'x', 6, 3), { lineStart: 4, lineEnd: 6 });
    // A non-numeric END offset reads as 0 -> the last line is excluded.
    assert.deepEqual(range(4, 1, 6, 'x'), { lineStart: 4, lineEnd: 5 });
});

test('the mapped range drives the chip: full lines, verbatim, in the marker', () => {
    const lines = [
        'class ToolExecutor:',
        '',
        '    async def run(self, call):',
        '        return await self._dispatch(call)',
    ];
    const mapped = range(1, 6, 3, 9);
    assert.deepEqual(mapped, { lineStart: 1, lineEnd: 3 });

    const chip = makeChipPart({
        path: '/Users/o/ouroboros/tools.py',
        lineStart: mapped.lineStart,
        lineEnd: mapped.lineEnd,
        content: lines.slice(mapped.lineStart - 1, mapped.lineEnd).join('\n'),
    });
    assert.equal(serializeParts([chip]), [
        '[context: /Users/o/ouroboros/tools.py L1-L3]',
        '```',
        'class ToolExecutor:',
        '',
        '    async def run(self, call):',
        '```',
    ].join('\n'));
});

test('an unrepresentable path yields no chip (the page discloses instead)', () => {
    assert.equal(makeChipPart({ path: 'weird]name.py', lineStart: 1, lineEnd: 2, content: 'x' }), null);
    assert.equal(makeChipPart({ path: '', lineStart: 1, lineEnd: 2, content: 'x' }), null);
});

// ---------------------------------------------------------------------------
// Element boundaries: a child index is not a character offset (DOM spec)
// ---------------------------------------------------------------------------

test('a boundary before the row text reads as offset 0 whatever its child index', () => {
    // A START boundary on the row ELEMENT with child index 1 (after the gutter
    // span, before the code text). Read as a character offset that would be "one
    // char into line 4"; read correctly it is the very start of line 4 — which for
    // a START boundary changes nothing, so the range must still begin at 4.
    // SPEC-HARDENING, not a reproduced browser bug: probing Chromium and WebKit, a
    // drag STARTED over the gutter never yielded a nonzero child index. This case
    // pins that the flag is harmless where it cannot matter.
    assert.deepEqual(
        selectionLineRange({ startLine: 4, startOffset: 1, startBeforeText: true, endLine: 6, endOffset: 3 }),
        { lineStart: 4, lineEnd: 6 },
    );
    // The same flag on the END boundary is what actually decides a line: child
    // index 1 would keep line 6, but the boundary precedes every character of it.
    assert.deepEqual(
        selectionLineRange({ startLine: 4, startOffset: 2, endLine: 6, endOffset: 1, endBeforeText: true }),
        { lineStart: 4, lineEnd: 5 },
    );
    // Without the flag the same numbers include line 6 — the flag is load-bearing.
    assert.deepEqual(
        selectionLineRange({ startLine: 4, startOffset: 2, endLine: 6, endOffset: 1 }),
        { lineStart: 4, lineEnd: 6 },
    );
});

test('element boundaries survive a backward drag through the gutter', () => {
    // Backward drag whose FOCUS ended in line 4's gutter: ordering happens after
    // normalization, so the (later) line-6 boundary keeps its own offset and the
    // gutter boundary reads as the start of line 4.
    assert.deepEqual(
        selectionLineRange({ startLine: 6, startOffset: 5, endLine: 4, endOffset: 1, endBeforeText: true }),
        { lineStart: 4, lineEnd: 6 },
    );
    // Backward drag that ENDS on line 6's gutter (start boundary is the later
    // line): after ordering, line 6 is the excluded one.
    assert.deepEqual(
        selectionLineRange({ startLine: 6, startOffset: 2, startBeforeText: true, endLine: 4, endOffset: 3 }),
        { lineStart: 4, lineEnd: 5 },
    );
    // A gutter-to-gutter drag over two rows names just the first row.
    assert.deepEqual(
        selectionLineRange({
            startLine: 4, startOffset: 1, startBeforeText: true,
            endLine: 5, endOffset: 1, endBeforeText: true,
        }),
        { lineStart: 4, lineEnd: 4 },
    );
});

// ---------------------------------------------------------------------------
// Truncated preview: the last shown line may be a fragment, so no inline bytes
// ---------------------------------------------------------------------------

test('a complete preview always inlines its bytes', () => {
    assert.equal(captureInlinesContent({ truncated: false, lineEnd: 40, shownLines: 40 }), true);
    assert.equal(captureInlinesContent({ truncated: false, lineEnd: 1, shownLines: 1 }), true);
});

test('a truncated preview refuses to inline the line the server cut', () => {
    // The last SHOWN line of a prefix can end mid-statement, so a range touching
    // it must not claim to be those bytes.
    assert.equal(captureInlinesContent({ truncated: true, lineEnd: 900, shownLines: 900 }), false);
    // Lines strictly before the cut are whole and inline normally.
    assert.equal(captureInlinesContent({ truncated: true, lineEnd: 899, shownLines: 900 }), true);
    // Defensive: unusable numbers degrade to the honest side (no bytes).
    assert.equal(captureInlinesContent({ truncated: true, lineEnd: NaN, shownLines: 900 }), false);
    assert.equal(captureInlinesContent(), true);
});

test('the truncated capture ships the RANGE, not a fragment pretending to be it', () => {
    const shown = ['def loop():', '    while True:', '        step()  # cut he'];
    const mapped = range(2, 4, 3, 9);
    assert.deepEqual(mapped, { lineStart: 2, lineEnd: 3 });
    const inline = captureInlinesContent({ truncated: true, lineEnd: mapped.lineEnd, shownLines: shown.length });
    assert.equal(inline, false);
    const chip = makeChipPart({
        path: '/Users/o/ouroboros/loop.py',
        lineStart: mapped.lineStart,
        lineEnd: mapped.lineEnd,
        content: inline ? shown.slice(mapped.lineStart - 1, mapped.lineEnd).join('\n') : null,
    });
    // Ranged BARE marker: true line numbers, no fenced block at all.
    assert.equal(chip.content, undefined);
    assert.equal(serializeParts([chip]), '[context: /Users/o/ouroboros/loop.py L2-L3]');
});

// ---------------------------------------------------------------------------
// Which range a capture reads (the stale-range rule)
// ---------------------------------------------------------------------------

test('a live selection always wins over the cache', () => {
    const live = { lineStart: 3, lineEnd: 9 };
    const cached = { lineStart: 100, lineEnd: 120 };
    assert.equal(resolveCaptureRange({ live, cached }), live);
    assert.equal(resolveCaptureRange({ live, cached, selectionOnly: true }), live);
    assert.equal(resolveCaptureRange({ live, cached, focusInDock: true }), live);
});

test('only the selection button and a dock-focused ⌘L may read the cache', () => {
    const cached = { lineStart: 12, lineEnd: 14 };
    // The sticky button is reachable only while a selection is visible; clicking
    // it can collapse that selection before the handler runs.
    assert.equal(resolveCaptureRange({ live: null, cached, selectionOnly: true }), cached);
    // "Select code, type a comment in the dock, then ⌘L": focusing the dock is
    // exactly what collapsed the selection, so the cache is the truth here.
    assert.equal(resolveCaptureRange({ live: null, cached, focusInDock: true }), cached);
    // ⌘L anywhere else with nothing selected means "the whole file" — a remembered
    // range would silently narrow the capture to stale lines.
    assert.equal(resolveCaptureRange({ live: null, cached }), null);
    assert.equal(resolveCaptureRange({ live: null, cached: null, selectionOnly: true }), null);
    assert.equal(resolveCaptureRange({ live: null, cached: null, focusInDock: true }), null);
    assert.equal(resolveCaptureRange(), null);
});
