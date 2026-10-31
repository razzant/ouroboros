// The open strip's fit (capsule_fit.js) counts a segment's minimum width: Subagents' options keep a finger's width, so
// a line that holds the words with their padding but not those minimums must not be called inline (it would wrap).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createCapsuleFit } from '../modules/capsule_fit.js';

const box = (width, extra = {}) => ({ getBoundingClientRect: () => ({ width }), dataset: {}, style: { setProperty() {} }, ...extra });

// Head 109 px with its name (6 + ring 22 + 4, name 60 with 4 a side, divider 4 + 1 + 4), two peers of 120 px with an
// 8 px gap, and Off · Auto · Swarm words of 19, 20 and 26 px.
function openStrip({ rowWidth, segMin }) {
    const parts = {
        '.chat-capsule-head': box(0), '.chat-capsule-glyph': box(22), '.chat-capsule-name': box(60),
        '.chat-capsule-value': box(30), '.chat-capsule-body': box(0),
    };
    const segs = [19, 20, 26].map((word) => box(word, { firstElementChild: box(word) }));
    const el = box(200, { querySelector: (selector) => parts[selector] || null });
    const strip = box(0);
    const row = box(rowWidth, { children: [el, box(120), box(120)] });
    const styles = new Map([
        [row, { paddingLeft: '0px', paddingRight: '0px', columnGap: '8px' }],
        [parts['.chat-capsule-head'], { paddingLeft: '6px', paddingRight: '4px' }],
        [parts['.chat-capsule-name'], { marginLeft: '4px', marginRight: '4px' }],
        ...segs.map((seg) => [seg, { minWidth: `${segMin}px` }]),
    ]);
    const win = {
        getComputedStyle: (node, pseudo) => (pseudo === '::before' && node === parts['.chat-capsule-body']
            ? { marginLeft: '4px', width: '1px', marginRight: '4px' } : styles.get(node) || {}),
    };
    const fit = createCapsuleFit({ el, row, kind: 'subagents', strip, segs, isOpen: () => true, win, doc: {} });
    fit.fit();
    return el.dataset.fit;
}

test('a strip whose minimum widths do not fit beside the other capsules stands alone', () => {
    // 209 px beside the peers: the words with 5 px a side fit (204), the 36 px minimums do not (217).
    assert.equal(openStrip({ rowWidth: 466, segMin: 36 }), 'solo');
    assert.equal(openStrip({ rowWidth: 466, segMin: 0 }), 'inline');
});

test('a strip whose minimum widths fit stays in the row', () => {
    assert.equal(openStrip({ rowWidth: 486, segMin: 36 }), 'inline');
});
