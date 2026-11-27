// The open strip's fit (capsule_fit.js) counts everything that stands on the line with it: a segment's minimum width
// (Subagents' options keep a finger's width) and a mark that stays in the head while the strip is open (Subagents'
// lock). A line that holds the words with their padding but not those must not be called inline: it would wrap.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createCapsuleFit } from '../modules/capsule_fit.js';

const box = (width, extra = {}) => ({ getBoundingClientRect: () => ({ width }), dataset: {}, style: { setProperty() {} }, ...extra });

// Head 109 px with its name (6 + ring 22 + 4, name 60 with 4 a side, divider 4 + 1 + 4), two peers of 120 px with an
// 8 px gap, and Off · Auto · Swarm words of 19, 20 and 26 px. `mark`: the width of a mark in the head beside the
// ring, the name and the value, 0 for a head without one. Answers the fit and the padding it wrote.
function fitted({ rowWidth, segMin, latest = false, mark = 0 }) {
    const parts = {
        '.chat-capsule-head': box(0), '.chat-capsule-glyph': box(22), '.chat-capsule-name': box(60),
        '.chat-capsule-value': box(30), '.chat-capsule-body': box(0),
    };
    parts['.chat-capsule-head'].children = [parts['.chat-capsule-glyph'], parts['.chat-capsule-name'],
        parts['.chat-capsule-value'], ...(mark ? [box(mark)] : [])];
    const segs = [19, 20, 26].map((word) => box(word, { firstElementChild: box(word) }));
    const written = {};
    const el = box(200, { querySelector: (selector) => parts[selector] || null,
        style: { setProperty(name, value) { written[name] = value; } } });
    const strip = box(0);
    const endPad = latest ? 38 : 0;
    const row = box(rowWidth + endPad, { children: [el, box(120), box(120)] });
    if (latest) row.children.push(box(30, { classList: { contains: name => name === 'chat-scroll-bottom-btn' } }));
    const styles = new Map([
        [row, { paddingLeft: '0px', paddingRight: `${endPad}px`, columnGap: '8px' }],
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
    return { mode: el.dataset.fit, pad: written['--capsule-seg-pad'] };
}
const openStrip = (options) => fitted(options).mode;

test('a strip whose minimum widths do not fit beside the other capsules stands alone', () => {
    // 209 px beside the peers: the words with 5 px a side fit (204), the 36 px minimums do not (217).
    assert.equal(openStrip({ rowWidth: 466, segMin: 36 }), 'solo');
    assert.equal(openStrip({ rowWidth: 466, segMin: 0 }), 'inline');
});

test('a strip whose minimum widths fit stays in the row', () => {
    assert.equal(openStrip({ rowWidth: 486, segMin: 36 }), 'inline');
    assert.equal(openStrip({ rowWidth: 486, segMin: 36, latest: true }), 'inline', 'latest is already allowed for in the end pad');
});

test('a mark that stays in the head beside the open strip takes its room on the line', () => {
    // 229 px beside the peers. Without a mark the strip keeps its widest padding (109 + 65 + 48 = 222); a 20 px lock
    // leaves it 5 px a side (129 + 65 + 30 = 224), where the widest would stand 13 px over the line and wrap it.
    assert.deepEqual(fitted({ rowWidth: 486, segMin: 0 }), { mode: 'inline', pad: '8px' });
    assert.deepEqual(fitted({ rowWidth: 486, segMin: 0, mark: 20 }), { mode: 'inline', pad: '5px' });
    // 209 px beside the peers hold the strip alone at 5 px a side (204), never the strip and the lock (218 at the
    // least padding beside the others): the others step aside.
    assert.equal(openStrip({ rowWidth: 466, segMin: 0, mark: 20 }), 'solo');
});
