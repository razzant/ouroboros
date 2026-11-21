import test from 'node:test';
import assert from 'node:assert/strict';

import { renderSegmentedField } from '../modules/page_header.js';
import { bindSegmentFill } from '../modules/segment_fill.js';

const labels = (count) => Array.from({ length: count }, (_, index) => ({ value: `v${index}`, label: `Choice ${index}` }));

test('the strip is markup alone: no hover title, no inline style, no count; its hooks stay', () => {
    for (const count of [1, 2, 3, 5, 8]) {
        const html = renderSegmentedField({ target: 's-x', modifier: 'data-review-cycles-group', options: labels(count) });
        assert.doesNotMatch(html, /\stitle=/, 'a fact the owner needs is never a hover title');
        assert.doesNotMatch(html, /\sstyle=/, 'the stylesheet lays the strip out; nothing is written inline');
        assert.doesNotMatch(html, /--segment-count/);
        // What its reader (place_sheet.js) finds: the group, its target input and each choice's value.
        assert.match(html, /<div class="ui-segment-group" data-effort-group data-review-cycles-group data-effort-target="s-x">/);
        assert.deepEqual([...html.matchAll(/<button type="button" class="ui-segment" data-effort-value="(v\d+)">Choice \d+<\/button>/g)]
            .map((match) => match[1]), labels(count).map(({ value }) => value));
    }
    // The sentence a caller once hid in the title has no attribute left to land in.
    const html = renderSegmentedField({ target: 's-x', title: 'Changes at once', options: labels(2) });
    assert.doesNotMatch(html, /Changes at once/);
    assert.throws(() => renderSegmentedField({ options: labels(2) }), /requires target/);
});

for (const scale of [1, 0.97]) test(`the selection fill keeps layout coordinates through a page arrival at scale ${scale}`, () => {
    const values = new Map();
    const option = { matches: () => true, offsetTop: 3.25,
        getBoundingClientRect: () => ({ left: 30 + 11.25 * scale, top: 40 + 4.25 * scale,
            width: 40.5 * scale, height: 24.5 * scale }) };
    const group = { isConnected: true, children: [option], dataset: {}, clientLeft: 1, clientTop: 1,
        style: { setProperty: (key, value) => values.set(key, value), removeProperty: key => values.delete(key) },
        getBoundingClientRect: () => ({ left: 30, top: 40, width: 200 * scale, height: 40 * scale }),
        append(node) { if (!this.children.includes(node)) this.children.push(node); node.parentNode = this; } };
    const win = { MutationObserver: class { observe() {} disconnect() {} },
        getComputedStyle: () => ({ width: '200px', height: '40px' }) };
    group.ownerDocument = { defaultView: win, createElement: () => ({ setAttribute() {},
        remove() { group.children = group.children.filter(node => node !== this); this.parentNode = null; } }) };
    const dispose = bindSegmentFill(group);
    try {
        for (const [edge, expected] of Object.entries({ x: 10.25, y: 3.25, w: 40.5, h: 24.5 })) {
            assert.ok(Math.abs(parseFloat(values.get(`--segment-fill-${edge}`)) - expected) < 1e-10,
                `${edge}: the page's temporary scale must not become a stored CSS dimension`);
        }
        assert.equal(group.children.length, 2);
        assert.equal(group.children[1].hidden, false);
    } finally { dispose(); }
    assert.equal(values.size, 0);
    assert.deepEqual(group.children, [option]);
});
