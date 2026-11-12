import test from 'node:test';
import assert from 'node:assert/strict';

import { renderSegmentedField } from '../modules/page_header.js';

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
