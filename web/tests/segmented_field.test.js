import test from 'node:test';
import assert from 'node:assert/strict';

import { renderSegmentedField } from '../modules/page_header.js';
import { bindEffortSegments, syncEffortSegments } from '../modules/settings_controls.js';

const labels = (count) => Array.from({ length: count }, (_, index) => ({ value: `v${index}`, label: `Choice ${index}` }));

test('the strip is markup alone: no hover title, no inline style, no count; the binder\'s hooks stay', () => {
    for (const count of [1, 2, 3, 5, 8]) {
        const html = renderSegmentedField({ target: 's-x', modifier: 'data-review-cycles-group', options: labels(count) });
        assert.doesNotMatch(html, /\stitle=/, 'a fact the owner needs is never a hover title');
        assert.doesNotMatch(html, /\sstyle=/, 'the stylesheet lays the strip out; nothing is written inline');
        assert.doesNotMatch(html, /--segment-count/);
        // What bindEffortSegments reads: the group, its target input and each choice's value.
        assert.match(html, /<div class="ui-segment-group" data-effort-group data-review-cycles-group data-effort-target="s-x">/);
        assert.deepEqual([...html.matchAll(/<button type="button" class="ui-segment" data-effort-value="(v\d+)">Choice \d+<\/button>/g)]
            .map((match) => match[1]), labels(count).map(({ value }) => value));
    }
    // The sentence a caller once hid in the title has no attribute left to land in.
    const html = renderSegmentedField({ target: 's-x', title: 'Changes at once', options: labels(2) });
    assert.doesNotMatch(html, /Changes at once/);
    assert.throws(() => renderSegmentedField({ options: labels(2) }), /requires target/);
});

/* The smallest DOM the binder touches: a labelled hidden input and a group of buttons. */
function mount(values, initial) {
    const input = { value: initial, dataset: {}, previousElementSibling: { tagName: 'LABEL', id: '' } };
    const buttons = values.map((value) => {
        const classes = new Set();
        const listeners = {};
        return {
            dataset: { effortValue: value }, attrs: {}, classes,
            classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)) },
            setAttribute(key, value_) { this.attrs[key] = value_; },
            addEventListener: (type, fn) => { listeners[type] = fn; },
            click() { listeners.click(); },
        };
    });
    const group = {
        dataset: { effortTarget: 's-inherit-choice' }, attrs: {},
        setAttribute(key, value) { this.attrs[key] = value; },
        querySelectorAll: () => buttons,
    };
    const root = {
        querySelectorAll: () => [group],
        querySelector: (selector) => (selector === '#s-inherit-choice' ? input : null),
    };
    bindEffortSegments(root);
    const button = (value) => buttons.find((candidate) => candidate.dataset.effortValue === value);
    return { root, input, group, button, buttons };
}

test('an empty inherit choice is a real value: high -> the empty choice saves ""', () => {
    const s = mount(['', 'none', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'], 'high');
    assert.equal(s.button('high').attrs['aria-pressed'], 'true');
    s.button('').click();
    assert.equal(s.input.value, '');
    assert.equal(s.input.dataset.effortTouched, '1');
    assert.equal(s.button('').attrs['aria-pressed'], 'true');
    assert.equal(s.button('high').attrs['aria-pressed'], 'false');
    assert.ok(s.button('').classes.has('active'));
    s.button('low').click();
    s.button('').click();
    assert.equal(s.input.value, '');
    // A reload of the saved '' marks the inherit choice, not a stale button.
    s.input.value = '';
    syncEffortSegments(s.root);
    assert.deepEqual(s.buttons.filter((candidate) => candidate.classes.has('active')).map((b) => b.dataset.effortValue), ['']);
    assert.equal(s.group.attrs.role, 'group');
});
