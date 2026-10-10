import test from 'node:test';
import assert from 'node:assert/strict';

import { renderSegmentedField } from '../modules/page_header.js';
import { bindEffortSegments, syncEffortSegments } from '../modules/settings_controls.js';

const labels = (count) => Array.from({ length: count }, (_, index) => ({ value: `v${index}`, label: `Choice ${index}` }));

test('the generator writes how many choices it rendered, once, as the layout input', () => {
    for (const count of [2, 3, 4, 5, 7, 8]) {
        const html = renderSegmentedField({ target: 's-x', options: labels(count) });
        assert.equal((html.match(/data-effort-value=/g) || []).length, count);
        assert.match(html, new RegExp(`style="--segment-count: ${count}"`));
        assert.equal((html.match(/--segment-count/g) || []).length, 1);
    }
    const html = renderSegmentedField({ target: 's-x', modifier: 'data-review-cycles-group', options: labels(5) });
    assert.match(html, /data-review-cycles-group data-effort-target="s-x" style="--segment-count: 5"/);
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
