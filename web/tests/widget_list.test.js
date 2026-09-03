import test from 'node:test';
import assert from 'node:assert/strict';

import {
    planWidgetListPatch,
    widgetCardSignature,
    widgetKey,
    widgetTabsSignature,
} from '../modules/widget_list.js';

function tab(overrides = {}) {
    return {
        key: 'demo:main',
        skill: 'demo',
        tab_id: 'main',
        title: 'Demo',
        icon: '',
        ws_prefix: 'demo:',
        render: { kind: 'module', entry: 'widget.js', start: 'manual', height: 480 },
        span: 1,
        grid_span: 1,
        revision: 'a'.repeat(64),
        ...overrides,
    };
}

test('widgetKey prefers the server key and falls back to skill:tab_id', () => {
    assert.equal(widgetKey(tab()), 'demo:main');
    assert.equal(widgetKey({ skill: 's', tab_id: 't' }), 's:t');
});

test('list signature ignores card order and serializer key order', () => {
    const a = tab({ key: 'demo:a', tab_id: 'a' });
    const b = tab({ key: 'demo:b', tab_id: 'b' });
    assert.equal(widgetTabsSignature([a, b]), widgetTabsSignature([b, a]));
    const reordered = { ...a, render: { height: 480, start: 'manual', entry: 'widget.js', kind: 'module' } };
    assert.equal(widgetCardSignature(a), widgetCardSignature(reordered));
    assert.equal(widgetTabsSignature([]), '');
    assert.equal(widgetTabsSignature(null), '');
});

test('card signature tracks every fact the mount consumes plus the revision', () => {
    const base = widgetCardSignature(tab());
    assert.notEqual(widgetCardSignature(tab({ revision: 'b'.repeat(64) })), base);
    assert.notEqual(widgetCardSignature(tab({ render: { ...tab().render, start: 'auto' } })), base);
    assert.notEqual(widgetCardSignature(tab({ render: { ...tab().render, height: 640 } })), base);
    assert.notEqual(widgetCardSignature(tab({ span: 2 })), base);
    assert.notEqual(widgetCardSignature(tab({ title: 'Renamed' })), base);
    assert.notEqual(widgetCardSignature(tab({ icon: '🧩' })), base);
    assert.notEqual(widgetCardSignature(tab({ ws_prefix: 'other:' })), base);
    // `grid_span` is the legacy alias of `span`: same width, same signature.
    assert.equal(widgetCardSignature(tab({ span: undefined, grid_span: 1 })), base);
    // Missing optional fields normalize instead of throwing.
    assert.equal(typeof widgetCardSignature({ skill: 's', tab_id: 't' }), 'string');
});

test('patch plan is keyed: added, removed, and own-entry-changed cards only', () => {
    const a = tab({ key: 'demo:a', tab_id: 'a' });
    const b = tab({ key: 'demo:b', tab_id: 'b' });
    const c = tab({ key: 'demo:c', tab_id: 'c' });
    assert.deepEqual(planWidgetListPatch([a, b], [b, a]), { added: [], changed: [], removed: [] });
    assert.deepEqual(
        planWidgetListPatch([a, b], [a, c]),
        { added: ['demo:c'], changed: [], removed: ['demo:b'] },
    );
    const bumped = { ...a, revision: 'f'.repeat(64) };
    assert.deepEqual(
        planWidgetListPatch([a, b], [bumped, b]),
        { added: [], changed: ['demo:a'], removed: [] },
    );
    assert.deepEqual(planWidgetListPatch(null, [a]), { added: ['demo:a'], changed: [], removed: [] });
    assert.deepEqual(planWidgetListPatch([a], []), { added: [], changed: [], removed: ['demo:a'] });
});
