import test from 'node:test';
import assert from 'node:assert/strict';
import { createHint } from '../modules/ui_hint.js';
import { stubDocument } from './composer_controls_fixture.js';

const mounted = (options = {}) => {
    const doc = stubDocument();
    const host = doc.createElement('span');
    const control = doc.createElement('button');
    host.append(control);
    return { doc, host, control, hint: createHint(host, { doc, ...options }) };
};

test('by default only its consumer hides a hint: an outside press leaves it, and a scroll is heard only while it stands', () => {
    const { doc, hint } = mounted();
    assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [0, 0]);
    hint.say({ text: 'What this value means' });
    hint.show();
    assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [0, 1]);
    // A capsule's floating list is outside its host and still its own (capsule.js `parts`): the capsule decides.
    doc.fire('pointerdown', { target: doc.createElement('div'), pointerType: 'touch' });
    assert.equal(hint.el.hidden, false);
    hint.hide();
    hint.hide();
    assert.deepEqual([hint.el.hidden, doc.count('scroll')], [true, 0]);
    hint.show();
    hint.destroy();
    assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [0, 0]);
});

test('with dismissOutside an outside press dismisses a hint, an inside action stays reachable, and disposal removes the listeners', () => {
    let actions = 0;
    const { doc, host, control, hint } = mounted({ onAction: () => actions++, dismissOutside: true });
    assert.equal(doc.count('pointerdown'), 0);
    hint.say({ text: 'Why this control is held', action: 'Learn more' });
    const action = hint.el.querySelector('.ui-hint-action');
    hint.show();
    assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [1, 1]);
    for (const target of [control, hint.el, action]) {
        doc.fire('pointerdown', { target, pointerType: 'touch' });
        assert.equal(hint.el.hidden, false);
    }
    for (const fn of action.listeners.get('click')) fn();
    assert.equal(actions, 1);
    for (const pointerType of ['touch', 'mouse', 'pen']) {
        hint.show();
        doc.fire('pointerdown', { target: doc.createElement('div'), pointerType });
        assert.equal(hint.el.hidden, true);
        assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [0, 0]);
    }
    hint.show();
    hint.hide();
    assert.equal(hint.el.hidden, true);
    hint.show();
    hint.destroy();
    assert.deepEqual([doc.count('pointerdown'), doc.count('scroll')], [0, 0]);
    assert.equal(host.querySelector('.ui-hint'), null);
});
