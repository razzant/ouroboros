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

test('its words live in a body that carries the shared cue; the shell keeps the material', () => {
    const { hint } = mounted();
    const body = hint.el.querySelector('.ui-hint-body');
    assert.ok(body, 'the hint has a body for its words');
    assert.equal(body.classList.contains('scroll-fade-y'), true, 'the body carries the shared cue');
    assert.equal(hint.el.classList.contains('scroll-fade-y'), false, 'the shell carries no cue of its own');
    assert.equal(hint.el.classList.contains('scroll-fade-focus-frame'), true, 'a keyboard in the words rings the shell');
    for (const part of ['.ui-hint-word', '.ui-hint-text', '.ui-hint-rule', '.ui-hint-action']) {
        assert.ok(body.querySelector(part), `${part} lives in the body`);
    }
});

test('the room it needs is measured from the words themselves, so a cramped hint does not breathe', () => {
    const doc = stubDocument();
    doc.documentElement = { clientWidth: 800, clientHeight: 600 };
    doc.defaultView = {
        getComputedStyle: () => ({ direction: 'ltr', getPropertyValue: (name) => (
            name === 'padding-top' || name === 'padding-bottom' ? '8px' : '') }),
        requestAnimationFrame: () => 1, cancelAnimationFrame() {},
        addEventListener() {}, removeEventListener() {},
    };
    const host = doc.createElement('button');
    host.getBoundingClientRect = () => ({ top: 400, bottom: 420, left: 0, right: 100 });
    const hint = createHint(host, { doc });
    hint.say({ text: 'words' });
    const body = hint.el.querySelector('.ui-hint-body');
    body.scrollHeight = 500;  // the words are taller than the room they are given
    hint.show();
    assert.equal(hint.el.dataset.cramped, 'true', 'the hint knows it does not fit');
    assert.equal(hint.el.dataset.side, 'above', 'the room above is the bigger one');
    // The same words stay over the room: the height comes from the words, not from the room they were given.
    hint.say({ text: 'more words' });
    assert.equal(hint.el.dataset.cramped, 'true', 'the hint keeps its room instead of growing into it');
});

test('focus on its way into its words keeps a hint its consumer hides on focusout; focus leaving the hint lets it go', () => {
    const { doc, host, control, hint } = mounted({ dismissOutside: true });
    hint.say({ text: 'Why this control is held' });
    hint.show();
    const body = hint.el.querySelector('.ui-hint-body');
    const elsewhere = doc.createElement('button');
    // A focusout reaches the host's captured listeners, then its consumer (subagents_settings.js hides the hint on its
    // host's focusout), then the document.
    const focusout = (target, relatedTarget) => {
        for (const fn of host.listeners.get('focusout') || []) fn({ target, relatedTarget });
        hint.hide();
        doc.fire('focusout', { target, relatedTarget });
    };
    focusout(control, body);
    assert.equal(hint.el.hidden, false, 'the words taking the focus stay');
    focusout(body, elsewhere);
    assert.equal(hint.el.hidden, true, 'focus leaving the hint lets its consumer hide it');
    hint.show();
    focusout(control, elsewhere);
    assert.equal(hint.el.hidden, true, 'focus going anywhere else hides it as before');
    hint.show();
    for (const fn of host.listeners.get('focusout') || []) fn({ target: control, relatedTarget: body });
    hint.destroy();  // a consumer disposing it while focus is on its way in
    assert.deepEqual([hint.el.hidden, host.querySelector('.ui-hint')], [true, null], 'disposal removes it all the same');
});
