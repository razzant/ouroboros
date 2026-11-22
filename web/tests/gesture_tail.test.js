// The tail of a finished gesture (gesture_tail.js): the click (and, where asked, the mousedown) a host still fires for
// the finger that swiped, dragged or held is swallowed until it arrives or the next touch, pointer or key begins; no
// timer, and nothing listens outside that span. The gestures that arm it are tested in their own files
// (chat_row_swipe, drawer_drag, chat_row_menu, row_menu_gesture); real engines are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createGestureTail } from '../modules/gesture_tail.js';
import { domDouble } from './dom_double.js';

function fixture(options) {
    const { doc, el } = domDouble();
    const plus = el('button', { className: 'plus' });
    doc.body.append(plus);
    const pressed = [];
    plus.addEventListener('click', (event) => pressed.push(event.detail));
    const listening = () => [...doc.captures.values()].reduce((n, set) => n + set.size, 0);
    return { doc, plus, pressed, listening, tail: createGestureTail(doc, options) };
}

test('nothing listens until a gesture ends; its click is swallowed once, and then nothing listens again', () => {
    const f = fixture();
    assert.equal(f.listening(), 0, 'no standing document layer');
    f.tail.arm();
    assert.equal(f.tail.armed, true);
    f.tail.arm();
    assert.equal(f.doc.captures.get('click')?.size, 1, 'arming twice adds nothing');
    const ending = f.plus.emit('click', { detail: 1 });
    assert.equal(ending.defaultPrevented && ending.stopped, true, 'the lifted finger presses nothing under it');
    assert.deepEqual(f.pressed, []);
    assert.equal(f.tail.armed, false, 'the click is the tail\'s last event');
    assert.equal(f.listening(), 0);
    assert.equal(f.plus.emit('click', { detail: 1 }).defaultPrevented, undefined);
    assert.deepEqual(f.pressed, [1]);
});

test('the next touch, pointer or key ends the tail at once: the tap it begins acts, once, wherever it lands', () => {
    const f = fixture();
    for (const [type, extra] of [['pointerdown', { pointerType: 'touch' }], ['touchstart', {}],
        ['pointerdown', { pointerType: 'mouse' }], ['pointerdown', { pointerType: 'pen' }], ['keydown', { key: 'Enter' }]]) {
        f.tail.arm();
        f.plus.emit(type, extra);
        assert.equal(f.tail.armed, false, `${type} ${extra.pointerType || extra.key || ''}`);
        assert.equal(f.listening(), 0);
        assert.equal(f.plus.emit('click', { detail: extra.key ? 0 : 1 }).defaultPrevented, undefined);
    }
    assert.equal(f.pressed.length, 5, 'each acted once');
});

test('a click no press made (a key\'s, an assistive technology\'s) or a script\'s passes, and the tail still waits', () => {
    const f = fixture();
    f.tail.arm();
    assert.equal(f.plus.emit('click', { detail: 0 }).defaultPrevented, undefined, 'Enter, Space or a screen reader');
    assert.equal(f.plus.emit('click', { isTrusted: false, detail: 1 }).defaultPrevented, undefined, 'el.click()');
    assert.equal(f.tail.armed, true, 'neither was the finger\'s');
    assert.equal(f.plus.emit('click', { detail: 1 }).defaultPrevented, true, 'the finger\'s own click still presses nothing');
    assert.equal(f.pressed.length, 2);
});

test('the mousedown only where its owner asks; it does not end the tail, its click does', () => {
    const plain = fixture();
    plain.tail.arm();
    assert.equal(plain.plus.emit('mousedown', { detail: 1 }).defaultPrevented, undefined, 'a click-only tail');
    plain.tail.disarm();
    const f = fixture({ mousedown: true });
    f.tail.arm();
    const down = f.plus.emit('mousedown', { detail: 1 });
    assert.equal(down.defaultPrevented && down.stopped, true, 'it would take the focus out of a menu just opened');
    assert.equal(f.tail.armed, true);
    assert.equal(f.plus.emit('click', { detail: 1 }).defaultPrevented, true);
    assert.equal(f.listening(), 0);
    f.plus.emit('pointerdown', { pointerType: 'mouse' });
    assert.equal(f.plus.emit('mousedown', { detail: 1 }).defaultPrevented, undefined, 'a new press keeps its own');
});

test('disarm is the disposer: idempotent, and it leaves no listener', () => {
    const f = fixture({ mousedown: true });
    f.tail.disarm();
    f.tail.arm();
    assert.ok(f.listening() > 0);
    f.tail.disarm();
    f.tail.disarm();
    assert.equal(f.listening(), 0);
    assert.equal(f.plus.emit('click', { detail: 1 }).defaultPrevented, undefined);
    assert.doesNotThrow(() => createGestureTail(null).arm(), 'no document: nothing to guard');
});
