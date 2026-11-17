// The capsule machine's `parts` option (capsule.js): what stands outside a capsule's element and is still its own
// (Model's floating suggestions). A press there is a press in the capsule, the pointer there is still at the capsule,
// and focus there holds a strip the keyboard opened; without the option, and for every other node, nothing changes.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createCapsule } from '../modules/capsule.js';
import { ElementStub } from './chat_dom_fixture.js';
import { hoverWindow, sleep, stubDocument } from './composer_controls_fixture.js';

function capsule({ parts, hover = true } = {}) {
    const doc = stubDocument();
    const row = doc.createElement('div');
    row.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 });  // not laid out: no fit
    const el = doc.createElement('div');
    el.innerHTML = '<button class="chat-capsule-head"></button><span class="chat-capsule-glyph"></span>'
        + '<span class="chat-capsule-name"></span><span class="chat-capsule-value"></span><div class="chat-capsule-body"></div>';
    row.appendChild(el);
    const strip = doc.createElement('div');
    el.querySelector('.chat-capsule-body').append(strip);
    // The floating part: a box well away from the control (the stub control is x 0..100, y 0..20).
    const panel = doc.createElement('div');
    panel.getBoundingClientRect = () => ({ top: 200, bottom: 400, left: 300, right: 500, width: 200, height: 200 });
    const inside = doc.createElement('div');
    panel.appendChild(inside);
    const control = createCapsule({ el, row, kind: 'test', strip, segs: [], render() {}, focusTarget: () => null,
        ...(parts === false ? {} : { parts: () => [parts === 'gone' ? null : panel] }), win: hoverWindow(hover), doc });
    const fire = (node, type, event) => { for (const fn of node.listeners.get(type) || []) fn(event); };
    return { control, doc, el, panel, inside, head: el.querySelector('.chat-capsule-head'),
        press: () => fire(el.querySelector('.chat-capsule-head'), 'click', { detail: 1 }),
        enter: () => fire(el, 'pointerenter', { pointerType: 'mouse' }),
        leave: (x, y) => fire(el, 'pointerleave', { pointerType: 'mouse', clientX: x, clientY: y }),
        move: (x, y) => doc.fire('pointermove', { pointerType: 'mouse', clientX: x, clientY: y }),
        down: (target) => doc.fire('pointerdown', { target }) };
}

test('a press on a part is a press in the capsule; a press anywhere else still closes it', () => {
    const c = capsule();
    c.press();
    c.down(c.inside);
    assert.equal(c.control.isOpen(), true, 'the part is the capsule\'s own');
    c.down(new ElementStub('div', c.doc));
    assert.equal(c.control.isOpen(), false);
    // Without the option the same node is outside, as it always was.
    const plain = capsule({ parts: false });
    plain.press();
    plain.down(plain.inside);
    assert.equal(plain.control.isOpen(), false);
    // A part that is not there (the list not open) is nothing.
    const gone = capsule({ parts: 'gone' });
    gone.press();
    gone.down(gone.inside);
    assert.equal(gone.control.isOpen(), false);
    for (const made of [c, plain, gone]) made.control.destroy();
});

test('a mouse resting on a part keeps a strip hover opened; leaving the part closes it after the delay', async () => {
    const c = capsule();
    c.enter();
    await sleep(220);
    assert.equal(c.control.isOpen(), true, 'hover opened it');
    c.leave(350, 300);  // straight onto the part
    await sleep(380);
    assert.equal(c.control.isOpen(), true, 'the pointer is still at the capsule');
    c.move(503, 300);
    await sleep(380);
    assert.equal(c.control.isOpen(), true, 'a few px past its edge are forgiven, as at the control');
    c.move(700, 300);
    await sleep(380);
    assert.equal(c.control.isOpen(), false, 'away from all of it: closed');
    c.control.destroy();

    const plain = capsule({ parts: false });
    plain.enter();
    await sleep(220);
    plain.leave(350, 300);
    await sleep(380);
    assert.equal(plain.control.isOpen(), false, 'without the option that place is outside');
    plain.control.destroy();
});

test('focus the keyboard put on a part holds the strip as focus in the capsule does', async () => {
    const c = capsule();
    c.enter();
    await sleep(220);
    c.doc.fire('keydown', { key: 'Tab', defaultPrevented: false, preventDefault() {} });
    c.doc.activeElement = c.inside;
    c.leave(700, 300);
    await sleep(380);
    assert.equal(c.control.isOpen(), true);
    c.doc.activeElement = new ElementStub('div', c.doc);
    c.control.closeSoon();
    await sleep(380);
    assert.equal(c.control.isOpen(), false);
    c.control.destroy();
});
