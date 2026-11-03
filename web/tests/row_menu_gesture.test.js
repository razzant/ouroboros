// The row-menu gesture (row_menu_gesture.js), bound once per root: a right-click, Shift+F10 or the ContextMenu key
// on the focused row, a long press on touch. The menus are the callers' (chat_row_menu.test.js, the Projects in
// tests/test_status_marks_nav_browser.py); real engines and fingers are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { LONG_PRESS_MS, MOVE_TOLERANCE_PX, bindRowMenuGesture } from '../modules/row_menu_gesture.js';
import { installDomDouble } from './dom_double.js';

function fixture({ offers = () => true } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    const scroller = el('div', { className: 'scroller' });
    const root = el('div', { className: 'list' });
    const label = el('span', { className: 'label', text: 'Alpha' });
    const row = el('div', { className: 'row' }, el('button', { className: 'row-button' }, label), el('button', { className: 'kebab' }));
    const link = el('a', { attrs: { href: 'https://example.com' } });
    const other = el('div', { className: 'row' }, link);
    root.append(row, other);
    scroller.append(root);
    const elsewhere = el('div', { className: 'elsewhere' });
    doc.body.append(scroller, elsewhere);
    const opened = [];
    let openRow = null;
    const dispose = bindRowMenuGesture({
        root, rowOf: (node) => node?.closest?.('.row') || null, isOpen: (target) => openRow === target,
        open: (target, how) => { opened.push({ row: target, ...how }); if (offers(target)) openRow = target; return offers(target); },
    });
    const touch = (node, type, extra = {}) => node.emit(type, { pointerType: 'touch', isPrimary: true, pointerId: 1,
        clientX: 0, clientY: 0, ...extra });
    return { ...env, scroller, root, row, label, link, other, elsewhere, opened, dispose, touch,
        closeMenu() { openRow = null; }, close() { dispose(); env.restore(); } };
}

test('a long touch press opens the row\'s menu where the finger is, once; the tap that ends it presses nothing', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.touch(f.label, 'pointerdown', { clientX: 30, clientY: 60 });
        t.mock.timers.tick(LONG_PRESS_MS - 1);
        assert.deepEqual(f.opened, [], 'not before the press is long');
        t.mock.timers.tick(1);
        assert.deepEqual(f.opened, [{ row: f.row, point: { x: 30, y: 60 } }]);
        assert.equal(f.label.emit('contextmenu', { clientX: 30, clientY: 60 }).defaultPrevented, true,
            'the contextmenu Android also fires for the same hold is answered already');
        assert.equal(f.opened.length, 1);
        // The menu mounted under the finger: the finger's end lands there, never in the root, and is still heard.
        f.touch(f.elsewhere, 'pointerup');
        const down = f.elsewhere.emit('mousedown');
        assert.equal(down.defaultPrevented && down.stopped, true, 'a host\'s mouse events for the lift move no focus');
        const tap = f.elsewhere.emit('click');
        assert.equal(tap.defaultPrevented && tap.stopped, true, 'the lifting finger presses nothing under it');
        t.mock.timers.tick(1000);
        assert.equal(f.elsewhere.emit('click').defaultPrevented, undefined, 'a deliberate tap afterwards works');
    } finally { f.close(); }
});

test('a contextmenu fired for a hold before the long press is that press: one opening, its lifting tap swallowed', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.touch(f.label, 'pointerdown', { clientX: 30, clientY: 60 });
        t.mock.timers.tick(LONG_PRESS_MS - 50);  // Android's own hold delay can be shorter than ours
        assert.equal(f.label.emit('contextmenu', { clientX: 30, clientY: 60 }).defaultPrevented, true);
        assert.equal(f.opened.length, 1);
        t.mock.timers.tick(100);
        assert.equal(f.opened.length, 1, 'the timer does not open it a second time');
        f.touch(f.elsewhere, 'pointerup');
        const tap = f.elsewhere.emit('click');
        assert.equal(tap.defaultPrevented && tap.stopped, true, 'the lifting finger presses nothing under it');
    } finally { f.close(); }
});

test('movement past the tolerance, a scroll of the root or around it, a cancel, a mouse or a lift is no long press', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        const held = (before) => { f.touch(f.label, 'pointerdown'); before(); t.mock.timers.tick(LONG_PRESS_MS + 50); };
        held(() => f.touch(f.label, 'pointermove', { clientX: 0, clientY: MOVE_TOLERANCE_PX }));
        assert.equal(f.opened.length, 1, 'a finger that stays within the tolerance still holds');
        f.touch(f.elsewhere, 'pointerup');
        // A layer that took the finger's capture (the drawer's drag) gets its moves: they still end the press.
        held(() => f.touch(f.elsewhere, 'pointermove', { clientX: MOVE_TOLERANCE_PX + 1, clientY: 0 }));
        assert.equal(f.opened.length, 1, 'a move past the tolerance, wherever it lands');
        held(() => f.root.emit('scroll'));
        assert.equal(f.opened.length, 1, 'the root scrolled');
        held(() => f.scroller.emit('scroll'));
        assert.equal(f.opened.length, 1, 'a scroller around the root scrolled (a short window scrolls the column)');
        held(() => f.doc.body.emit('scroll', { target: f.doc }));
        assert.equal(f.opened.length, 1, 'the page scrolled');
        held(() => f.elsewhere.emit('scroll'));
        assert.equal(f.opened.length, 2, 'an unrelated scroller (a feed beside the drawer) does not cancel it');
        f.touch(f.elsewhere, 'pointerup');
        held(() => f.touch(f.label, 'pointercancel'));
        assert.equal(f.opened.length, 2, 'the browser took the finger for a scroll');
        held(() => f.touch(f.label, 'pointerup'));
        assert.equal(f.opened.length, 2, 'a short tap');
        f.label.emit('pointerdown', { pointerType: 'mouse', isPrimary: true, pointerId: 2 });
        t.mock.timers.tick(LONG_PRESS_MS + 50);
        f.touch(f.label, 'pointerdown', { isPrimary: false, pointerId: 3 });
        t.mock.timers.tick(LONG_PRESS_MS + 50);
        f.touch(f.link, 'pointerdown');
        t.mock.timers.tick(LONG_PRESS_MS + 50);
        f.touch(f.elsewhere, 'pointerdown');
        t.mock.timers.tick(LONG_PRESS_MS + 50);
        assert.equal(f.opened.length, 2, 'a held mouse button, a second finger, a link and a press outside a row');
        // Another pointer's end does not end this one's press.
        f.touch(f.label, 'pointerdown', { pointerId: 7 });
        f.touch(f.elsewhere, 'pointerup', { pointerId: 8 });
        t.mock.timers.tick(LONG_PRESS_MS);
        assert.equal(f.opened.length, 3);
    } finally { f.close(); }
});

test('a right-click opens the row\'s menu at the pointer; a link, selected words or a row that offers nothing keep the browser\'s', () => {
    const f = fixture({ offers: (row) => row !== f.other });
    try {
        const event = f.label.emit('contextmenu', { clientX: 40, clientY: 50 });
        assert.equal(event.defaultPrevented, true);
        assert.deepEqual(f.opened, [{ row: f.row, point: { x: 40, y: 50 } }]);
        assert.equal(f.label.emit('contextmenu', { clientX: 9, clientY: 9 }).defaultPrevented, true,
            'a second right-click on the row whose menu is open keeps that menu');
        assert.equal(f.opened.length, 1);
        f.closeMenu();
        assert.equal(f.link.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined, 'a link');
        f.doc.selection = { isCollapsed: false, anchorNode: f.label };
        assert.equal(f.label.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined, 'the owner\'s words');
        // A right button that selects the word under it (macOS) does so before contextmenu: that selection is the
        // gesture's own and goes with the opened menu.
        f.doc.selection = { isCollapsed: true, anchorNode: f.label };
        f.label.emit('pointerdown', { button: 2, pointerType: 'mouse', isPrimary: true, pointerId: 9 });
        let cleared = false;
        f.doc.selection = { isCollapsed: false, anchorNode: f.label, removeAllRanges() { cleared = true; } };
        assert.equal(f.label.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, true);
        assert.equal(cleared, true);
        f.closeMenu();
        f.doc.selection = null;
        const nothing = f.other.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(nothing.defaultPrevented, undefined, 'a row that offers nothing leaves the browser\'s menu');
        assert.equal(f.elsewhere.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined);
    } finally { f.close(); }
});

test('Shift+F10 and the ContextMenu key open the focused row\'s menu under the focused element; F10 alone does not', () => {
    const f = fixture();
    try {
        const button = f.row.children[0];
        button.focus();
        assert.equal(button.emit('keydown', { key: 'F10', shiftKey: true }).defaultPrevented, true);
        assert.deepEqual(f.opened, [{ row: f.row, anchor: button }]);
        f.closeMenu();
        assert.equal(button.emit('keydown', { key: 'ContextMenu' }).defaultPrevented, true);
        assert.equal(button.emit('keydown', { key: 'F10' }).defaultPrevented, undefined);
        assert.equal(f.opened.length, 2);
    } finally { f.close(); }
});

test('the disposer ends a waiting press and leaves no listener on the root or the document', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    const before = f.doc.listenerCount();
    f.touch(f.label, 'pointerdown');
    f.dispose();
    t.mock.timers.tick(LONG_PRESS_MS + 50);
    assert.deepEqual(f.opened, []);
    assert.equal(f.root.listenerCount(), 0);
    assert.equal(f.doc.listenerCount(), before);
    assert.equal([...f.doc.captures.values()].reduce((n, set) => n + set.size, 0), 0);
    f.restore();
});
