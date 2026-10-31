// The sent draft rises into the feed (send_flight.js, owner 20.10 2A): the draft's place is read from the field before
// it empties; a copy of the sent bubble, stripped of what identifies the row and inert, stands where the draft stood
// (its words on the draft's words, in either direction) and glides to the row's box, turning from the field's surface,
// corner and shadow into the bubble's, while the row hides and shows as the copy lands; the copy follows the row if it
// moves under the flight (a keyboard going down) and lets go at once when the room goes away (the drawer, a closed
// Project); it leaves and the row is restored when the motion ends. Reduced motion, a hidden page, no draft or no row,
// and a row outside its feed's view or gone before the flight mean no flight.
// Real frames are in the browser lane (tests/test_composer_field_browser.py).
import assert from 'node:assert/strict';
import test from 'node:test';
import { FLIGHT, draftPlace, flySent } from '../modules/send_flight.js';

function anim(log, el, frames, opts) {
    let settle, fail;
    const a = {
        el, frames, opts, cancelled: false, playState: 'running', updates: [],
        finished: new Promise((resolve, reject) => { settle = resolve; fail = reject; }),
        effect: { setKeyframes: (next) => { a.frames = next; a.updates.push(next); } },
        cancel() { this.cancelled = true; this.playState = 'idle'; fail(new Error('AbortError')); },
        finish() { this.playState = 'finished'; settle(); },
    };
    a.finished.catch(() => {});
    log.push(a);
    return a;
}

function rig({ reduce = false, hidden = false, rtl = false, rowRect = { left: 120, top: 400, width: 250, height: 96 } } = {}) {
    const log = [];
    const frames = [];
    const styles = new Map();
    const look = (el, values) => { styles.set(el, values); return el; };
    const name = { style: {}, animate: (f, o) => anim(log, 'name', f, o) };
    const child = { attrs: new Set(['id']), removeAttribute(k) { this.attrs.delete(k); } };
    const ghost = {
        attrs: new Map([['class', 'chat-bubble user'], ['id', 'msg-1'], ['data-row-address', 'row:1'],
            ['data-client-message-id', 'c-1'], ['role', 'listitem']]),
        classes: new Set(['chat-bubble', 'user']), style: {}, inert: false, isConnected: false,
        getAttributeNames() { return [...this.attrs.keys()]; },
        removeAttribute(k) { this.attrs.delete(k); },
        setAttribute(k, v) { this.attrs.set(k, v); },
        classList: { add: (c) => ghost.classes.add(c) },
        querySelectorAll: (sel) => (sel === '[id]' ? [child] : []),
        querySelector: (sel) => (sel === ':scope > .sender' ? name : null),
        animate: (f, o) => anim(log, 'ghost', f, o),
        remove() { this.isConnected = false; },
    };
    const sender = look({ getBoundingClientRect: () => ({ height: 16 }) }, { marginBottom: '4px' });
    const feed = { getBoundingClientRect: () => ({ top: 60, bottom: 700 }) };
    const room = { shown: true, inert: false };
    const row = look({
        isConnected: true, style: { opacity: '' }, parentElement: feed, rect: rowRect,
        getBoundingClientRect() { return this.rect; },
        getClientRects: () => (room.shown ? [1] : []),
        closest: (sel) => (sel === '[inert]' && room.inert ? {} : null),
        querySelector: (sel) => (sel === ':scope > .sender' ? sender : null),
        cloneNode: (deep) => { assert.equal(deep, true, 'the copy carries the row\'s words'); return ghost; },
        animate: (f, o) => anim(log, 'row', f, o),
    }, { paddingLeft: '18px', paddingRight: '18px', backgroundColor: 'rgb(233, 238, 249)', borderTopLeftRadius: '18px',
        boxShadow: 'rest' });
    const input = look({ getBoundingClientRect: () => ({ left: 64, right: 302 }) },
        { paddingLeft: '4px', paddingRight: '4px', direction: rtl ? 'rtl' : 'ltr' });
    const field = look({ getClientRects: () => [1], getBoundingClientRect: () => ({ top: 640, bottom: 780, left: 16, right: 374 }) },
        { backgroundColor: 'rgb(255, 255, 255)', borderTopLeftRadius: '24px', boxShadow: 'float' });
    const body = { appended: [], append(node) { node.isConnected = true; this.appended.push(node); } };
    const doc = { hidden, body };
    const win = {
        getComputedStyle: (el) => styles.get(el) || {},
        matchMedia: (q) => ({ matches: reduce && q.includes('reduce') }),
        requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
    };
    const tick = () => { const run = frames.splice(0); for (const fn of run) fn(); };
    const settle = () => new Promise((resolve) => setImmediate(resolve));
    return { log, frames, tick, settle, ghost, child, name, row, room, input, field, feed, body, doc, win };
}

const fly = (r) => flySent({ from: draftPlace(r.field, r.input, r.win), row: r.row, win: r.win, doc: r.doc });

test('the draft\'s place is read from the field and its words, before the field empties', () => {
    const r = rig();
    assert.deepEqual(draftPlace(r.field, r.input, r.win), { top: 640, bottom: 780, left: 16, right: 374, rtl: false,
        textStart: 68, background: 'rgb(255, 255, 255)', radius: '24px', shadow: 'float' });
    const rtl = rig({ rtl: true });
    assert.equal(draftPlace(rtl.field, rtl.input, rtl.win).textStart, 298, 'right to left, the words start on the right');
    assert.equal(draftPlace({ getClientRects: () => [] }, r.input, r.win), null, 'a field not on screen has no place');
    assert.equal(draftPlace(null, r.input, r.win), null);
});

test('the copy stands on the draft, glides to the sent row as the bubble, and leaves the row shown', async () => {
    const r = rig();
    const ghost = fly(r);
    assert.equal(ghost, r.ghost);
    assert.deepEqual(r.body.appended, [r.ghost], 'the copy flies over the room, outside the feed');
    assert.deepEqual(r.ghost.getAttributeNames().sort(), ['aria-hidden', 'class', 'role'],
        'nothing that identifies the row: no id, no address, no keys');
    assert.equal(r.child.attrs.has('id'), false, 'and no id inside it');
    assert.equal(r.ghost.inert, true);
    assert.ok(r.ghost.classes.has('chat-send-flight'));
    // Its words start where the draft's did: the draft's words (64 + 4) less the bubble's inset (18).
    assert.deepEqual(r.ghost.style, { left: '50px', top: '640px', width: '324px', height: '140px',
        backgroundColor: 'rgb(255, 255, 255)', borderRadius: '24px', boxShadow: 'float' });
    assert.deepEqual(r.name.style, { overflow: 'hidden', maxHeight: '0px', marginBottom: '0px', opacity: '0' },
        'the sender line is not in the draft: it starts folded');
    assert.equal(r.row.style.opacity, '0', 'the row hides before it is painted in place');
    assert.equal(r.log.length, 0, 'nothing moves before the feed has laid the row out');
    r.tick();
    assert.equal(r.log.length, 0, 'one frame is not enough');
    r.tick();
    const flight = r.log.find((a) => a.el === 'ghost');
    const landing = r.log.find((a) => a.el === 'row');
    const grow = r.log.find((a) => a.el === 'name');
    assert.ok(flight && landing && grow, 'the copy flies, the row lands, the name grows in');
    assert.deepEqual(flight.frames[0], { left: '50px', top: '640px', width: '324px', height: '140px',
        backgroundColor: 'rgb(255, 255, 255)', borderRadius: '24px', boxShadow: 'float' });
    assert.deepEqual(flight.frames[1], { left: '120px', top: '400px', width: '250px', height: '96px',
        backgroundColor: 'rgb(233, 238, 249)', borderRadius: '18px', boxShadow: 'rest' },
        'it lands on the row\'s box as the bubble: the owner\'s voice, its corner, its shadow');
    assert.ok(!('transform' in flight.frames[1]), 'the box moves and resizes, never a scale: the words keep their size');
    assert.equal(flight.opts.duration, FLIGHT.ms);
    assert.equal(flight.opts.easing, 'cubic-bezier(0.32, 0.72, 0, 1)', 'the iOS sheet curve');
    assert.deepEqual(landing.frames.map((f) => f.opacity), [0, 0, 1], 'the row shows only as the copy lands');
    assert.equal(landing.frames[1].offset, FLIGHT.landing);
    assert.deepEqual(grow.frames[2], { maxHeight: '16px', marginBottom: '4px', opacity: 1 });
    r.tick();
    assert.deepEqual(flight.updates, [], 'a row that stays put keeps the flight as it is');
    flight.finish();
    await r.settle();
    assert.equal(r.ghost.isConnected, false, 'the copy leaves when the motion ends');
    assert.equal(landing.cancelled, true);
    assert.equal(r.row.style.opacity, '', 'and the row is itself again');
});

test('right to left, the copy\'s words start on the draft\'s right edge', () => {
    const r = rig({ rtl: true });
    fly(r);
    // The field's left edge (16) to the draft's words (302 - 4) plus the bubble's inset (18).
    assert.equal(r.ghost.style.left, '16px');
    assert.equal(r.ghost.style.width, '300px');
});

test('a row that moves under the flight (a keyboard going down) is followed to its new place', async () => {
    const r = rig();
    fly(r);
    r.tick();
    r.tick();
    const flight = r.log.find((a) => a.el === 'ghost');
    r.row.rect = { left: 120, top: 488, width: 250, height: 96 };  // the keyboard went down: the feed's end moved
    r.tick();
    assert.equal(flight.updates.length, 1, 'the flight is re-aimed once the row moved');
    assert.deepEqual(flight.frames[1], { left: '120px', top: '488px', width: '250px', height: '96px',
        backgroundColor: 'rgb(233, 238, 249)', borderRadius: '18px', boxShadow: 'rest' }, 'at where the row is now');
    assert.equal(flight.frames[0].top, '640px', 'from where the draft stood');
    r.tick();
    assert.equal(flight.updates.length, 1, 'and only when it moves');
    flight.finish();
    await r.settle();
    assert.equal(r.ghost.isConnected, false);
    r.tick();  // the frame already asked for sees the landing and stops
    assert.equal(r.frames.length, 0, 'nothing keeps watching after the landing');
});

test('a room that goes away mid-flight (the drawer over it, a Project closed) ends the flight and shows the row', async () => {
    for (const away of ['inert', 'hidden']) {
        const r = rig();
        fly(r);
        r.tick();
        r.tick();
        const flight = r.log.find((a) => a.el === 'ghost');
        if (away === 'inert') r.room.inert = true; else r.room.shown = false;
        r.tick();
        await r.settle();
        assert.equal(flight.cancelled, true, `${away}: the flight stops at once`);
        assert.equal(r.ghost.isConnected, false, `${away}: the copy leaves`);
        assert.equal(r.row.style.opacity, '', `${away}: the row is itself again`);
    }
});

test('reduced motion, a hidden page, no draft or no row: the row simply appears', () => {
    for (const opts of [{ reduce: true }, { hidden: true }]) {
        const r = rig(opts);
        assert.equal(fly(r), null);
        assert.equal(r.body.appended.length, 0);
        assert.equal(r.row.style.opacity, '', 'the row is never hidden');
        assert.equal(r.frames.length, 0);
    }
    const r = rig();
    assert.equal(flySent({ from: null, row: r.row, win: r.win, doc: r.doc }), null, 'no draft on screen');
    assert.equal(flySent({ from: draftPlace(r.field, r.input, r.win), row: undefined, win: r.win, doc: r.doc }), null,
        'a send the feed already had (no new row)');
    assert.equal(r.body.appended.length, 0);
});

test('a row outside its feed\'s view, or gone before the flight, shows without one', () => {
    const below = rig({ rowRect: { left: 120, top: 760, width: 250, height: 96 } });
    fly(below);
    below.tick();
    below.tick();
    assert.equal(below.log.length, 0, 'the owner is not at the newest end: nothing flies');
    assert.equal(below.ghost.isConnected, false);
    assert.equal(below.row.style.opacity, '');
    const gone = rig();
    fly(gone);
    gone.row.isConnected = false;  // the room closed under it
    gone.tick();
    gone.tick();
    assert.equal(gone.log.length, 0);
    assert.equal(gone.ghost.isConnected, false);
    assert.equal(gone.row.style.opacity, '');
});
