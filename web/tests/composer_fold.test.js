// The row of capsules folds while the owner reads back through a room's history (composer_fold.js, owner 19.10 2A):
// a run of reading back folds it and ☰ lands on the field's last line beside "+", whatever the folded field's height and
// a reply staged inside the field; the way down, the latest, the field's focus, an open capsule, another layout and the end of
// the room bring it back; a focused field keeps it; history paged in or let go, rows arriving or growing and a hidden
// typing row move nothing, and a press on another control is that control's. Unless the owner's own gesture took the room
// off its newest row (not a swipe begun in a card's frame, which the feed never sees, nor the app's own navigation such
// as the waiting pill's jump) no scroll is the owner reading back.
// Real frames are in the browser lane (tests/test_composer_fold_browser.py).
import assert from 'node:assert/strict';
import test from 'node:test';
import { FOLD, createComposerFold } from '../modules/composer_fold.js';

function rig({ stack = true } = {}) {
    const timers = new Map();
    let timerId = 0;
    const listeners = new Map();
    const on = (target) => (type, fn) => listeners.set(`${target}:${type}`, fn);
    const off = (target) => (type, fn) => { if (listeners.get(`${target}:${type}`) === fn) listeners.delete(`${target}:${type}`); };
    // The newest row lies 90px under the visible bottom: the owner has read back a little already.
    const rowAt = (bottom) => ({ bottom, getClientRects: () => [1], getBoundingClientRect() { return { bottom: this.bottom }; } });
    const newest = rowAt(900);
    const rows = [newest];  // every row a scroll moves
    const typing = { getClientRects: () => [], getBoundingClientRect: () => ({ bottom: 0 }), previousElementSibling: newest };
    const feed = {
        lastElementChild: typing, scrollHeight: 6000, clientHeight: 810, scrollTop: 5190,
        getBoundingClientRect: () => ({ bottom: 810 }), addEventListener: on('feed'), removeEventListener: off('feed'),
    };
    // The field's box (.chat-field) lies the wrap's gap (8px) under the line; a staged reply's strip makes it taller.
    const field = { top: 725, height: 52, getBoundingClientRect() { return { top: this.top, height: this.height }; },
        get offsetHeight() { return this.height; } };
    // "+" rides on the field's last line: its centre 24px above the field's bottom, whatever the field holds.
    const plus = { getBoundingClientRect: () => ({ top: field.top + field.height - 24 - 18, height: 36 }) };
    const input = { name: 'textarea' };  // the field the owner writes in
    const menu = { getBoundingClientRect: () => ({ top: 685, height: 32 }), contains: (el) => el === menu };
    const swarm = { name: 'swarm' };  // a capsule of the row, hidden while it is folded
    const props = new Map();
    // The row's line as it stands: its box and its bottom margin (both mid-motion while the line comes back).
    const line = { top: 685, height: 32, margin: 0 };
    const row = {
        dataset: {}, open: false,
        getBoundingClientRect: () => ({ top: line.top, height: line.height }),
        contains: (el) => el === menu || el === swarm,
        querySelector(selector) {
            if (selector.includes('composer-menu')) return menu;
            if (selector.includes('data-open')) return this.open ? {} : null;
            return null;
        },
    };
    const wrap = {
        dataset: {}, offsetHeight: 120, inside: new Set([field]),
        style: { setProperty: (k, v) => props.set(k, v), removeProperty: (k) => props.delete(k) },
        contains(el) { return this.inside.has(el); },
        querySelector: (selector) => ({ '.chat-field': field, '[data-composer-plus]': plus, textarea: input })[selector] || null,
        addEventListener: on('wrap'), removeEventListener: off('wrap'),
    };
    const keys = new Set();
    const doc = { activeElement: null, body: { classList: { contains: (c) => keys.has(c) } } };
    const observers = [];
    const win = {
        requestAnimationFrame: (fn) => { fn(); return 0; },
        cancelAnimationFrame: () => {},
        setTimeout: (fn, ms) => { timerId += 1; timers.set(timerId, fn); return timerId; },
        clearTimeout: (id) => { timers.delete(id); },
        getComputedStyle: (el) => (el === row ? { marginBottom: `${line.margin}px` } : { rowGap: '8px' }),
        ResizeObserver: class {
            constructor(fn) { this.fn = fn; this.targets = new Set(); observers.push(this); }
            observe(el) { this.targets.add(el); }
            disconnect() { this.targets.clear(); }
        },
    };
    // A layout pass changed `el`'s size: its observers hear of it.
    const resize = (el) => { for (const o of observers) if (o.targets.has(el)) o.fn([{ target: el }]); };
    const env = { stack, layout: null, readingBack: true };
    const fold = createComposerFold({ feed, wrap, row, win, doc, isStack: () => env.stack,
        onLayout: (fn) => { env.layout = fn; return () => { env.layout = null; }; }, readingBack: () => env.readingBack });
    // The owner's finger moves the feed: reading back brings every row further down and the scroll off its end.
    const scroll = (down, by = 30) => {
        for (const r of rows) r.bottom += down ? by : -by;
        feed.scrollTop -= down ? by : -by;
        listeners.get('feed:scroll')?.();
    };
    // A row becomes the newest one with a box (a message or a newer page arriving, the newest page let go), and the
    // feed's content changes by `grow` px; no scroll is the owner's.
    const newestIs = (r, grow) => { rows.push(r); typing.previousElementSibling = r; feed.scrollHeight += grow; };
    const runTimers = () => { for (const [id, fn] of [...timers]) { timers.delete(id); fn(); } };
    return { fold, feed, wrap, row, line, field, input, menu, swarm, newest, rowAt, newestIs, props, doc, keys, env,
        listeners, scroll, runTimers, resize, observers };
}

test('reading back folds the row, ☰ lands on the field\'s last line, and the way down brings it back', () => {
    const r = rig();
    r.scroll(true);  // the first read is the baseline
    r.scroll(true);
    assert.equal('folded' in r.wrap.dataset, false, 'one step of 30px is not a run');
    r.scroll(true);
    assert.equal(r.fold.folded(), true, 'a run of 60px folds it');
    assert.equal('folded' in r.wrap.dataset, true);
    assert.equal(r.props.get('--fold-row-h'), '32px', 'the line collapses from its own height');
    assert.equal(r.props.get('--fold-menu-dy'), `${52 - 24 - 16}px`, 'down to the field\'s last line, less its own centre on the line');
    r.scroll(false);
    assert.equal(r.fold.folded(), true, 'one step back is not a run');
    r.scroll(false);
    assert.equal(r.fold.folded(), false, 'a run toward the latest brings it back');
    assert.equal('fold' in r.wrap.dataset, true, 'and holds its height until the motion is over');
    r.runTimers();
    assert.equal('fold' in r.wrap.dataset, false);
    assert.equal(r.props.size, 0, 'then lets go of it');
    assert.ok(FOLD.run <= 60 && FOLD.away < 90, 'the fixture reads past both bounds');
});

test('folded, ☰ waits on the field\'s last line whatever its height, a reply staged inside the field included', () => {
    const r = rig();
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    assert.equal(r.props.get('--fold-menu-dy'), '12px', 'a field of one line: down to its line, less ☰\'s centre on the line');
    // Folded, the field is narrower by ☰'s place: a draft of several lines wraps into one more, its bottom staying.
    r.field.height = 75;
    r.resize(r.field);
    assert.equal(r.props.get('--fold-menu-dy'), `${75 - 24 - 16}px`, '☰ follows the field\'s last line, beside "+"');
    r.scroll(false);
    r.scroll(false);
    r.field.height = 52;
    r.resize(r.field);
    r.runTimers();
    assert.equal(r.fold.folded(), false);
    assert.equal(r.props.has('--fold-menu-dy'), false, 'back, the row is ☰\'s line again');
    // A reply staged: its strip (56px) is the field's first line, so the field is taller and still right under the line.
    r.field.height = 52 + 56;
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    assert.equal(r.props.get('--fold-menu-dy'), `${52 + 56 - 24 - 16}px`, 'past the strip, down to the last line');
    r.field.height = 75 + 56;
    r.resize(r.field);
    assert.equal(r.props.get('--fold-menu-dy'), `${75 + 56 - 24 - 16}px`);
});

test('a reply cancelled or staged while the line is still held: the next fold reads the field again', () => {
    const r = rig();
    r.field.height = 52 + 56;  // a reply staged: its strip (56px) is the field's first line
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.props.get('--fold-menu-dy'), `${52 + 56 - 24 - 16}px`);
    // × on the strip: the strip goes and the field takes the focus (chat_reply.js), which brings the row back.
    const back = () => {
        r.doc.activeElement = r.input;
        r.listeners.get('wrap:focusin')({ target: r.input });
        assert.equal(r.fold.folded(), false);
        r.doc.activeElement = null;  // and the owner leaves the field again
    };
    back();
    // Reading back before FOLD.settle (no timer runs): the line is still held and comes back mid-motion (20px of its 32,
    // a margin of -3px); the field, without the strip now, lies right under it.
    Object.assign(r.line, { height: 20, margin: -3 });
    Object.assign(r.field, { top: 685 + 20 - 3 + 8, height: 52 });
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    assert.equal('fold' in r.wrap.dataset, true, 'the line is still the held one');
    assert.equal(r.props.get('--fold-menu-dy'), `${52 - 24 - 16}px`, 'the strip is gone: down to the one line, not past a strip');
    // A reply staged inside the hold: the field is taller again.
    back();
    r.field.height = 52 + 56;
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    assert.equal(r.props.get('--fold-menu-dy'), `${52 + 56 - 24 - 16}px`, 'past the new strip, to the last line');
});

test('at the latest the row stays; a wobble, older history above and the hidden typing row move nothing', () => {
    const r = rig();
    r.feed.scrollTop = r.feed.scrollHeight - r.feed.clientHeight - 2;  // at the end
    for (let i = 0; i < 4; i += 1) { r.newest.bottom += 30; r.listeners.get('feed:scroll')(); }
    assert.equal(r.fold.folded(), false, 'at the latest the row is always there');
    r.feed.scrollTop = 4000;
    r.scroll(true, 0);
    for (let i = 0; i < 6; i += 1) r.scroll(i % 2 === 0, 20);
    assert.equal(r.fold.folded(), false, 'reading back and forth by a little never folds it');
    // Older history prepended above: the scroll position jumps, the newest row stays where it was.
    for (let i = 0; i < 3; i += 1) { r.feed.scrollTop += 900; r.feed.scrollHeight += 900; r.listeners.get('feed:scroll')(); }
    assert.equal(r.fold.folded(), false, 'history added above moves nothing');
    r.scroll(true, 50);
    assert.equal(r.fold.folded(), true, 'the reader is read through the hidden typing row, from the newest row with a box');
});

test('while the owner writes, the field holds the focus or a capsule is worked the row stays; the field\'s focus brings it back', () => {
    const r = rig();
    r.keys.add('keyboard-open');
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), false, 'writing (the keyboard up): the row stays');
    r.keys.delete('keyboard-open');
    r.doc.activeElement = r.input;
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), false, 'the field holding the focus, keyboard or not: the row stays');
    r.doc.activeElement = r.swarm;
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), false, 'a focused capsule, which the fold would hide: the row stays');
    r.doc.activeElement = r.menu;
    for (let i = 0; i < 2; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true, '☰ stays in sight folded: its focus keeps nothing');
    r.listeners.get('wrap:focusin')({ target: r.menu });
    assert.equal(r.fold.folded(), true, 'nor does a focus or press on ☰ (or ↓, 📎, Send) move the row under the finger');
    assert.equal(r.listeners.has('wrap:pointerdown'), false, 'a press elsewhere in the composer is its control\'s own');
    r.listeners.get('wrap:focusin')({ target: r.input });
    assert.equal(r.fold.folded(), false, 'the field\'s focus brings it back');
    r.doc.activeElement = null;
    r.row.dataset.ownerSolo = 'effort';
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), false, 'a strip open alone: the row stays');
    delete r.row.dataset.ownerSolo;
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
});

test('history paged in or let go, rows arriving or growing are not the owner\'s reading; the owner\'s own still is', () => {
    const r = rig();
    for (let i = 0; i < 3; i += 1) r.scroll(true, 400);
    assert.equal(r.fold.folded(), true, 'reading back deep into the history folds it');
    // The pager lets the newest page go under the screen: the newest row is now one 1000px higher, still under it.
    r.newestIs(r.rowAt(r.newest.bottom - 1000), -1000);
    assert.ok(r.newest.bottom - 1000 > 810 + 80, 'the fixture: still deep in the history');
    r.scroll(true, 10);
    assert.equal(r.fold.folded(), true, 'a page let go under the screen brings nothing back');
    r.scroll(false);
    r.scroll(false);
    assert.equal(r.fold.folded(), false, 'the way down still brings it back');
    // While the owner heads down, a message (or a newer page) arrives under the screen, then grows.
    const arrived = r.rowAt(r.newest.bottom + 2000);
    r.newestIs(arrived, 2000);
    r.scroll(false, 10);
    assert.equal(r.fold.folded(), false, 'a row arriving under the screen folds nothing');
    arrived.bottom += 160;
    r.feed.scrollHeight += 160;
    r.scroll(false, 10);
    assert.equal(r.fold.folded(), false, 'nor does a row growing (a live card, a long reply)');
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true, 'and the owner reading back folds it again');
});

test('only the owner\'s own reading back folds the row: not a frame\'s swipe, not the app\'s navigation', () => {
    const r = rig();
    // A swipe begun in a card's frame (the frame takes the touches, so the feed saw no gesture of the owner's and the room
    // still follows its newest row), or the app's own jump (the waiting pill to a question): the browser moves the feed.
    r.env.readingBack = false;
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), false, 'a run of 90px read back without the owner\'s own gesture folds nothing');
    assert.equal('fold' in r.wrap.dataset, false, 'and holds no height for a resize to lay the feed out by');
    r.env.readingBack = true;  // the owner's own gesture on the feed ends following
    for (let i = 0; i < 3; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true, 'the owner\'s own reading back folds it');
    r.env.readingBack = false;  // the waiting pill's next jump, ↓ or a Send
    r.scroll(true);
    assert.equal(r.fold.folded(), false, 'the app\'s navigation brings the row (and the pill on it) back');
});

test('only on a phone; another layout brings the row back; the end of the room clears it', () => {
    const docked = rig({ stack: false });
    for (let i = 0; i < 4; i += 1) docked.scroll(true);
    assert.equal(docked.fold.folded(), false, 'a desk never folds it');
    const r = rig();
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    r.env.layout();
    assert.equal(r.fold.folded(), false, 'a layout change brings it back');
    for (let i = 0; i < 4; i += 1) r.scroll(true);
    assert.equal(r.fold.folded(), true);
    r.fold.destroy();
    assert.equal('folded' in r.wrap.dataset || 'fold' in r.wrap.dataset, false);
    assert.equal(r.props.size, 0);
    assert.equal(['feed:scroll', 'wrap:focusin', 'wrap:pointerdown'].some((k) => r.listeners.has(k)), false, 'no listener stays');
    assert.equal(r.env.layout, null, 'nor the layout watch');
    assert.equal(r.observers.some((o) => o.targets.size), false, 'nor the field\'s observer');
});
