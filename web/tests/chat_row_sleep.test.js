import test from 'node:test';
import assert from 'node:assert/strict';

import { createRowSleep } from '../modules/chat_row_sleep.js';

// A feed of rows, its observers and its frames, by hand: a test fires what the browser would.
function stand() {
    const frames = [];
    const observers = { changes: null, root: null, size: null, near: null };
    const fonts = new Map();
    const doc = { fonts: { addEventListener: (type, fn) => fonts.set(type, fn), removeEventListener: (type) => fonts.delete(type) },
        documentElement: { nodeType: 1 } };
    // A row is a bubble unless told otherwise; `matches` answers for its class.
    const element = (tag = 'div', kind = tag === 'div' ? 'chat-bubble' : '') => {
        const attrs = new Map();
        const node = {
            nodeType: 1, tagName: tag.toUpperCase(), parentElement: null, children: [], ownerDocument: doc,
            listeners: new Map(), complete: true,
            matches: (selector) => Boolean(kind) && selector.split(',').some((part) => part.trim() === `.${kind}`),
            hasAttribute: (name) => attrs.has(name),
            setAttribute: (name, value) => { attrs.set(name, String(value)); },
            removeAttribute: (name) => { attrs.delete(name); },
            append(...kids) { for (const kid of kids) { kid.parentElement = node; node.children.push(kid); } },
            remove() {
                const parent = node.parentElement;
                if (parent) parent.children.splice(parent.children.indexOf(node), 1);
                node.parentElement = null;
            },
            querySelectorAll(selector) {
                const tags = selector.split(',').map((part) => part.trim());
                const found = [];
                const walk = (el) => { for (const kid of el.children) { if (tags.includes(kid.tagName.toLowerCase())) found.push(kid); walk(kid); } };
                walk(node);
                return found;
            },
            addEventListener(type, fn) { node.listeners.set(type, fn); },
            emit(type) { const fn = node.listeners.get(type); node.listeners.delete(type); fn?.(); },
        };
        return node;
    };
    const feed = element();
    feed.clientWidth = 0;
    const win = {
        requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
        MutationObserver: class {
            constructor(callback) { this.callback = callback; this.connected = false; this.pending = []; }
            observe(target, options) {
                this.target = target; this.options = options; this.connected = true;
                if (target === doc.documentElement) observers.root = this; else observers.changes = this;
            }
            takeRecords() { return this.pending.splice(0); }
            disconnect() { this.connected = false; }
        },
        ResizeObserver: class {
            constructor(callback) { this.callback = callback; observers.size = this; this.connected = false; }
            observe() { this.connected = true; }
            disconnect() { this.connected = false; }
        },
        IntersectionObserver: class {
            constructor(callback, options) { this.callback = callback; this.options = options; this.watched = new Set(); observers.near = this; }
            observe(target) { this.watched.add(target); }
            unobserve(target) { this.watched.delete(target); }
            disconnect() { this.watched.clear(); }
        },
    };
    const run = (count = 1) => { for (let i = 0; i < count; i += 1) frames.splice(0).forEach((fn) => fn()); };
    const show = (width) => { feed.clientWidth = width; observers.size.callback([]); };
    const added = (...rows) => { feed.append(...rows); observers.changes.callback([{ target: feed, addedNodes: rows }]); };
    const changed = (node) => observers.changes.callback([{ target: node, addedNodes: [] }]);
    const laid = () => feed.children.map((row) => row.hasAttribute('data-laid'));
    return { feed, win, doc, fonts, observers, element, run, show, added, changed, laid, frames };
}

test('rows sleep only once laid out at a shown feed\'s width, two frames after they arrive', () => {
    const s = stand();
    s.feed.append(s.element(), s.element());
    const sleep = createRowSleep({ feed: s.feed, win: s.win });
    s.run(5);
    assert.deepEqual(s.laid(), [false, false], 'a feed not shown lays nothing out: its rows stay awake');
    s.show(390);
    s.run(1);
    assert.deepEqual(s.laid(), [false, false], 'shown: laid out first');
    s.run(1);
    assert.deepEqual(s.laid(), [true, true]);
    const row = s.element();
    s.added(row);
    s.run(1);
    assert.equal(row.hasAttribute('data-laid'), false, 'a row that just arrived is laid out first');
    s.run(1);
    assert.equal(row.hasAttribute('data-laid'), true);
    assert.equal(s.frames.length, 0, 'nothing due: no frame is asked for');
    sleep.destroy();
    assert.equal(s.observers.changes.connected || s.observers.size.connected || s.observers.root.connected, false);
    assert.equal(s.observers.near.watched.size, 0);
    assert.equal(s.fonts.size, 0);
});

test('a row stays awake until its pictures arrive, then sleeps', () => {
    const s = stand();
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    const row = s.element(), picture = s.element('img');
    picture.complete = false;
    row.append(picture);
    s.added(row);
    s.run(4);
    assert.equal(row.hasAttribute('data-laid'), false, 'a picture still loading has no size yet');
    picture.complete = true;
    picture.emit('load');
    s.run(1);
    assert.equal(row.hasAttribute('data-laid'), false);
    s.run(1);
    assert.equal(row.hasAttribute('data-laid'), true, 'laid out with its picture, it sleeps');
});

test('a row that changes wakes at once and sleeps again once it held still', () => {
    const s = stand();
    const row = s.element(), text = s.element('p');
    row.append(text);
    s.feed.append(row);
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.run(2);
    assert.equal(row.hasAttribute('data-laid'), true);
    s.changed(text);
    assert.equal(row.hasAttribute('data-laid'), false, 'woken in the same task: it is laid out with what changed');
    s.run(20);
    s.changed(text);  // a streaming answer keeps changing: it stays awake
    s.run(29);
    assert.equal(row.hasAttribute('data-laid'), false);
    s.run(1);
    assert.equal(row.hasAttribute('data-laid'), true);
    assert.deepEqual(s.observers.changes.options.attributeFilter.includes('data-laid'), false,
        'its own mark is no change of the row');
});

test('a new width wakes every row; they sleep again once the width held and the reading owner\'s mark is gone', () => {
    const s = stand();
    s.feed.append(s.element(), s.element(), s.element());
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.run(2);
    assert.deepEqual(s.laid(), [true, true, true]);
    s.feed.setAttribute('data-resizing', '');
    s.show(380);
    assert.deepEqual(s.laid(), [false, false, false], 'each row is laid out again at the new width');
    s.run(3);
    s.show(370);
    s.run(30);
    assert.deepEqual(s.laid(), [false, false, false], 'nothing sleeps while the width moves');
    s.feed.removeAttribute('data-resizing');
    s.run(11);
    assert.deepEqual(s.laid(), [false, false, false]);
    s.run(1);
    assert.deepEqual(s.laid(), [true, true, true], 'the width held: laid out at it, the rows sleep');
    s.show(0);
    s.show(370);
    assert.deepEqual(s.laid(), [true, true, true], 'hidden and shown again at its width: nothing to lay out again');
    s.fonts.get('loadingdone')();
    assert.deepEqual(s.laid(), [false, false, false], 'new fonts: every row is laid out again');
});

test('a row added while the feed is hidden sleeps only after the feed is shown', () => {
    const s = stand();
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.show(0);
    const row = s.element();
    s.added(row);
    s.run(10);
    assert.equal(row.hasAttribute('data-laid'), false, 'never laid out: its size is not known');
    assert.equal(s.frames.length, 0, 'and no frame is asked for while hidden');
    s.show(390);
    s.run(2);
    assert.equal(row.hasAttribute('data-laid'), true);
});

test('a write settles the rows it changed at once: the reader\'s place is measured with their new size', () => {
    const s = stand();
    const row = s.element(), text = s.element('p');
    row.append(text);
    s.feed.append(row);
    const sleep = createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.run(2);
    assert.equal(row.hasAttribute('data-laid'), true);
    s.observers.changes.pending.push({ target: text, addedNodes: [] });  // written, its record not delivered yet
    assert.equal(row.hasAttribute('data-laid'), true, 'until the observer\'s microtask, the row would keep its old size');
    sleep.flush();
    assert.equal(row.hasAttribute('data-laid'), false, 'flushed inside the write: laid out before the place is restored');
    assert.equal(s.observers.changes.pending.length, 0);
});

test('another theme, layout or language wakes every row; their own changes stay theirs', () => {
    const s = stand();
    s.feed.append(s.element(), s.element());
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.run(2);
    assert.deepEqual(s.laid(), [true, true]);
    assert.deepEqual(s.observers.root.options.attributeFilter, ['data-theme', 'data-layout', 'lang', 'dir']);
    s.observers.root.callback([{ target: s.doc.documentElement, attributeName: 'data-theme' }]);
    assert.deepEqual(s.laid(), [false, false], 'the dark theme\'s recorded answer has a line the light one has not');
    s.run(2);
    assert.deepEqual(s.laid(), [true, true], 'laid out in the new theme, they sleep again');
});

test('a row near the view never sleeps, one that leaves it sleeps again; only message rows sleep', () => {
    const s = stand();
    const row = s.element(), card = s.element('div', 'chat-live-card'), control = s.element('div', 'chat-load-older');
    s.feed.append(control, row, card);
    createRowSleep({ feed: s.feed, win: s.win });
    assert.equal(s.observers.near.options.root, s.feed);
    assert.deepEqual([...s.observers.near.watched], [row, card], 'a history control is never watched');
    s.show(390);
    s.observers.near.callback([{ target: row, isIntersecting: true }]);
    s.run(2);
    assert.deepEqual(s.laid(), [false, false, true], 'near the reader a row is not contained: the browser anchors on it');
    s.observers.near.callback([{ target: card, isIntersecting: true }]);
    assert.equal(card.hasAttribute('data-laid'), false, 'coming near, a sleeping row wakes at once');
    s.observers.near.callback([{ target: row, isIntersecting: false }]);
    s.run(2);
    assert.deepEqual(s.laid(), [false, true, false], 'left behind, it sleeps; the one now near does not');
    s.run(40);
    assert.equal(control.hasAttribute('data-laid'), false, 'the history control never sleeps (its touch area reaches past it)');
});

test('without observers the module is inert, and a write may still ask it to settle', () => {
    const sleep = createRowSleep({ feed: {}, win: { requestAnimationFrame: () => 0 } });
    assert.doesNotThrow(() => { sleep.flush(); sleep.destroy(); }, 'a page without MutationObserver keeps working');
});

test('a restyle wakes every row and asks the reading owner to place the passage again', () => {
    const s = stand();
    s.feed.append(s.element(), s.element());
    const placed = [];
    createRowSleep({ feed: s.feed, win: s.win, onRestyle: () => placed.push(s.laid()) });
    s.show(390);
    s.run(2);
    s.observers.root.callback([{ target: s.doc.documentElement, attributeName: 'data-theme' }]);
    assert.deepEqual(placed, [[false, false]], 'woken first, then placed: the passage is measured at the new sizes');
    s.run(2);
    s.fonts.get('loadingdone')();
    assert.deepEqual(placed, [[false, false], [false, false]], 'new fonts too');
});

test('a row removed before its frame is never marked; the row that stays still sleeps', () => {
    const s = stand();
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    const gone = s.element(), kept = s.element();
    s.added(gone, kept);
    gone.remove();  // a history page replaced, a row superseded
    s.run(2);
    assert.equal(gone.hasAttribute('data-laid'), false, 'out of the feed, it is not marked');
    assert.deepEqual(s.laid(), [true]);
});

test('a row holding a part sized by the window\'s height never sleeps; its neighbour does', () => {
    const s = stand();
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    const chart = s.element(), plain = s.element();
    chart.querySelector = (selector) => (selector.split(',').some((part) => part.trim() === '.widget-chart-canvas') ? {} : null);
    plain.querySelector = () => null;
    s.added(chart, plain);
    s.run(40);
    assert.deepEqual(s.laid(), [false, true], 'only the window\'s height changes the chart: a size kept asleep would go stale');
});

test('a video sleeps only once its metadata has given it its size', () => {
    const s = stand();
    createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    const row = s.element(), clip = s.element('video');
    clip.readyState = 0;
    row.append(clip);
    s.added(row);
    s.run(4);
    assert.equal(row.hasAttribute('data-laid'), false, 'a video is 300x150 until its metadata arrives');
    clip.readyState = 1;
    clip.emit('loadedmetadata');
    s.run(2);
    assert.equal(row.hasAttribute('data-laid'), true);
});

test('a row without a box (a deck\'s tucked note) is never marked laid; drawn again, it sleeps like any row', () => {
    const s = stand();
    const tucked = s.element();
    let box = [];
    tucked.getClientRects = () => box;
    s.feed.append(tucked, s.element());
    const sleep = createRowSleep({ feed: s.feed, win: s.win });
    s.show(390);
    s.run(2);
    assert.deepEqual(s.laid(), [false, true], 'nothing to remember: marked, it would open at the placeholder size');
    box = [{ width: 390, height: 120 }];
    s.changed(tucked);
    s.run(40);
    assert.equal(tucked.hasAttribute('data-laid'), true, 'once it has a box it sleeps');
    sleep.destroy();
});
