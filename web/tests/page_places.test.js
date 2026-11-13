// What the owner scrolled keeps its place while hidden (page_places.js): a box shown again after `display: none`
// starts at the top in WebKit, so it is put back when its page, Dashboard tab or room is shown again. A room's feed
// and the Logs list keep their own place and are left alone.
import assert from 'node:assert/strict';
import test from 'node:test';

import { initPagePlaces } from '../modules/page_places.js';

function world() {
    const listeners = { doc: new Map(), win: new Map() };
    const on = (where) => (type, fn) => listeners[where].set(type, fn);
    const pages = new Map();
    const doc = { addEventListener: on('doc'), getElementById: (id) => pages.get(id) || null };
    const win = { addEventListener: on('win') };
    const restore = initPagePlaces({ doc, win });
    const box = (root, { owned = false } = {}) => {
        const node = { nodeType: 1, isConnected: true, scrollTop: 0, scrollLeft: 0, matches: () => owned };
        root.boxes.add(node);
        return node;
    };
    const page = (id) => {
        const root = { boxes: new Set(), contains: (node) => root.boxes.has(node) };
        pages.set(id, root);
        return root;
    };
    const scrolled = (node, top, left = 0) => {
        node.scrollTop = top;
        node.scrollLeft = left;
        listeners.doc.get('scroll')({ target: node });
    };
    return { listeners, page, box, scrolled, restore };
}

test('a box scrolled on a page goes back when the page or its tab is shown again; another page is untouched', () => {
    const w = world();
    const settings = w.page('page-settings'), tab = w.page('dashboard-panel-activity');
    const list = w.box(settings), panel = w.box(tab);
    w.scrolled(list, 400, 12);
    w.scrolled(panel, 90);
    list.scrollTop = 0; list.scrollLeft = 0; panel.scrollTop = 0;  // hidden and shown: WebKit starts at the top
    w.listeners.win.get('ouro:page-shown')({ detail: { page: 'settings' } });
    assert.deepEqual([list.scrollTop, list.scrollLeft, panel.scrollTop], [400, 12, 0]);
    w.listeners.win.get('ouro:dashboard-subtab-shown')({ detail: { tab: 'activity' } });
    assert.equal(panel.scrollTop, 90);
});

test('a kept room\'s inner box returns with the room; its feed and the Logs list own their place; a removed box is forgotten', () => {
    const w = world();
    const room = w.page('room');
    const timeline = w.box(room), feed = w.box(room, { owned: true });
    w.scrolled(timeline, 30);
    w.scrolled(feed, 700);
    timeline.scrollTop = 0; feed.scrollTop = 0;
    w.restore(room);
    assert.deepEqual([timeline.scrollTop, feed.scrollTop], [30, 0], 'the feed is its reading owner\'s');
    timeline.isConnected = false;
    timeline.scrollTop = 0;
    w.restore(room);
    assert.equal(timeline.scrollTop, 0, 'gone from the page: nothing to put back');
    assert.doesNotThrow(() => w.restore(null));
});

test('disposal releases a room\'s places even before another page restore', () => {
    const w = world(), room = w.page('room');
    const box = w.box(room);
    w.scrolled(box, 120);
    w.restore.forget(room);
    box.scrollTop = 0;
    w.restore(room);
    assert.equal(box.scrollTop, 0);
});

test('Logs owns its actual scroller; generic page restoration still restores other boxes', () => {
    const w = world(), tab = w.page('dashboard-panel-logs');
    // Match the real IDs against the production selector rather than pre-answering `matches` as owned.
    const byId = (id) => ({ nodeType: 1, isConnected: true, scrollTop: 0, scrollLeft: 0,
        matches: (selector) => selector.split(',').some(part => part.trim() === `#${id}`) });
    const logs = byId('page-logs'), other = byId('activity-list');
    tab.boxes.add(logs); tab.boxes.add(other);
    w.scrolled(logs, 900); w.scrolled(other, 40);
    logs.scrollTop = 0; other.scrollTop = 0;
    w.listeners.win.get('ouro:dashboard-subtab-shown')({ detail: { tab: 'logs' } });
    assert.deepEqual([logs.scrollTop, other.scrollTop], [0, 40], 'only logs.js restores #page-logs');
});

// A repaint keeps the reading place: WebKit drops a box in a size container to its top when a part drawn again holds
// an <svg> (page_places.js). The change is seen after it lands, before the frame is painted.
function repaintWorld({ element: platform = false } = {}) {
    let notify = null;
    const scrollListeners = [];
    // With `element`, boxes stand on an Element whose `scrollTop` the platform owns (`raw`), as a browser's do.
    const Element = platform ? class {} : null;
    if (Element) {
        Object.defineProperty(Element.prototype, 'scrollTop', { configurable: true,
            get() { return this.raw; }, set(value) { this.raw = Math.max(0, value); } });
        Element.prototype.scrollTo = function scrollTo({ top }) { this.scrollTop = top; };
    }
    const element = (parent, { container = false, height = 2000, owned = false } = {}) => {
        const node = Object.assign(Element ? Object.create(Element.prototype) : { scrollTop: 0 }, { nodeType: 1,
            parentElement: parent, isConnected: true, scrollLeft: 0, scrollHeight: height, clientHeight: 500, container,
            contains(other) { for (let n = other; n; n = n.parentElement) if (n === this) return true; return false; },
            closest: () => (owned ? node : null), matches: () => owned });
        if (Element) node.raw = 0;
        return node;
    };
    const content = element(null);
    const doc = { body: content, getElementById: () => null, addEventListener: (type, fn) => scrollListeners.push(fn) };
    const win = { Element, addEventListener() {}, getComputedStyle: (el) => ({ containerType: el.container ? 'inline-size' : 'normal' }),
        MutationObserver: class { constructor(fn) { notify = fn; } observe(root, options) { assert.equal(root, content); assert.ok(options.subtree); } } };
    initPagePlaces({ doc, win });
    const scrolled = (box, top) => { box.scrollTop = top; scrollListeners.forEach((fn) => fn({ target: box })); };
    return { content, element, scrolled, changed: (...targets) => notify(targets.map((target) => ({ target }))) };
}

test('a box in a size container that a repaint dropped to its top goes back where the owner left it', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const box = w.element(page);
    const part = w.element(box);
    w.scrolled(box, 480);
    box.scrollTop = 0;  // WebKit: the part drawn again held an <svg>
    w.changed(part);
    assert.equal(box.scrollTop, 480);
});

test('a repaint leaves alone a box the owner left at its top, one that cannot reach its place, and one outside a container', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const top = w.element(page), short = w.element(page, { height: 600 }), plain = w.element(w.content);
    w.scrolled(top, 0); w.scrolled(short, 300); w.scrolled(plain, 300);
    short.scrollTop = 0; plain.scrollTop = 0;  // what it holds now ends above 300; a page sent its own box to the top
    w.changed(w.element(top), w.element(short), w.element(plain));
    assert.deepEqual([top.scrollTop, short.scrollTop, plain.scrollTop], [0, 0, 0]);
});

test('a repaint that moved a box only part of the way, or not at all, is not undone; a feed is its owner\'s', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const box = w.element(page), feed = w.element(page, { owned: true });
    w.scrolled(box, 480);
    box.scrollTop = 200;  // a page that moved its box on purpose (a search to its field)
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 200);
    feed.scrollTop = 0;
    w.changed(w.element(feed));
    assert.equal(feed.scrollTop, 0);
});

test('a page that moves a remembered box itself moves its place: a repaint in the same moment keeps that move', () => {
    const w = repaintWorld({ element: true });
    const page = w.element(w.content, { container: true });
    const box = w.element(page);
    w.scrolled(box, 480);
    box.scrollTop = 0;  // a new circle's form, shown from the top, drawn in the same moment
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 0);
    box.scrollTo({ top: 300 });
    box.raw = 0;  // WebKit: then a repaint drops it
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 300);
});

test('a change outside a size container costs the box no layout read', () => {
    const w = repaintWorld({ element: true });
    const plain = w.element(w.content);
    w.scrolled(plain, 300);
    let reads = 0;
    Object.defineProperty(plain, 'scrollHeight', { get() { reads += 1; return 2000; } });
    plain.raw = 0;  // at its top after a change, as a dropped box would be
    w.changed(w.element(plain));
    assert.deepEqual([reads, plain.scrollTop], [0, 0]);
});
