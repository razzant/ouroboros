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

// A repaint keeps the reading place: WebKit holds a box in a size container to what remains while a part with an <svg>
// is drawn again (page_places.js). The change is seen after it lands, before the frame is painted. Boxes stand on an
// Element whose `scrollTop` the platform owns (`raw`), as a browser's do: a write is held to what the box can reach, a
// box out of sight reads 0 and takes no write, and the platform's `scrollTo` and `scrollIntoView` do not pass through
// `scrollTop`. `engine` is what the engine's one probe answers.
function repaintWorld({ engine = true } = {}) {
    let notify = null;
    let observed = false;
    let asked = 0;
    const scrollListeners = [];
    class Element {}
    const land = (box, top) => { if (box.rendered) box.raw = Math.max(0, Math.min(Number(top) || 0, box.scrollHeight - box.clientHeight)); };
    Object.defineProperty(Element.prototype, 'scrollTop', { configurable: true,
        get() { return this.rendered ? this.raw : 0; }, set(value) { land(this, value); } });
    Element.prototype.scrollTo = function scrollTo({ top }) { land(this, top); };
    Element.prototype.scrollIntoView = function scrollIntoView() { land(this.parentElement, this.at); };
    const platformIntoView = Element.prototype.scrollIntoView;
    const element = (parent, { container = false, height = 2000, owned = false, at = 0 } = {}) => {
        const node = Object.assign(Object.create(Element.prototype), { nodeType: 1, parentElement: parent, isConnected: true,
            rendered: true, raw: 0, scrollLeft: 0, scrollHeight: height, clientHeight: 500, container, owned, at,
            getClientRects() { return this.rendered ? [{}] : []; },
            contains(other) { for (let n = other; n; n = n.parentElement) if (n === this) return true; return false; },
            closest() { for (let n = this; n; n = n.parentElement) if (n.owned) return n; return null; },
            matches() { return this.owned; } });
        return node;
    };
    const content = element(null);
    const doc = { body: content, getElementById: () => content, addEventListener: (type, fn) => scrollListeners.push(fn) };
    const pageShown = [];
    const win = { Element, addEventListener: (type, fn) => { if (type === 'ouro:page-shown') pageShown.push(fn); },
        getComputedStyle: (el) => ({ containerType: el.container ? 'inline-size' : 'normal' }),
        MutationObserver: class { constructor(fn) { notify = fn; }
            observe(root, options) { assert.equal(root, content); assert.ok(options.childList && options.subtree); observed = true; } } };
    initPagePlaces({ doc, win, drops: () => { asked += 1; return engine; } });
    // The owner scrolls: the platform moves the box, then says so. The engine's drop moves it and says nothing yet.
    const scrolled = (box, top) => { box.raw = top; scrollListeners.forEach((fn) => fn({ target: box })); };
    const dropped = (box, to = 0) => { box.raw = to; };
    const changed = (...targets) => { assert.ok(observed, 'the document is watched'); notify(targets.map((target) => ({ target }))); };
    return { content, element, scrolled, dropped, changed, shown: () => pageShown.forEach((fn) => fn({ detail: { page: 'any' } })),
        watched: () => observed, asked: () => asked, intoViewIsPlatforms: () => Element.prototype.scrollIntoView === platformIntoView };
}

test('a box a repaint dropped goes back where the owner left it, in a size container only', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const box = w.element(page), under = w.element(page, { height: 4000 }), plain = w.element(w.content);
    w.scrolled(box, 480); w.scrolled(under, 2500); w.scrolled(plain, 480);
    w.dropped(box); w.dropped(plain);  // WebKit: the part drawn again held an <svg>
    w.dropped(under, 910);             // part of the way: 1410 px of cards stand above the part drawn again
    w.changed(w.element(box), w.element(under), w.element(plain));
    assert.deepEqual([box.scrollTop, under.scrollTop, plain.scrollTop], [480, 2500, 0]);
});

test('a change outside a size container costs the box no layout read; inside one the box is read and put back', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const box = w.element(page), plain = w.element(w.content);
    w.scrolled(box, 300); w.scrolled(plain, 300);
    const reads = new Map([[box, 0], [plain, 0]]);
    for (const each of [box, plain]) Object.defineProperty(each, 'scrollHeight', { get() { reads.set(each, reads.get(each) + 1); return 2000; } });
    w.dropped(box); w.dropped(plain);
    w.changed(w.element(box), w.element(plain));
    assert.equal(reads.get(plain), 0);
    assert.ok(reads.get(box) > 0);
    assert.equal(box.scrollTop, 300);
});

test('a box that lost only what it no longer holds is where it should be; one that lost more goes to its place or its new end', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const top = w.element(page), kept = w.element(page), end = w.element(page), on = w.element(page);
    w.scrolled(top, 0); w.scrolled(kept, 1000); w.scrolled(end, 1500); w.scrolled(on, 480);
    kept.scrollHeight = 1700; w.dropped(kept, 700);  // a card above the reader went: the engine kept the row, 300 up
    end.scrollHeight = 1972; w.dropped(end);         // read to its end, one row fewer, and dropped
    w.dropped(on, 700);                              // the owner scrolled on; the platform has not said so yet
    w.changed(w.element(top), w.element(kept), w.element(end), w.element(on));
    assert.deepEqual([top.scrollTop, kept.scrollTop, end.scrollTop, on.scrollTop], [0, 700, 1472, 700]);
    // Where each was left is its place now: the next drop puts it back there.
    w.dropped(kept); w.dropped(on);
    w.changed(w.element(kept), w.element(on));
    assert.deepEqual([kept.scrollTop, on.scrollTop], [700, 700]);
    // A list read to its end that loses a few rows at a time follows its end, however much it has lost in all.
    for (const height of [1700, 1440]) {
        end.scrollHeight = height; w.dropped(end);
        w.changed(w.element(end));
        assert.equal(end.scrollTop, height - 500);
    }
});

test('something else shown in the box is not sent to the old place; the same thing a little longer is', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const folder = w.element(page), list = w.element(page), grown = w.element(page);
    for (const each of [folder, list, grown]) w.scrolled(each, 480);
    folder.scrollHeight = 5000;  // another folder: much more than it held
    list.scrollHeight = 2300;    // the same list with a few more rows
    w.dropped(folder); w.dropped(list);
    w.changed(w.element(folder), w.element(list));
    assert.deepEqual([folder.scrollTop, list.scrollTop], [0, 480]);
    // A list that grows through repaints that leave it in place: each tells what it holds now.
    for (const height of [2400, 2800, 3200]) { grown.scrollHeight = height; w.changed(w.element(grown)); }
    w.dropped(grown);
    w.changed(w.element(grown));
    assert.equal(grown.scrollTop, 480);
});

test('a feed and what it holds are its owner\'s; their neighbour goes back', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const feed = w.element(page, { owned: true }), box = w.element(page);
    const inFeed = w.element(feed);  // a long code block inside a message
    w.scrolled(feed, 480); w.scrolled(inFeed, 480); w.scrolled(box, 480);
    w.dropped(feed); w.dropped(inFeed); w.dropped(box);
    w.changed(w.element(feed), w.element(inFeed), w.element(box));
    assert.deepEqual([feed.scrollTop, inFeed.scrollTop, box.scrollTop], [0, 0, 480]);
});

test('a page that moves a remembered box itself moves its place: a repaint in the same moment keeps that move', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const box = w.element(page);
    w.scrolled(box, 480);
    box.scrollTop = 0;  // a new circle's form, shown from the top, drawn in the same moment
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 0);
    box.scrollTo({ top: 300 });
    w.dropped(box);  // WebKit: then a repaint drops it
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 300);
    box.scrollTo({ top: 0 });
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 0);
    w.scrolled(box, 480);
    w.element(box, { at: 120 }).scrollIntoView();  // a found row is brought into view, above where the owner was
    w.changed(w.element(box));
    assert.equal(box.scrollTop, 120);
});

test('a write that does not land keeps the place: a box out of sight, and content that is not there yet', () => {
    const w = repaintWorld();
    const page = w.element(w.content, { container: true });
    const tab = w.element(page), away = w.element(page), late = w.element(page);
    w.scrolled(tab, 300); w.scrolled(away, 300); w.scrolled(late, 300);
    tab.rendered = false;                       // another tab is chosen: this one is out of sight
    away.rendered = false;                      // its page was left
    late.scrollHeight = 500; late.raw = 0;      // its rows are being read again
    w.shown();                                  // the page is shown again: every box under it is asked back
    away.scrollTop = away.scrollTop;            // a page drawing again while away puts back what it read: 0
    assert.deepEqual([tab.scrollTop, away.scrollTop, late.scrollTop], [0, 0, 0]);
    for (const each of [tab, away]) { each.rendered = true; each.raw = 0; }  // shown again, each starts at its top (WebKit)
    late.scrollHeight = 2000;                   // the rows arrived
    w.shown();
    assert.deepEqual([tab.scrollTop, away.scrollTop, late.scrollTop], [300, 300, 300]);
});

test('an engine that keeps the place is left alone; one that drops it is asked once, at the first scroll', () => {
    const keeps = repaintWorld({ engine: false }), drops = repaintWorld();
    assert.deepEqual([keeps.asked(), drops.asked()], [0, 0]);
    const boxes = [keeps, drops].map((w) => {
        const box = w.element(w.element(w.content, { container: true }));
        w.scrolled(box, 480); w.scrolled(box, 500);
        return box;
    });
    assert.deepEqual([keeps.asked(), drops.asked()], [1, 1]);
    assert.deepEqual([keeps.watched(), drops.watched()], [false, true]);
    assert.deepEqual(boxes.map((box) => Object.prototype.hasOwnProperty.call(box, 'scrollTop')), [false, true]);
    assert.deepEqual([keeps.intoViewIsPlatforms(), drops.intoViewIsPlatforms()], [true, false]);
});
