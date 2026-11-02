import assert from 'node:assert/strict';
import test from 'node:test';

import { createSharedReadTracker, noteReadAhead, onReadAheadChange, readAheadFor, unreadCount } from '../modules/read_ahead.js';

test('a room counts its revisions past the cursor less those read in full elsewhere', () => {
    assert.equal(unreadCount(3, 0, []), 3);
    assert.equal(unreadCount(3, 0, [[3, 3]]), 2);              // m3 read in Main: p1 and p2 stay unread
    assert.equal(unreadCount(3, 2, [[1, 3]]), 0);              // spans below the cursor change nothing
    assert.equal(unreadCount(10, 4, [[2, 6], [9, 12]]), 2);    // clipped to (seen, visible]: 5-6 and 9-10
    assert.equal(unreadCount(5, 7, [[6, 6]]), 0);
    assert.equal(unreadCount('3', null, undefined), 3);
});

test('read-ahead follows the host answer and repaints its listeners only on change', () => {
    let repaints = 0;
    const dispose = onReadAheadChange(() => { repaints += 1; });
    noteReadAhead({ read_ahead_revisions: { 1: [[4, 4]], 5551: [[3, 3]] } });
    noteReadAhead({ read_ahead_revisions: { 1: [[4, 4]], 5551: [[3, 3]] } });
    noteReadAhead({ main_seen_revision: 2 });                  // an answer without the record keeps it
    assert.deepEqual(readAheadFor(5551), [[3, 3]]);
    assert.deepEqual(readAheadFor(1), [[4, 4]]);
    assert.deepEqual(readAheadFor(77), []);
    assert.equal(repaints, 1);
    dispose();
    noteReadAhead({ read_ahead_revisions: {} });
    assert.equal(repaints, 1);
});

// Computed overflow of the fake content parts below (the browser's getComputedStyle in the module).
globalThis.getComputedStyle = (part) => ({ overflowX: part.overflowX || 'visible', overflowY: part.overflowY || 'visible' });

// A feed of bubbles with a scrollable band: each bubble's box moves with `scroll`. `parts` are the
// elements inside its `.message` content (a clipping code block is {overflowX: 'auto', scrollWidth > clientWidth}),
// each drawn `at` px below its bubble's top, `clientHeight` tall and `clientWidth` wide, scrolled by `scrollLeft`.
function feedOf(nodes, band = { top: 100, bottom: 500 }) {
    const state = { scroll: 0 };
    const elements = nodes.map(({ address, revisions, top, height, folded = false, parts = [], hiddenPart = false }) => {
        const content = { overflowX: 'visible', overflowY: 'visible', scrollWidth: 600, clientWidth: 600,
            scrollHeight: height, clientHeight: height,
            querySelector: (selector) => (hiddenPart && selector.includes('[hidden]') ? {} : null),
            querySelectorAll: () => parts };
        const node = {
            dataset: { rowAddress: address, roomRevisions: JSON.stringify(revisions) },
            height, top, width: 600,
            querySelector: (selector) => (selector === ':scope > .message' ? content : null),
            closest: (selector) => (folded && selector.includes('.is-folded') ? node : null),
            getBoundingClientRect: () => ({ top: node.top - state.scroll, bottom: node.top - state.scroll + node.height,
                height: node.height, width: node.width, left: 0, right: node.width }),
        };
        for (const part of parts) {
            Object.assign(part, { scrollLeft: part.scrollLeft || 0, scrollTop: part.scrollTop || 0, clientLeft: 0, clientTop: 0,
                contains: (other) => other === part });
            part.getBoundingClientRect = () => {
                const partTop = node.top - state.scroll + (part.at || 0);
                return { top: partTop, bottom: partTop + part.clientHeight, left: 0, right: part.clientWidth,
                    height: part.clientHeight, width: part.clientWidth };
            };
        }
        return node;
    });
    const feed = { querySelectorAll: () => elements, getBoundingClientRect: () => ({ left: 0, right: 600 }) };
    return { state, band: () => band, feed, elements };
}

test('a long message is read by parts while scrolling; seeing only its bottom never counts', async () => {
    const posted = [];
    const view = feedOf([{ address: 'row:1@a#long', revisions: { 1: 7, 5551: 3 }, top: 200, height: 1000 }]);
    const tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true,
        post: async (address) => { posted.push(address); return true; } });
    view.state.scroll = 750;  // its bottom 350px (650-1000 of the message) is in the band
    tracker.sample();
    assert.deepEqual(posted, []);
    view.state.scroll = 0; tracker.sample();      // its top 300px: 300-650 never shown yet
    assert.deepEqual(posted, []);
    view.state.scroll = 350; tracker.sample();    // the middle: every part has now been shown
    assert.deepEqual(posted, ['row:1@a#long']);
    tracker.sample();
    assert.deepEqual(posted, ['row:1@a#long'], 'one record per message');
});

test('a resize starts the reading again; folded, hidden and own-room-only rows are never recorded', () => {
    const posted = [];
    const view = feedOf([
        { address: 'row:1@b#resized', revisions: { 1: 8, 5551: 4 }, top: 150, height: 300 },
        { address: 'row:1@c#folded', revisions: { 1: 9, 5551: 5 }, top: 150, height: 200, folded: true },
        { address: 'row:5551@d#own', revisions: { 5551: 6 }, top: 150, height: 100 },
    ]);
    const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true,
        post: async (address) => { posted.push(address); return true; } });
    view.state.scroll = 0;
    view.elements[0].height = 600;                 // grew: 150..750, band 100..500 shows its top part only
    tracker.sample();
    view.elements[0].height = 300;                 // a new size: the earlier part does not carry over
    view.state.scroll = 300;                       // 150..450 -> box -150..150: only its lower half
    tracker.sample();
    assert.deepEqual(posted, []);
    view.state.scroll = 0; tracker.sample();       // whole at one size
    assert.deepEqual(posted, ['row:1@b#resized']);
    const hidden = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => false,
        post: async (address) => { posted.push(address); return true; } });
    hidden.sample();
    assert.deepEqual(posted, ['row:1@b#resized']);
});

test('a failed record is offered again only after the message is shown whole again', async () => {
    let ok = false;
    const posted = [];
    const view = feedOf([{ address: 'row:1@e#retry', revisions: { 1: 10, 5551: 7 }, top: 150, height: 100 }]);
    const tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true,
        post: async (address) => { posted.push(address); return ok; } });
    tracker.sample();
    await new Promise((resolve) => setTimeout(resolve, 0));
    ok = true;
    tracker.sample();
    assert.deepEqual(posted, ['row:1@e#retry', 'row:1@e#retry']);
});

test('words a part clips are not shown by its box: a clipped code block seen at one place, or a hidden part, records nothing', () => {
    const posted = [];
    const wide = { overflowX: 'auto', overflowY: 'visible', scrollWidth: 1400, clientWidth: 560, scrollHeight: 40, clientHeight: 40 };
    const fits = { overflowX: 'auto', overflowY: 'visible', scrollWidth: 560, clientWidth: 560, scrollHeight: 40, clientHeight: 40 };
    const view = feedOf([
        { address: 'row:1@f#clipped', revisions: { 1: 11, 5551: 8 }, top: 150, height: 120, parts: [wide] },
        { address: 'row:1@g#hidden', revisions: { 1: 12, 5551: 9 }, top: 280, height: 60, hiddenPart: true },
        { address: 'row:1@h#fits', revisions: { 1: 13, 5551: 10 }, top: 350, height: 100, parts: [fits] },
    ]);
    const tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true,
        post: async (address) => { posted.push(address); return true; } });
    tracker.sample();
    tracker.sample();
    assert.deepEqual(posted, ['row:1@h#fits']);  // whole boxes all three; only the one hiding nothing is read
});

const recorder = () => {
    const posted = [];
    return { posted, post: async (address) => { posted.push(address); return true; } };
};
const wideBlock = (extra = {}) => ({ overflowX: 'auto', overflowY: 'visible', scrollWidth: 1400, clientWidth: 560,
    scrollHeight: 200, clientHeight: 200, at: 20, ...extra });

test('a wide block is read once every part of it was in view: by parts while it scrolls sideways', () => {
    const { posted, post } = recorder();
    const block = wideBlock();
    const view = feedOf([{ address: 'row:1@w#wide', revisions: { 1: 21, 5551: 11 }, top: 150, height: 260, parts: [block] }]);
    const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
    tracker.sample();                              // the bubble whole, the block's first 560 of 1400 columns
    assert.deepEqual(posted, []);
    block.scrollLeft = 500; tracker.sample();       // 500-1060
    assert.deepEqual(posted, []);
    block.scrollLeft = 840; tracker.sample();       // 840-1400: every column of every row has been in view
    assert.deepEqual(posted, ['row:1@w#wide']);
});

test('sideways scrolling never shows the rows of a block that lie outside the read band', () => {
    const { posted, post } = recorder();
    const block = wideBlock();
    // The bubble 150..410 and the block 170..370 in the feed; the band ends at 300: rows 130-200 of the block are below it.
    const view = feedOf([{ address: 'row:1@x#half', revisions: { 1: 22, 5551: 12 }, top: 150, height: 260, parts: [block] }],
        { top: 100, bottom: 300 });
    const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
    for (const left of [0, 500, 840]) { block.scrollLeft = left; tracker.sample(); }
    view.state.scroll = 120; block.scrollLeft = 840; tracker.sample();   // the bubble's bottom rows, at the block's right end
    assert.deepEqual(posted, [], 'the lower rows were in view only at their right end');
    for (const left of [500, 0]) { block.scrollLeft = left; tracker.sample(); }
    assert.deepEqual(posted, ['row:1@x#half']);
});

test('rows an overlay lies over are not read: a toast over the last lines, a menu; a modal layer or dialog hides the room', () => {
    const { posted, post } = recorder();
    const view = feedOf([{ address: 'row:1@t#toasted', revisions: { 1: 23, 5551: 13 }, top: 150, height: 300 }]);
    const overlays = { '.toast': [], '[role="menu"]': [], '.ui-hint': [], '[aria-modal="true"]': [] };
    const drawnBox = (box) => ({ getClientRects: () => [box], getBoundingClientRect: () => box });
    view.feed.ownerDocument = { querySelectorAll: (selector) => selector.split(', ').flatMap((one) => overlays[one] || []) };
    let inert = false;
    view.feed.closest = (selector) => (inert && selector === '[inert]' ? {} : null);
    const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
    overlays['.toast'] = [drawnBox({ top: 400, bottom: 460, left: 20, right: 580 })];  // over the bubble's last 50px
    tracker.sample();
    assert.deepEqual(posted, []);
    overlays['.toast'] = [drawnBox({ top: 400, bottom: 460, left: 620, right: 900 })];  // beside the feed: hides nothing
    overlays['[role="menu"]'] = [drawnBox({ top: 380, bottom: 420, left: 0, right: 200 })];
    tracker.sample();
    assert.deepEqual(posted, [], 'the menu still covers rows 230-270');
    overlays['[role="menu"]'] = [];
    inert = true; tracker.sample();                 // a modal layer over the room (layers.js makes it inert)
    inert = false;
    overlays['[aria-modal="true"]'] = [drawnBox({ top: 0, bottom: 900, left: 0, right: 900 })];
    tracker.sample();
    assert.deepEqual(posted, []);
    overlays['[aria-modal="true"]'] = [];
    tracker.sample();                               // gone: the next reading moment shows the rows it covered
    assert.deepEqual(posted, ['row:1@t#toasted']);
});

test('a part that scrolls inside the feed is a reading moment; dispose ends it', () => {
    const { posted, post } = recorder();
    const block = wideBlock({ scrollWidth: 700 });
    const view = feedOf([{ address: 'row:1@s#scrolled', revisions: { 1: 24, 5551: 14 }, top: 150, height: 260, parts: [block] }]);
    const listeners = [];
    view.feed.addEventListener = (type, listener, options) => listeners.push({ type, listener, options });
    view.feed.removeEventListener = (type, listener) => {
        const at = listeners.findIndex((item) => item.type === type && item.listener === listener);
        if (at >= 0) listeners.splice(at, 1);
    };
    const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
    assert.equal(listeners.length, 1);
    assert.equal(listeners[0].type, 'scroll');
    assert.equal(listeners[0].options.capture, true, 'a scroll does not bubble: the feed captures its parts\' scrolls');
    tracker.sample();
    block.scrollLeft = 140;
    listeners[0].listener({ target: view.feed });   // the feed's own scroll is the receipt's moment, not this one
    assert.deepEqual(posted, []);
    listeners[0].listener({ target: block });
    assert.deepEqual(posted, ['row:1@s#scrolled']);
    tracker.dispose();
    assert.equal(listeners.length, 0);
});

test('an older preferences answer arriving last never takes back a message read in full', () => {
    const dispose = onReadAheadChange(() => {});
    noteReadAhead({ read_ahead_revisions: { 1: [[30, 30]], 7001: [[5, 5]] } });   // the newer answer (m3 read)
    noteReadAhead({ read_ahead_revisions: { 1: [[30, 30]] } });                  // an older state GET, answered late
    assert.deepEqual(readAheadFor(7001), [[5, 5]]);
    assert.equal(unreadCount(5, 2, readAheadFor(7001)), 2, 'p1, p2 stay unread; m3 stays read');
    noteReadAhead({ read_ahead_revisions: { 7001: [[6, 6]] } });
    assert.deepEqual(readAheadFor(7001), [[5, 6]]);
    dispose();
});

test('a room under a modal layer reads nothing, and its end is a reading moment without a scroll', async () => {
    const { posted, post } = recorder();
    const view = feedOf([{ address: 'row:1@m#modal', revisions: { 1: 25, 5551: 15 }, top: 150, height: 200 }]);
    const room = { inert: true, parentNode: {} };
    view.feed.closest = (selector) => (selector === '[inert]' && room.inert ? room : null);
    const observers = [];
    globalThis.MutationObserver = class {
        constructor(callback) { this.callback = callback; this.targets = []; observers.push(this); }
        observe(target, options) { this.targets.push({ target, options }); }
        disconnect() { this.targets = []; this.disconnected = true; }
    };
    try {
        const tracker = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
        tracker.sample();
        assert.deepEqual(posted, [], 'the whole message lies under the layer');
        assert.equal(observers.length, 1);
        assert.ok(observers[0].targets.some(({ target, options }) => target === room && options.attributeFilter.includes('inert')));
        tracker.sample();
        assert.equal(observers.length, 1, 'one watch at a time');
        room.inert = false;                      // the layer closed: the room is the reader's again
        observers[0].callback([]);
        assert.equal(observers[0].disconnected, true);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(posted, ['row:1@m#modal']);
        tracker.sample();
        assert.equal(observers.length, 1, 'nothing left unread: no watch');
        const again = createSharedReadTracker({ chatId: 5551, feed: view.feed, band: view.band, shown: () => true, post });
        room.inert = true;
        view.elements[0].dataset.rowAddress = 'row:1@n#other';
        again.sample();
        assert.equal(observers.length, 2);
        again.dispose();
        assert.equal(observers[1].disconnected, true, 'dispose ends the watch');
    } finally {
        delete globalThis.MutationObserver;
    }
});

test('the Project sheet over Main hides Main; the room inside the sheet still reads', () => {
    const { posted, post } = recorder();
    const main = feedOf([{ address: 'row:1@p#under-sheet', revisions: { 1: 26, 5551: 16 }, top: 150, height: 200 }]);
    const room = feedOf([{ address: 'row:1@q#in-sheet', revisions: { 1: 27, 5551: 17 }, top: 150, height: 200 }]);
    const open = { value: true };
    const sheet = { getClientRects: () => (open.value ? [{}] : []),
        getBoundingClientRect: () => ({ top: 0, bottom: 900, left: 420, right: 600 }),
        contains: (node) => node === room.feed };
    const doc = { querySelectorAll: (selector) => (selector.includes('.project-panel.open') ? [sheet] : []) };
    main.feed.ownerDocument = doc;
    room.feed.ownerDocument = doc;
    const underneath = createSharedReadTracker({ chatId: 1, feed: main.feed, band: main.band, shown: () => true, post });
    const inside = createSharedReadTracker({ chatId: 5551, feed: room.feed, band: room.band, shown: () => true, post });
    underneath.sample();
    inside.sample();
    assert.deepEqual(posted, ['row:1@q#in-sheet'], 'Main lies under the sheet; the sheet holds its own room');
    open.value = false;                              // closed: Main is the reader's again
    underneath.sample();
    assert.deepEqual(posted, ['row:1@q#in-sheet', 'row:1@p#under-sheet']);
});


test('width-only reflow resets partial reading even when the message height does not change', () => {
    const { posted, post } = recorder();
    const view = feedOf([{ address: 'row:1@r#reflow', revisions: { 1: 28, 5551: 18 }, top: 150, height: 600 }]);
    const tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true, post });
    tracker.sample();                               // first 350px, in the original layout
    assert.deepEqual(posted, []);
    view.elements[0].width = 640;
    view.state.scroll = 250;
    tracker.sample();                               // lower 400px of a different word layout
    assert.deepEqual(posted, [], 'partial coverage of the narrower layout cannot fill the unseen top');
    view.state.scroll = 0;
    tracker.sample();
    assert.deepEqual(posted, ['row:1@r#reflow']);
    tracker.dispose();
});

test('a non-modal room work list conceals a shared message until its dismissal triggers sampling', async () => {
    const { posted, post } = recorder();
    const view = feedOf([{ address: 'row:1@r#work-list', revisions: { 1: 29, 5551: 19 }, top: 150, height: 100 }]);
    let open = true, observer;
    const sheet = { parentNode: {}, getClientRects: () => open ? [{}] : [],
        getBoundingClientRect: () => ({ top: 100, bottom: 300, left: 0, right: 600 }) };
    view.feed.ownerDocument = { querySelectorAll: (selector) => selector.split(', ').includes('.rw-shade') ? [sheet] : [] };
    globalThis.MutationObserver = class {
        constructor(callback) { this.callback = callback; observer = this; }
        observe() {}
        disconnect() {}
    };
    const tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true, post });
    try {
        tracker.sample();
        assert.deepEqual(posted, []);
        assert.ok(observer, 'the existing overlay watcher must watch the work list too');
        open = false;
        observer.callback([]);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(posted, ['row:1@r#work-list']);
    } finally {
        tracker.dispose();
        delete globalThis.MutationObserver;
    }
});


test('completed and released partial messages retain no DOM subtrees', () => {
    const { post } = recorder();
    const view = feedOf([
        { address: 'row:1@retention#whole', revisions: { 1: 40, 5551: 30 }, top: 150, height: 100 },
        { address: 'row:1@retention#partial', revisions: { 1: 41, 5551: 31 }, top: 280, height: 1000 },
    ]);
    const NativeMap = globalThis.Map, stores = [];
    let tracker;
    globalThis.Map = class extends NativeMap {
        constructor(...args) { super(...args); stores.push(this); }
    };
    try {
        tracker = createSharedReadTracker({ chatId: 1, feed: view.feed, band: view.band, shown: () => true, post });
    } finally {
        globalThis.Map = NativeMap;
    }
    try {
        tracker.sample();
        assert.equal(stores.length, 1);
        assert.deepEqual([...stores[0].keys()], ['row:1@retention#partial'], 'a receipt retains identity, not the completed node');
        tracker.release(view.elements[1]);
        assert.equal(stores[0].size, 0, 'history page retirement drops its partial reading state');
        tracker.release(view.elements[0]);
        tracker.release({ dataset: {} });
        assert.equal(stores[0].size, 0);
    } finally {
        tracker.dispose();
    }
});
