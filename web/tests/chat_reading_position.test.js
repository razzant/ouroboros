import assert from 'node:assert/strict';
import test from 'node:test';
import { createReviewHydrator } from '../modules/review_presentation.js';
import { createChatReadingPosition } from '../modules/chat_reading_position.js';
import { createTimelineAnchors } from '../modules/chat_render_batch.js';

function fixture(t, exact = true, restoreTo = 80) {
    const previous = globalThis.requestAnimationFrame;
    const frames = [];
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    let ready = false, visible = true, restored = 0, changes = 0;
    // Dispatch topology: the feed hears only events targeted inside it; the document hears all.
    const target = () => { const listeners = new Map();
        return { listeners, addEventListener: (type, fn) => listeners.set(type, fn), removeEventListener: type => listeners.delete(type) }; };
    const ownerDocument = { ...target(), body: {}, documentElement: {} };
    const feed = { scrollTop: 0, scrollHeight: 300, clientHeight: 500, clientWidth: 400, ownerDocument, ...target(),
        contains: node => node === feed || Boolean(node?.inFeed), get offsetHeight() { return this.clientHeight; },
        getBoundingClientRect() { return { top: 0, height: this.clientHeight }; } };
    const dispatch = event => {
        if (feed.contains(event.target)) feed.listeners.get(event.type)?.(event);
        ownerDocument.listeners.get(event.type)?.(event);
    };
    const bookmark = { scrollTop: 2400, stick: false, historyAnchor: { historyId: 'chat:old', offset: 80 } };
    const reading = createChatReadingPosition({ feed, visible: () => visible,
        alive: () => true, ready: () => ready, anchors: { serialize: () => bookmark.historyAnchor, capture: () => null,
            restore: () => { restored++; if (restoreTo !== null) feed.scrollTop = restoreTo; return typeof exact === 'function' ? exact() : exact; } },
        changed() { changes++; }, afterWrite() {}, activity() {}, updateButton() {},
    });
    // A reader away from the bottom whose place awaits data (a window shown again).
    reading.top = bookmark.scrollTop; reading.stick = false; reading.request();
    return { reading, feed, frames, bookmark, gesture: event => dispatch({ target: feed, ...event }),
        ready: () => { ready = true; }, hide: () => { visible = false; },
        changed: () => changes, restored: () => restored, frame: () => { const work = frames.splice(0); work.forEach(callback => callback()); } };
}

test('explicit navigation clears approximate position and immediately republishes status', t => {
    const f = fixture(t, false);
    f.ready(); f.reading.position(); f.frame(); f.frame();
    assert.equal(f.reading.approximate, true);
    assert.equal(f.changed(), 1);
    f.reading.cancel();
    assert.equal(f.reading.approximate, false);
    assert.equal(f.changed(), 2, 'the persistent notice clears without waiting for another history read');
});

test('a jump keeps its own reading intent when a later visit moves the same room', t => {
    const f = fixture(t, false, null);
    f.feed.scrollHeight = 5000;
    const restore = f.reading.capture();
    f.reading.cancel();
    f.reading.top = 3500;
    f.reading.request();
    assert.equal(f.reading.target.scrollTop, 3500);
    restore();
    assert.equal(f.reading.target.scrollTop, 2400, 'the earlier jump keeps the earlier passage');
    f.ready(); f.reading.position(); f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, 2400);
    assert.equal(f.reading.pending, false);
});

test('an exact second positioning pass clears the earlier approximation', t => {
    let exact = false;
    const f = fixture(t, () => exact);
    f.ready(); f.reading.position(); f.frame();
    assert.equal(f.reading.approximate, true);
    exact = true;
    f.frame();
    assert.equal(f.reading.pending, false);
    assert.equal(f.reading.approximate, false);
});

test('a failed or delayed read schedules no layout polling and keeps the original target', t => {
    const f = fixture(t);
    f.reading.request();
    for (let i = 0; i < 100; i++) f.frame();
    assert.equal(f.frames.length, 0);
    assert.deepEqual(f.reading.target, f.bookmark);
    f.reading.mutate(() => { f.feed.scrollHeight = 400; });
    assert.equal(f.reading.stick, false, 'short/empty layout cannot opt the reader into follow');
    f.ready(); f.reading.position(); f.frame(); f.frame();
    assert.equal(f.reading.pending, false);
    assert.equal(f.reading.stick, false);
    assert.equal(f.restored(), 2);
});

test('wheel/latest/question cancellation wins even between positioning frames', t => {
    const f = fixture(t); f.ready(); f.reading.position(); f.frame();
    f.reading.cancel(); f.feed.scrollTop = 210;
    f.frame();
    assert.equal(f.feed.scrollTop, 210);
    assert.equal(f.reading.target, null);
    f.reading.followAfterLayout(); f.reading.cancel(); f.frame();
    assert.equal(f.feed.scrollTop, 210, 'a superseded latest layout cannot pull the reader');
});

test('hidden/disposed layout retains intent for a later data-aware show', t => {
    const f = fixture(t); f.ready(); f.reading.position(); f.hide(); f.frame();
    assert.deepEqual(f.reading.target, f.bookmark);
    assert.equal(f.restored(), 0);
});

test('only downward intent on a scrollable feed can resume follow', t => {
    const f = fixture(t), navigation = [];
    f.reading.bindGestures(direction => navigation.push(direction));
    f.gesture({ type: 'wheel', deltaY: 1 }); f.frame();
    assert.equal(f.reading.stick, false, 'empty/short geometry cannot opt into follow');
    f.feed.scrollHeight = 1000; f.feed.scrollTop = 500;
    f.gesture({ type: 'wheel', deltaY: -1 }); f.reading.scroll(); f.frame();
    assert.equal(f.reading.stick, false, 'upward intent at bottom stops follow');
    f.gesture({ type: 'wheel', deltaY: 1 }); f.frame();
    assert.equal(f.reading.stick, true);
    f.gesture({ type: 'keydown', key: ' ', shiftKey: true }); f.frame();
    assert.equal(f.reading.stick, false, 'Shift+Space is upward');
    f.gesture({ type: 'wheel', deltaY: 1 });
    f.gesture({ type: 'wheel', deltaY: -1 }); f.frame();
    assert.equal(f.reading.stick, false, 'latest gesture supersedes a queued downward frame');
    const generation = f.reading.generation;
    f.gesture({ type: 'keydown', key: ' ', target: { inFeed: true, closest: selector => selector.includes('button') ? {} : null } });
    f.gesture({ type: 'keydown', key: 'ArrowDown', defaultPrevented: true });
    f.frame();
    assert.equal(f.reading.generation, generation, 'control activation is not archive navigation');
    assert.deepEqual(navigation, [1, -1, 1, -1, -1]);
});

test('keys scroll the pressed feed or its focused control even though keydown never targets the feed', t => {
    const f = fixture(t), navigation = [];
    const control = { inFeed: true, closest: selector => selector.includes('button') ? {} : null };
    const body = f.feed.ownerDocument.body;
    f.reading.bindGestures(direction => navigation.push(direction));
    f.reading.request();
    f.gesture({ type: 'keydown', key: 'PageDown', target: body }); f.frame();
    assert.equal(f.reading.pending, true, 'keys after pressing elsewhere scroll another surface');
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    assert.equal(f.reading.pending, true, 'pressing the text being read is not navigation');
    f.gesture({ type: 'keydown', key: 'PageDown', target: body }); f.frame();
    assert.equal(f.reading.pending, false, 'nothing focused: the last pressed feed scrolls');
    f.reading.request();
    f.gesture({ type: 'keydown', key: ' ', target: control }); f.frame();
    assert.equal(f.reading.pending, true, 'Space activates a focused control');
    f.gesture({ type: 'keydown', key: 'ArrowUp', target: control }); f.frame();
    assert.equal(f.reading.pending, false, 'arrow keys on a focused control scroll its feed');
    f.reading.request();
    f.gesture({ type: 'keydown', key: 'ArrowUp', target: { inFeed: true, closest: selector => selector.includes('textarea') ? {} : null } });
    f.gesture({ type: 'pointerdown', target: {} });
    f.gesture({ type: 'keydown', key: 'PageUp', target: body }); f.frame();
    assert.equal(f.reading.pending, true, 'editable caret keys and presses outside the feed keep the bookmark');
    assert.deepEqual(navigation, [1, -1]);
});

test('a bounded box that can still move absorbs wheel, touch and keys: no follow or paging, and a saved place yields to it', t => {
    const f = fixture(t, false), navigation = [];
    f.feed.ownerDocument.defaultView = { getComputedStyle: node => ({ overflowY: node.overflowY || 'visible' }) };
    // An overflowing but unscrollable card, a bounded full body, the text read in it.
    const card = { inFeed: true, parentElement: f.feed, scrollTop: 0, scrollHeight: 900, clientHeight: 600 };
    const body = { inFeed: true, parentElement: card, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400,
        getBoundingClientRect: () => ({ top: 20 }) };
    const text = { inFeed: true, parentElement: body };
    f.reading.bindGestures(direction => navigation.push(direction));
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 560, clientHeight: 400 }); // 40px above the end
    f.ready(); f.reading.position(); f.frame(); f.frame();
    assert.equal(f.reading.approximate, true);
    const changes = f.changed();
    f.feed.scrollTop = 560;
    f.gesture({ type: 'wheel', deltaY: 1, target: text }); f.frame();
    assert.equal(f.reading.approximate, false, 'the reader moved on: the approximation notice clears');
    assert.equal(f.changed(), changes + 1);
    assert.equal(f.reading.stick, false, 'scrolling the body down 40px above the end cannot turn on following');
    // A saved place still positioning: the reader's own box scrolling supersedes it.
    f.reading.request(); f.feed.scrollTop = 560;
    f.gesture({ type: 'touchstart', touches: [{ clientY: 100 }], target: text });
    f.gesture({ type: 'pointerdown', target: text });
    assert.equal(f.reading.pending, true, 'touching or pressing the box moves nothing');
    f.gesture({ type: 'touchmove', touches: [{ clientY: 60 }], target: text });
    assert.equal(f.reading.pending, false);
    f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, 560, 'a superseded restore can no longer move the reader');
    f.reading.stick = true; f.feed.scrollTop = 600;
    f.gesture({ type: 'wheel', deltaY: 1, target: text }); f.frame();
    assert.equal(f.reading.stick, true, 'reading on down a box at the live edge keeps following');
    f.gesture({ type: 'wheel', deltaY: -1, target: text }); f.frame();
    assert.equal(f.reading.stick, false, 'turning back inside the box ends following: new replies cannot move it');
    f.reading.stick = true; f.feed.scrollTop = 20; // e.g. a programmatic reveal left the flag behind
    f.gesture({ type: 'wheel', deltaY: 1, target: text }); f.frame();
    assert.equal(f.reading.stick, false, 'away from the live edge a box gesture ends a stale follow');
    f.feed.scrollTop = 20;
    f.gesture({ type: 'wheel', deltaY: -1, target: text });
    f.gesture({ type: 'touchstart', touches: [{ clientY: 100 }], target: text });
    f.gesture({ type: 'touchmove', touches: [{ clientY: 140 }], target: text }); f.frame();
    f.gesture({ type: 'pointerdown', target: text });
    f.gesture({ type: 'keydown', key: 'PageUp', target: f.feed.ownerDocument.body });
    f.gesture({ type: 'keydown', key: 'ArrowUp', target: { inFeed: true, parentElement: body, closest: () => null } }); f.frame();
    assert.deepEqual(navigation, [], 'a body that can move up never pages the archive at the feed top');
    body.scrollTop = 0;
    f.gesture({ type: 'wheel', deltaY: -1, target: text }); f.frame();
    assert.deepEqual(navigation, [-1], 'at the body edge the gesture reaches the feed');
    body.scrollTop = 600; f.feed.scrollTop = 560;
    f.gesture({ type: 'keydown', key: 'PageDown', target: f.feed.ownerDocument.body }); f.frame();
    assert.equal(f.reading.stick, true, 'the last pressed body is at its end: keys scroll the feed');
    assert.deepEqual(navigation, [-1, 1]);
});

test('a wheel burst or touch stays with the box it began on, even at that box edge', t => {
    const f = fixture(t), navigation = [];
    f.feed.ownerDocument.defaultView = { getComputedStyle: node => ({ overflowY: node.overflowY || 'visible' }) };
    const body = { inFeed: true, parentElement: f.feed, overflowY: 'auto', scrollTop: 30, scrollHeight: 1000, clientHeight: 400 };
    const text = { inFeed: true, parentElement: body };
    f.reading.bindGestures(direction => navigation.push(direction));
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 0, clientHeight: 400 }); // the feed is at its archive edge
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 1000 }); f.frame();
    body.scrollTop = 0; // the browser moved the box to its edge
    for (const at of [1016, 1032, 1150]) { f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: at }); f.frame(); }
    assert.deepEqual(navigation, [], 'trailing events of one burst stay with the box: no archive page');
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 1400 }); f.frame();
    assert.deepEqual(navigation, [-1], 'a new burst over the box at its edge reaches the feed');
    body.scrollTop = 30;
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 1450 }); f.frame();
    assert.deepEqual(navigation, [-1, -1], 'a burst that began on the feed stays with the feed');
    f.gesture({ type: 'touchstart', touches: [{ clientY: 100 }], target: text });
    f.gesture({ type: 'touchmove', touches: [{ clientY: 120 }], target: text }); f.frame();
    body.scrollTop = 0;
    f.gesture({ type: 'touchmove', touches: [{ clientY: 160 }], target: text }); f.frame();
    assert.deepEqual(navigation, [-1, -1], 'one touch that began in the box stays with it');
    f.gesture({ type: 'touchstart', touches: [{ clientY: 100 }], target: text });
    f.gesture({ type: 'touchmove', touches: [{ clientY: 140 }], target: text }); f.frame();
    assert.deepEqual(navigation, [-1, -1, -1], 'a new touch at the box edge reaches the feed');
    // Browsers may scroll the box before reporting the wheel: the event sees the box
    // already at its edge, then the box's own scroll event tells whose gesture it was.
    body.scrollTop = 0;
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 3000 });
    f.gesture({ type: 'scroll', target: body, timeStamp: 3004 }); f.frame();
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 3020 }); f.frame();
    assert.deepEqual(navigation, [-1, -1, -1], 'the box scrolled for it: no archive page, and its burst stays latched');
    assert.equal(f.reading.stick, false);
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: 3500 }); f.frame();
    assert.deepEqual(navigation, [-1, -1, -1, -1], 'an older box scroll proves nothing about a new burst');
});

for (const input of ['wheel', 'touch']) {
    test(`successive ${input} gestures inside a box keep their own movement through resizes`, t => {
        const previous = globalThis.requestAnimationFrame, frames = [];
        globalThis.requestAnimationFrame = callback => frames.push(callback);
        t.after(() => { globalThis.requestAnimationFrame = previous; });
        const listeners = new Map(), navigation = [];
        const doc = { addEventListener() {}, defaultView: { getComputedStyle: () => ({ overflowY: 'auto' }) } };
        const feed = { ownerDocument: doc, clientWidth: 390, offsetWidth: 390, clientHeight: 400, scrollHeight: 1000,
            scrollTop: 600, get offsetHeight() { return this.clientHeight; },
            getBoundingClientRect() { return { top: 0, height: this.clientHeight }; },
            contains: node => node === box && box.isConnected,
            addEventListener: (type, fn) => listeners.set(type, fn) };
        let boxY = 680, lineY = 140, captures = 0, boxReads = 0;
        const restored = [];
        const box = { parentElement: feed, isConnected: true, clientHeight: 200, scrollHeight: 500, scrollTop: 100,
            getBoundingClientRect() { boxReads++; return { top: boxY - feed.scrollTop }; } };
        const line = { parentElement: box, getBoundingClientRect: () => ({ top: boxY - feed.scrollTop + lineY - box.scrollTop }) };
        const anchors = {
            capture() { captures++; return { node: line, offset: line.getBoundingClientRect().top }; },
            restore(anchor) {
                if (!anchor) return false;
                restored.push(anchor.node);
                feed.scrollTop += anchor.node.getBoundingClientRect().top - anchor.offset;
                return true;
            },
        };
        const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
            anchors, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
        reading.bindGestures(direction => navigation.push(direction));
        reading.reflow(() => {}); // pinned height-only fast path, no bookmark or transcript scan
        assert.equal(captures, 0);
        const generation = reading.generation;
        const emit = (type, extra) => listeners.get(type)({ type, target: line, ...extra });
        const readBox = (start, steps) => {
            if (input === 'touch') emit('touchstart', { touches: [{ clientX: 150, clientY: 100 }] });
            for (let i = 0; i < steps; i++) {
                const at = start + i * 16;
                if (input === 'touch') emit('touchmove', { timeStamp: at, touches: [{ clientX: 150, clientY: 124 + i * 24 }] });
                else emit('wheel', { timeStamp: at, deltaY: -24 });
                box.scrollTop -= 24; // browser default runs AFTER the event reaches the reading owner
                listeners.get('scroll')({ target: box, timeStamp: at + 1 });
            }
            frames.splice(0).forEach(callback => callback());
        };
        readBox(1000, 2);
        assert.deepEqual([reading.stick, reading.generation, feed.scrollTop, navigation], [false, generation, 600, []]);
        const at = line.getBoundingClientRect().top;
        reading.reflow(() => { feed.clientHeight -= 75; boxY -= 24; });
        assert.equal(line.getBoundingClientRect().top, at, 'the same line stays put after the box moves');
        assert.equal(box.scrollTop, 52, 'the gesture still moved the inner box; the outer feed did not undo it');
        reading.scroll();    // the app's first placement arrives as a feed scroll
        reading.reflow();    // a follow-up measurement must not select an inner line
        readBox(1400, 1);
        assert.deepEqual([feed.scrollTop, box.scrollTop, line.getBoundingClientRect().top], [576, 28, at + 24]);
        if (input === 'touch') {
            // A room grows: its clamp may notify before the ResizeObserver's placement.
            Object.assign(feed, { clientHeight: 435, scrollTop: 565 });
            reading.scroll();
        }
        reading.reflow(() => { if (input === 'wheel') feed.clientHeight -= 23; boxY -= 30; });
        assert.deepEqual([feed.scrollTop, box.scrollTop, line.getBoundingClientRect().top], [546, 28, at + 24],
            'the next resize compensates only the box geometry, not the second inner move');
        assert.deepEqual([boxReads, captures, reading.generation, navigation], [4, 0, generation, []],
            'geometry needs one box read per restoration, neither a viewport scan nor a capture per gesture event');

        reading.mutate(() => { lineY += 18; return true; });
        assert.equal(restored.at(-1), line, 'a true content write still captures and restores the exact visible line');
        assert.deepEqual([feed.scrollTop, box.scrollTop, line.getBoundingClientRect().top], [564, 28, at + 24]);
    });
}

// A card timeline the reader scrolled inside, in a feed the keyboard resizes (style.css: its max-height follows the viewport).
// The box sits 680 px down the feed; its passage line 440 px down its own content; anchors capture the passing line.
function nestedRoom(t) {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const target = () => { const heard = new Map();
        return { heard, addEventListener: (type, fn) => heard.set(type, fn), removeEventListener: type => heard.delete(type) }; };
    const doc = { ...target(), body: {}, documentElement: {}, defaultView: { getComputedStyle: () => ({ overflowY: 'auto' }) } };
    const clamped = { get scrollTop() { return this.top; },
        set scrollTop(value) { this.top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); } };
    const feed = Object.defineProperties({ ...target(), ownerDocument: doc, clientWidth: 390, offsetWidth: 390, clientHeight: 400,
        scrollHeight: 3000, top: 600, get offsetHeight() { return this.clientHeight; },
        getBoundingClientRect() { return { top: 0, height: this.clientHeight }; },
        contains: node => node === feed || node === box || node === line }, Object.getOwnPropertyDescriptors(clamped));
    const place = { boxY: 680, passing: true }; // `passing: false`: the write finds no line to hold (its row is being replaced)
    const box = Object.defineProperties({ parentElement: feed, isConnected: true, clientHeight: 65, scrollHeight: 500, top: 435,
        getBoundingClientRect: () => ({ top: place.boxY - feed.scrollTop }) }, Object.getOwnPropertyDescriptors(clamped));
    const line = { parentElement: box, getBoundingClientRect: () => ({ top: place.boxY - feed.scrollTop + 440 - box.scrollTop }) };
    const anchors = { capture: () => (place.passing ? { node: line, offset: line.getBoundingClientRect().top } : null),
        restore(anchor) { if (!anchor) return false; feed.scrollTop += anchor.node.getBoundingClientRect().top - anchor.offset; return true; } };
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.bindGestures(() => {});
    reading.stick = false; reading.remember(true);
    let at = 1000;
    const emit = (node, type, extra) => {
        const event = { type, target: node, timeStamp: at += 16, ...extra };
        if (node === feed || feed.contains(node)) feed.heard.get(type)?.(event);
        doc.heard.get(type)?.(event);
    };
    // The browser moves the box after the event reached the reading owner, then reports the box's scroll.
    const moveBox = by => { box.scrollTop += by; emit(box, 'scroll'); };
    const swipeBox = by => {
        emit(line, 'touchstart', { touches: [{ clientX: 150, clientY: 100 }] });
        emit(line, 'touchmove', { touches: [{ clientX: 150, clientY: 100 - by }] });
        moveBox(by);
        frames.splice(0).forEach(callback => callback());
    };
    return { reading, feed, box, line, place, emit, moveBox, swipeBox, passage: () => line.getBoundingClientRect().top };
}

test('a box grown with the viewport keeps the passage its clamped scroll would drop, and gets its scroll back', t => {
    const { reading, feed, box, moveBox, swipeBox, passage } = nestedRoom(t);
    swipeBox(-24); // the reader reads up inside the timeline
    const read = passage();
    assert.deepEqual([box.scrollTop, feed.scrollTop, read], [411, 600, 109]);
    // The keyboard goes away: the feed and the timeline grow 65 -> 132 px; the timeline's own scroll clamps 411 -> 368.
    reading.reflow(() => { feed.clientHeight += 67; box.clientHeight = 132; box.scrollTop = box.scrollTop; });
    assert.equal(box.scrollTop, 368, 'the clamp is the box\'s own geometry');
    assert.equal(passage(), read, 'the outer feed lifts the box by the clamp: the passage stays');
    assert.equal(feed.scrollTop, 643);
    moveBox(0); // the clamp's own scroll arrives after the reflow
    reading.scroll();
    reading.reflow();
    assert.deepEqual([passage(), feed.scrollTop], [read, 643], 'a follow-up measurement keeps it');
    // The keyboard rises again: the box can reach its scroll again, and is unmoved since.
    reading.reflow(() => { feed.clientHeight -= 67; box.clientHeight = 65; });
    assert.deepEqual([box.scrollTop, feed.scrollTop, passage()], [411, 600, read], 'the box and its scroll are back');
    // A reader's move after a lift starts from where the box is.
    reading.reflow(() => { feed.clientHeight += 67; box.clientHeight = 132; box.scrollTop = box.scrollTop; });
    moveBox(0);
    swipeBox(-30);
    const moved = passage();
    assert.equal(moved, read + 30);
    reading.reflow(() => { feed.clientHeight -= 30; });
    assert.deepEqual([passage(), box.scrollTop], [moved, 338], 'the inner move is the reader\'s, the outer lift stays');
});

test('a state poll while the keyboard is away keeps the box\'s own scroll and its lift for the keyboard\'s return', t => {
    const { reading, feed, box, moveBox, swipeBox, passage } = nestedRoom(t);
    swipeBox(-24);
    const read = passage();
    reading.reflow(() => { feed.clientHeight += 67; box.clientHeight = 132; box.scrollTop = box.scrollTop; });
    moveBox(0);
    reading.scroll();
    reading.mutate(() => true); // /api/state polling while the keyboard is away
    assert.deepEqual([box.scrollTop, feed.scrollTop, passage()], [368, 643, read], 'the write moves nothing');
    reading.reflow(() => { feed.clientHeight -= 67; box.clientHeight = 65; });
    assert.deepEqual([box.scrollTop, feed.scrollTop, passage()], [411, 600, read], 'the box and its scroll are back');
    assert.ok(passage() < 680 - feed.scrollTop + box.clientHeight, 'the passage is inside the box, not under its edge');
});

for (const via of ['swipe', 'scrollbar']) {
    test(`a state poll during ${via === 'swipe' ? 'a lifted swipe\'s momentum' : 'a box scrollbar drag'} keeps the box, not the passing line`, t => {
        const { reading, feed, box, emit, moveBox, swipeBox, passage } = nestedRoom(t);
        if (via === 'swipe') swipeBox(-24);
        else { emit(box, 'pointerdown'); moveBox(-24); }
        reading.mutate(() => true); // /api/state polling every few seconds, or a live frame
        moveBox(-20); // momentum, or the drag going on
        reading.mutate(() => true);
        moveBox(-20);
        if (via === 'scrollbar') emit(box, 'pointerup');
        const top = feed.scrollTop, read = passage();
        assert.equal(box.scrollTop, 371);
        reading.reflow(() => { feed.clientHeight -= 75; }); // the keyboard rises
        assert.deepEqual([feed.scrollTop, passage()], [top, read], 'the next reflow does not undo the inner movement with the feed');
        box.isConnected = false; // the card re-rendered: the ordinary capture takes over
        reading.mutate(() => true);
        reading.reflow(() => { feed.clientHeight += 75; });
        assert.equal(feed.scrollTop, top);
    });
}

// A phone room 150 px high whose reader is not following: rows 100 px apart, the last one (M) 99 px down the feed, the
// feed at its end. Anchors keep the first row in sight at its offset, as chat_render_batch.js does.
function shortRoom() {
    const feed = { clientWidth: 390, offsetWidth: 390, clientHeight: 150, scrollHeight: 3000, top: 2850,
        get scrollTop() { return this.top; },
        set scrollTop(value) { this.top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); },
        get offsetHeight() { return this.clientHeight; },
        getBoundingClientRect() { return { top: 0, bottom: this.clientHeight, height: this.clientHeight }; },
        contains: node => node === feed || rows.includes(node) };
    const rows = Array.from({ length: 30 }, (_, i) => ({ historyId: `chat:${i}`, parentElement: feed, isConnected: true, y: 49 + 100 * i,
        getBoundingClientRect() { const top = this.y - feed.scrollTop; return { top, bottom: top + 100, width: 390, height: 100 }; } }));
    const shown = row => { const r = row.getBoundingClientRect(); return r.bottom > 0 && r.top < feed.clientHeight; };
    const anchors = { serialize: () => null,
        capture() { const node = rows.find(shown); return node ? { node, historyId: node.historyId, offset: node.getBoundingClientRect().top } : null; },
        restore(anchor) {
            if (!anchor?.node?.isConnected) return false;
            feed.scrollTop += anchor.node.getBoundingClientRect().top - anchor.offset;
            return true;
        } };
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.stick = false; reading.remember(true);
    // The keyboard goes (+) or comes (-): the room's height moves and the browser clamps the feed to its new limit.
    const room = by => { feed.clientHeight += by; feed.scrollTop = feed.scrollTop; };
    return { reading, feed, rows, room, m: () => rows.at(-1).getBoundingClientRect().top };
}

for (const order of ['clamp-scroll-first', 'reflow-first']) for (const poll of [false, true]) {
    test(`a place the feed's scroll limit keeps it from is still the place when the room comes back (${order}${poll ? ', a state poll between' : ''})`, () => {
        const { reading, room, m } = shortRoom();
        assert.equal(m(), 99);
        // The keyboard goes: the room grows 420 px and the feed, at its end, cannot keep M where it was read.
        if (order === 'clamp-scroll-first') { room(420); reading.scroll(); reading.reflow(() => {}); }
        else { reading.reflow(() => room(420)); reading.scroll(); }
        assert.equal(m(), 519, 'the taller room shows the feed\'s end: nothing more can be done');
        if (poll) reading.mutate(() => true); // /api/state while the keyboard is away
        reading.reflow(() => room(-420)); // the keyboard comes back
        assert.equal(m(), 99, 'the message is back where it was read');
        reading.scroll();
        reading.reflow(() => room(-20)); // a second line in the composer
        assert.equal(m(), 99, 'and is an ordinary place again');
    });
}

test('a held place comes back as far as each smaller room allows, follows a reply in, and yields to the reader\'s own move', () => {
    let { reading, feed, room, m } = shortRoom();
    reading.reflow(() => room(420));
    reading.scroll();
    reading.reflow(() => room(-144)); // the keyboard half way up
    assert.equal(m(), 375, 'the feed gives back what the room took, px for px');
    reading.reflow(() => room(144));
    assert.equal(m(), 519);
    // A reply arrives below while the room is tall: the feed moves toward the place with it, as a follower's would.
    reading.mutate(() => { feed.scrollHeight += 200; return true; }, { remoteContent: true });
    assert.deepEqual([m(), reading.stick], [319, false]);
    reading.reflow(() => room(-420));
    assert.equal(m(), 99);
    // The reader's own scroll while the place is out of reach is the new place.
    ({ reading, feed, room, m } = shortRoom());
    reading.reflow(() => room(420));
    reading.scroll();
    feed.scrollTop -= 100; reading.scroll(); // read 100 px back in the tall room
    assert.equal(m(), 619);
    reading.mutate(() => true);
    reading.reflow(() => room(-420));
    assert.equal(m(), 619, 'the row the reader left in sight stays; nothing pulls the old place back');
});

test('a state poll while the feed\'s limit holds a lifted box back keeps the place for the keyboard\'s return', t => {
    const { reading, feed, box, place, moveBox, swipeBox, passage } = nestedRoom(t);
    feed.scrollHeight = 1000; // the timeline is the feed's last card: the feed is at its end
    swipeBox(-24);
    const read = passage();
    assert.deepEqual([feed.scrollTop, read], [600, 109]);
    // The keyboard goes: the feed grows 67 px and clamps to 533; the box clamps 411 -> 368 and the feed cannot lift it.
    const away = () => reading.reflow(() => { feed.clientHeight += 67; feed.scrollTop = feed.scrollTop; box.clientHeight = 132; box.scrollTop = box.scrollTop; });
    const back = () => reading.reflow(() => { feed.clientHeight -= 67; box.clientHeight = 65; });
    away();
    moveBox(0);
    reading.scroll();
    assert.deepEqual([feed.scrollTop, box.scrollTop, passage()], [533, 368, read + 110], 'at the limit the passage drops by what could not be applied');
    back();
    assert.deepEqual([feed.scrollTop, box.scrollTop, passage()], [600, 411, read], 'control: without a poll it comes back');
    away();
    moveBox(0);
    reading.scroll();
    reading.mutate(() => true); // one /api/state poll while the keyboard is away
    back();
    assert.deepEqual([feed.scrollTop, box.scrollTop, passage()], [600, 411, read], 'a poll that moved nothing rebases nothing');
    // A write that does move the box on screen (a row above it re-rendered 30 px taller, no passing line to hold) moves
    // the place by that much, with the box's own scroll and the lift standing in for its clamp untouched.
    away();
    moveBox(0);
    reading.scroll();
    place.passing = false;
    reading.mutate(() => { place.boxY += 30; feed.scrollHeight += 30; return true; });
    place.passing = true;
    assert.deepEqual([feed.scrollTop, passage()], [533, read + 140], 'the write moved the box 30 px down the screen');
    back();
    assert.deepEqual([box.scrollTop, passage()], [411, read + 30], 'the returning passage is those 30 px lower, no more');
});

test('a feed scrollbar drag follows where it leaves the feed and pages nothing', t => {
    const f = fixture(t), navigation = [];
    f.reading.bindGestures(direction => navigation.push(direction));
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.reading.stick = true;
    f.gesture({ type: 'pointerdown', target: f.feed });
    assert.equal(f.reading.stick, false, 'taking the feed scrollbar stops following');
    for (const [top, stick] of [[590, true], [300, false], [600, true]]) {
        f.feed.scrollTop = top; f.reading.scroll();
        assert.equal(f.reading.stick, stick, `dragged to ${top}`);
    }
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    f.feed.scrollTop = 300; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'a press on content is no drag: later scroll events do not decide following');
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.reading.cancel(); f.reading.stick = false;
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'explicit navigation ended the drag');
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.ready(); f.reading.request(); f.frame(); f.frame();
    assert.equal(f.reading.pending, false);
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'a restored place is no drag either');
    assert.deepEqual(navigation, [], 'a scrollbar drag reads no archive page');
});

test('a feed scrollbar drag ends with its own last scroll: later scrolls do not resume following', t => {
    const f = fixture(t);
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 600, clientHeight: 400 });
    const drag = (release, scrolls, end = [{ type: 'scrollend' }]) => {
        f.reading.stick = true; f.feed.scrollTop = 100;
        f.gesture({ type: 'pointerdown', target: f.feed });
        for (const top of scrolls) { f.feed.scrollTop = top; f.reading.scroll(); }
        f.gesture({ type: release, target: {} });
        return trailing => {
            for (const top of trailing) { f.feed.scrollTop = top; f.reading.scroll(); }
            for (const event of end) f.gesture(event);
            f.frame(); f.frame(); // a frame passes without another of the drag's scrolls
        };
    };
    // Chromium reports the release before the drag's last scroll; WebKit before any.
    for (const [release, during, after] of [['pointerup', [400], [300]], ['pointercancel', [], [300]]]) {
        drag(release, during)(after);
        assert.equal(f.reading.stick, false, `${release}: released away from the live edge`);
        f.feed.scrollTop = 580; f.reading.scroll(); // a taller viewport clamps the feed near its end
        assert.equal(f.reading.stick, false, `after ${release} a resize-driven scroll decides nothing`);
    }
    drag('pointerup', [300])([600]);
    assert.equal(f.reading.stick, true, 'the last scroll a released drag owes reaches the live edge: follow');
    drag('pointerup', [])([600]);
    assert.equal(f.reading.stick, true, 'the same when every scroll arrives after the release');
    // An engine without scrollend: the resize's own reflow ends the released drag.
    drag('pointerup', [300], [])([], []);
    f.reading.reflow();
    f.feed.scrollTop = 580; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'a resize after the release decides nothing');
    f.reading.stick = false; f.feed.scrollTop = 600;
    f.gesture({ type: 'pointerup', target: {} });
    assert.equal(f.reading.stick, false, 'a release without a feed drag decides nothing');
});

test('a resize clamp that reaches the feed before reflow is no part of a released drag', t => {
    const f = fixture(t);
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.reading.stick = true;
    // WebKit can report the drag's last scroll and its scrollend while the thumb is still held:
    // no scrollend follows the release.
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.feed.scrollTop = 500; f.reading.scroll(); f.gesture({ type: 'scrollend' });
    f.gesture({ type: 'pointerup', target: {} });
    assert.equal(f.reading.stick, false, 'released 100px above the live edge');
    // A taller window clamps the feed to its end; that scroll arrives before the resize's reflow.
    Object.assign(f.feed, { clientHeight: 550, scrollTop: 450 }); f.reading.scroll();
    assert.equal(f.reading.stick, false, 'the clamp to the end decides nothing');
    const restored = f.restored();
    f.reading.reflow();
    assert.equal(f.restored(), restored + 1, 'reflow keeps the passage instead of writing the feed to its end');
    assert.equal(f.reading.stick, false);
    // In an unchanged box the drag's own last scroll, arriving after the release, still decides.
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.feed.scrollTop = 200; f.reading.scroll();
    f.gesture({ type: 'pointerup', target: {} });
    f.feed.scrollTop = 450; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'released at the live edge: follow');
});

test('a released drag decides by its last scroll when WebKit ends each batch of them with a scrollend', t => {
    const f = fixture(t);
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.gesture({ type: 'pointerup', target: {} });
    // The drag's scrolls arrive after the release in two batches a frame apart, each ending in a scrollend.
    f.feed.scrollTop = 450; f.reading.scroll(); f.gesture({ type: 'scrollend' }); f.frame();
    f.feed.scrollTop = 600; f.reading.scroll(); f.gesture({ type: 'scrollend' });
    assert.equal(f.reading.stick, true, 'its last batch reaches the live edge: follow');
    f.frame(); f.frame();
    f.feed.scrollTop = 520; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'once a frame passes without another of its scrolls, a scroll decides nothing');
});

test('the app\'s own write while a drag is released leaves following as the release decided', t => {
    // An anchor that stays put: only the browser's clamp moves the feed.
    const f = fixture(t, true, null);
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    const release = (scrollend) => {
        Object.assign(f.feed, { scrollHeight: 1200, scrollTop: 100, clientHeight: 400 });
        f.gesture({ type: 'pointerdown', target: f.feed });
        f.feed.scrollTop = 720; f.reading.scroll(); // released 80px above the live edge
        f.gesture({ type: 'pointerup', target: {} });
        if (scrollend) f.gesture({ type: 'scrollend' });
        assert.equal(f.reading.stick, false);
    };
    // A settled Main question copy below the passage retires (chat_decision.js removeMirror): the feed
    // shrinks and the browser clamps it to its end; that clamp's scroll comes after the write.
    const retire = () => f.reading.mutate(() => {
        Object.assign(f.feed, { scrollHeight: 1000, scrollTop: Math.min(f.feed.scrollTop, 600) });
        return true;
    });
    // Chromium: the drag's scrollend came; the retirement lands before its owed scrolls end a frame later.
    release(true); retire(); f.reading.scroll();
    assert.equal(f.reading.stick, false, 'the clamp a removal causes does not turn on following');
    // WebKit: the drag's last scroll and scrollend came before the release; the drag stays released.
    release(false); for (let i = 0; i < 5; i++) f.frame();
    retire(); f.reading.scroll();
    assert.equal(f.reading.stick, false, 'nor much later, while WebKit leaves the drag released');
    // A scroll the released drag still owes moves the feed: it still decides, after the app's write too.
    release(false); f.reading.mutate(() => { f.feed.scrollHeight = 1300; return true; });
    f.feed.scrollTop = 880; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'its own last scroll at the live edge: follow');
});

test('a refresh that moves nothing before a released drag\'s queued scroll leaves that scroll deciding', t => {
    const f = fixture(t, true, null);
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.gesture({ type: 'pointerup', target: {} });
    assert.equal(f.reading.stick, false, 'the release still sees the feed where the drag began');
    // WebKit moves the feed to where the drag left it before that scroll event comes; an ordinary
    // census refresh (/api/state, a write that changes nothing here) runs in between.
    f.feed.scrollTop = 600;
    f.reading.mutate(() => false);
    f.reading.scroll();
    assert.equal(f.reading.stick, true, 'released at the live edge: follow');
});

test('an animated downward key or wheel follows where its own scrolls leave the feed', t => {
    const f = fixture(t);
    const body = f.feed.ownerDocument.body;
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    // The engine has only begun animating End when the gesture's frame runs.
    f.gesture({ type: 'keydown', key: 'End', target: body }); f.feed.scrollTop = 150; f.frame();
    assert.equal(f.reading.stick, false, 'still moving, not yet at the live edge');
    for (const top of [400, 600]) { f.feed.scrollTop = top; f.reading.scroll(); }
    assert.equal(f.reading.stick, true, 'its last scroll reaches the live edge: follow');
    f.gesture({ type: 'scrollend' });
    f.gesture({ type: 'wheel', deltaY: -1 }); f.feed.scrollTop = 150; f.frame();
    f.gesture({ type: 'wheel', deltaY: 1 }); f.feed.scrollTop = 320; f.frame();
    for (const top of [380, 450]) { f.feed.scrollTop = top; f.reading.scroll(); }
    f.gesture({ type: 'scrollend' }); f.frame(); f.frame();
    assert.equal(f.reading.stick, false, 'an animation that ends short of the edge does not follow');
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'a frame after its scrollend a scroll alone decides nothing');
    // WebKit reports scrollend after every step of an animated PageDown.
    f.gesture({ type: 'wheel', deltaY: -1 }); f.feed.scrollTop = 100; f.frame();
    f.gesture({ type: 'keydown', key: 'PageDown', target: body }); f.frame();
    for (const top of [300, 500, 594]) {
        f.feed.scrollTop = top; f.reading.scroll(); f.gesture({ type: 'scrollend' }); f.frame();
    }
    assert.equal(f.reading.stick, true, 'the step that lands at the live edge still decides');
    f.frame();
    f.feed.scrollTop = 300; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'once a frame passes without a step, a scroll decides nothing');
    f.gesture({ type: 'keydown', key: 'PageUp', target: body }); f.frame();
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'an upward gesture owes no scrolls that could resume following');
});

test('a settled gesture ends only the scrolls it owed, not those a newer gesture has yet to deliver', t => {
    const f = fixture(t);
    const body = f.feed.ownerDocument.body;
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    // PageDown ends short of the live edge; End is pressed before that settle's frames run.
    f.gesture({ type: 'keydown', key: 'PageDown', target: body }); f.frame();
    f.feed.scrollTop = 300; f.reading.scroll(); f.gesture({ type: 'scrollend' });
    f.gesture({ type: 'keydown', key: 'End', target: body });
    f.frame(); f.frame(); f.frame();
    // End's own scrolls arrive only now.
    for (const top of [450, 600]) { f.feed.scrollTop = top; f.reading.scroll(); }
    assert.equal(f.reading.stick, true, 'End reaches the live edge: follow');
    f.reading.mutate(() => { f.feed.scrollHeight = 1100; }, { remoteContent: true });
    assert.equal(f.feed.scrollTop, 1100, 'a new reply is followed');
    f.gesture({ type: 'scrollend' }); f.frame(); f.frame();
    f.feed.scrollTop = 300; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'End\'s own settle still ends what it owed');
});

test('a resize while a downward key still animates neither writes under it nor ends it; it lays out once the scrolls rest', t => {
    const f = fixture(t);
    const body = f.feed.ownerDocument.body;
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    // End has only begun to move the feed when an image loads; the composer grows near its end.
    f.gesture({ type: 'keydown', key: 'End', target: body }); f.feed.scrollTop = 150; f.frame();
    const restored = f.restored();
    f.reading.reflow();
    f.feed.scrollTop = 400; f.reading.scroll(); f.frame();
    f.reading.mutate(() => true, { remoteContent: true });
    assert.deepEqual([f.feed.scrollTop, f.restored()], [400, restored], 'mutate while owed neither restores nor writes scrollTop');
    f.feed.scrollTop = 580; f.reading.scroll(); f.frame();
    f.reading.reflow();
    assert.deepEqual([f.feed.scrollTop, f.restored()], [580, restored], 'nothing writes under the animated scroll: a write stops it short');
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'its own last scroll still reaches the live edge: follow');
    f.feed.scrollHeight = 1040;
    f.gesture({ type: 'scrollend' }); f.frame();
    assert.equal(f.feed.scrollTop, 600, 'not before a frame passes without another scroll');
    f.frame();
    assert.equal(f.feed.scrollTop, 1040, 'once its scrolls rest, the resize lays out at the live edge it follows');
    f.feed.scrollTop = 300; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'the resize is no part of the gesture: a later scroll decides nothing');
    // An engine without scrollend: once the owed scrolls rest, the resize ends them.
    f.gesture({ type: 'keydown', key: 'PageUp', target: body }); f.feed.scrollTop = 500; f.frame();
    f.gesture({ type: 'keydown', key: 'PageDown', target: body }); f.feed.scrollTop = 520; f.frame();
    f.feed.scrollTop = 560; f.reading.scroll();
    f.reading.reflow(); f.frame(); f.frame();
    assert.equal(f.restored(), restored + 1, 'resting short of the live edge, the place is kept');
    f.feed.scrollTop = 640; f.reading.scroll();
    assert.equal(f.reading.stick, false, 'a resize-driven scroll after it decides nothing');
});

test('a resize that waited on the owed scrolls keeps where they end; at rest it keeps the place read before it', t => {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const frame = () => frames.splice(0).forEach(callback => callback());
    const heard = new Map(), listen = (type, fn) => heard.set(type, fn);
    const ownerDocument = { body: {}, documentElement: {}, addEventListener: listen, removeEventListener() {} };
    const feed = { scrollTop: 100, scrollHeight: 1000, clientHeight: 400, clientWidth: 400, ownerDocument,
        addEventListener() {}, removeEventListener() {}, contains: node => node === feed || Boolean(node?.inFeed) };
    // The anchor is the place read: restoring it puts the feed back there.
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors: { capture: () => ({ top: feed.scrollTop }), restore: anchor => { feed.scrollTop = anchor.top; return true; },
            serialize: () => null }, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.bindGestures(() => {});
    reading.remember(true);
    heard.get('pointerdown')({ type: 'pointerdown', target: { inFeed: true } });
    heard.get('keydown')({ type: 'keydown', key: 'PageDown', target: ownerDocument.body }); feed.scrollTop = 150; frame();
    feed.clientWidth = 380; reading.reflow(); // the window narrows while PageDown animates
    for (const top of [300, 450]) { feed.scrollTop = top; reading.scroll(); frame(); }
    frame(); frame();
    assert.equal(feed.scrollTop, 450, 'not pulled back to where the resize found the reader');
    feed.clientWidth = 360; feed.scrollTop = 470; reading.scroll(); // narrowed at rest: its rewrap moved the feed
    reading.reflow();
    assert.equal(feed.scrollTop, 450, 'at rest a resize keeps the place read before it');
});

test('rows are lazy only while the width keeps moving; once it holds, every row is laid out and the place re-applied', t => {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const frame = () => frames.splice(0).forEach(callback => callback());
    const marks = new Set(), settled = [];
    const feed = { scrollTop: 100, scrollHeight: 1000, clientHeight: 400, clientWidth: 385, offsetWidth: 400,
        ownerDocument: {}, addEventListener() {}, removeEventListener() {}, contains: node => node === feed,
        toggleAttribute: (name, on) => { if (on) marks.add(name); else marks.delete(name); return on; },
        dispatchEvent: event => { settled.push([event.type, event.bubbles, feed.scrollTop]); return true; } };
    const width = value => Object.assign(feed, { offsetWidth: value, clientWidth: value ? value - 15 : 0 });
    // Laying the rows out again at the held width moves the passage; restoring the place puts it back.
    let restores = 0;
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors: { capture: () => ({ top: 100 }), restore: anchor => { restores++; feed.scrollTop = anchor.top; return true; },
            serialize: () => null }, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.stick = false; reading.remember(true);
    const lazy = () => marks.has('data-resizing');
    feed.clientWidth = 400; reading.reflow();
    assert.equal(lazy(), false, 'a scrollbar coming or going changes only the content width: no move');
    width(380); reading.reflow(); frame();
    assert.equal(lazy(), false, 'one move (a window jump, a layout switch) is laid out at once');
    for (let i = 0; i < 3; i++) frame();
    assert.equal(frames.length, 0, 'and nothing keeps counting after it');
    assert.deepEqual(settled, [], 'nothing was skipped, so nothing waits to be measured');
    width(360); reading.reflow(); frame();
    width(340); reading.reflow();
    assert.equal(lazy(), true, 'the width keeps moving: rows off the screen keep their size');
    for (let i = 0; i < 6; i++) frame();
    width(320); reading.reflow(); // the drag goes on: the hold starts again
    for (let i = 0; i < 11; i++) frame();
    assert.equal(lazy(), true, 'still within the hold after the last move');
    feed.scrollTop = 140; // the rows above the passage were laid out at the new width
    const before = restores;
    frame();
    assert.equal(lazy(), false, 'the width held: every row is laid out again');
    assert.deepEqual([restores - before, feed.scrollTop], [1, 100], 'and the place is re-applied in that same frame');
    assert.deepEqual(settled, [['ouro:feed-width-settled', true, 100]],
        'then the rows are announced as laid out, once, for whatever measures them (the answer folds)');
    for (let i = 0; i < 20; i++) frame();
    assert.equal(frames.length, 0, 'nothing keeps counting at rest');
    // A feed hidden and shown again at another width (another room was open) is not a move.
    width(0); reading.reflow(); frame();
    width(300); reading.reflow(); frame();
    assert.equal(lazy(), false);
    // Following the newest message, the held width lands at the new bottom.
    reading.stick = true;
    width(310); reading.reflow(); frame();
    width(320); reading.reflow();
    assert.equal(lazy(), true);
    for (let i = 0; i < 11; i++) frame();
    feed.scrollHeight = 1300;
    frame();
    assert.deepEqual([lazy(), feed.scrollTop], [false, 1300]);
});

test('reading inside a box ends what an earlier feed gesture still owes, not a scrollbar still held', t => {
    const f = fixture(t);
    f.feed.ownerDocument.defaultView = { getComputedStyle: node => ({ overflowY: node.overflowY || 'visible' }) };
    const box = { inFeed: true, parentElement: f.feed, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400 };
    const text = { inFeed: true, parentElement: box };
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    let at = 0;
    const read = ({ inBox, beforeFrame = false }) => {
        Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
        // An animated wheel down the feed has only begun to move it.
        f.gesture({ type: 'wheel', deltaY: 1, timeStamp: at += 1000 }); f.feed.scrollTop = 250;
        if (!beforeFrame) f.frame();
        const generation = f.reading.generation;
        if (inBox) f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: at + 500 });
        const cancelled = f.reading.generation !== generation;
        f.frame();
        f.feed.scrollTop = 600; f.reading.scroll(); // the rest of the feed's animation
        const restored = f.restored();
        f.reading.mutate(() => { f.feed.scrollHeight = 1100; }, { remoteContent: true });
        return { stick: f.reading.stick, followed: f.feed.scrollTop === 1100, kept: f.restored() > restored, cancelled };
    };
    assert.deepEqual(read({ inBox: false }), { stick: true, followed: true, kept: false, cancelled: false },
        'control: the feed gesture\'s own last scroll reaches the live edge');
    assert.deepEqual(read({ inBox: true }), { stick: false, followed: false, kept: true, cancelled: false },
        'reading up inside a box ends the feed animation\'s claim; no layout is cancelled');
    assert.deepEqual(read({ inBox: true, beforeFrame: true }), { stick: false, followed: false, kept: true, cancelled: false },
        'the same before the feed gesture\'s own frame has run');
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100 });
    f.gesture({ type: 'pointerdown', target: f.feed });
    f.feed.scrollTop = 300; f.reading.scroll();
    f.gesture({ type: 'wheel', deltaY: -1, target: text, timeStamp: at += 1000 }); f.frame();
    f.feed.scrollTop = 600; f.reading.scroll();
    assert.equal(f.reading.stick, true, 'a scrollbar still held goes on deciding where it leaves the feed');
});

test('a box gesture supersedes a ↓ still awaiting history; with nothing pending it cancels nothing', t => {
    const f = fixture(t);
    f.feed.ownerDocument.defaultView = { getComputedStyle: node => ({ overflowY: node.overflowY || 'visible' }) };
    const body = { inFeed: true, parentElement: f.feed, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400,
        getBoundingClientRect: () => ({ top: 20 }) };
    const text = { inFeed: true, parentElement: body };
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    const current = f.reading.claim(); // ↓ pressed; its latest page is still loading
    f.reading.wait(current);
    f.reading.followAfterLayout(); f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, f.feed.scrollHeight, 'the retained end is already reached');
    assert.equal(current(), true);
    f.gesture({ type: 'wheel', deltaY: 1, target: text, timeStamp: 1000 }); f.frame();
    assert.equal(current(), false, 'reading on inside a box keeps the reader where they are');
    assert.equal(f.reading.returning, false, 'the pending load loses its landing too');
    assert.equal(f.reading.stick, false, 'a shared read must not follow after the pending return was cancelled');
    // With no ↓ waiting, box scrolling leaves layout already on its way alone.
    const generation = f.reading.generation;
    f.gesture({ type: 'wheel', deltaY: 1, target: text, timeStamp: 2000 }); f.frame();
    assert.equal(f.reading.generation, generation);
    // A ↓ that reached the present releases its claim: a later box gesture
    // cannot cancel a follow a sent message started.
    const done = f.reading.claim();
    f.reading.followAfterLayout(); f.frame(); f.frame();
    assert.equal(done(), true);
    f.feed.scrollTop = 100; f.reading.followAfterLayout();
    f.gesture({ type: 'wheel', deltaY: 1, target: text, timeStamp: 3000 });
    f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, f.feed.scrollHeight, 'the sent message is still followed');
});

test('an unread opening still owed yields to a box gesture after its claim reached the present, and lands only its own', t => {
    const f = fixture(t);
    f.feed.ownerDocument.defaultView = { getComputedStyle: node => ({ overflowY: node.overflowY || 'visible' }) };
    const body = { inFeed: true, parentElement: f.feed, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400 };
    const text = { inFeed: true, parentElement: body };
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.reading.claim(true);
    const stale = f.reading.arriving;
    f.reading.claim(true);  // opened again
    const token = f.reading.arriving;
    f.reading.land(() => -50, stale);
    assert.equal(f.reading.arriving, token, 'a superseded opening spends nothing');
    f.reading.followAfterLayout(); f.frame(); f.frame();  // the opening's ↓ reached the present
    f.gesture({ type: 'wheel', deltaY: 1, target: text, timeStamp: 1000 }); f.frame();
    assert.equal(f.reading.arriving, null, 'reading inside a box ends the opening still owed');
    f.feed.scrollTop = 100; f.reading.land(() => -50, token);
    assert.equal(f.feed.scrollTop, 100, 'its later landing moves no one');
});

test('↓ is drawn busy only while its own claim waits: a gesture or newer navigation ends it, a superseded end changes nothing', t => {
    const previous = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = () => {};
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    let drawn = 0, live = true;
    const feed = { scrollTop: 0, scrollHeight: 1000, clientHeight: 400, clientWidth: 400, offsetWidth: 400 };
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => live, ready: () => true,
        anchors: { serialize: () => null, capture: () => null, restore: () => true },
        changed() {}, afterWrite() {}, activity() {}, updateButton() { drawn++; } });
    const first = reading.claim();
    reading.wait(first);
    assert.deepEqual([reading.returning, drawn], [true, 1], 'drawn busy at once, before any read answers');
    reading.cancel();  // the reader's own gesture
    assert.deepEqual([reading.returning, drawn], [false, 2], 'and drawn idle the moment it is superseded');
    reading.wait(first, false);
    assert.equal(drawn, 2, 'its read settling later draws nothing');
    const second = reading.claim();
    reading.wait(second);
    const third = reading.claim();  // another navigation, or ↓ pressed again as a fresh Retry
    assert.equal(reading.returning, false);
    reading.wait(third);
    reading.wait(second, false);
    assert.equal(reading.returning, true, 'an older ↓ settling leaves the newer one busy');
    reading.wait(third, false);
    assert.equal(reading.returning, false);
    const fourth = reading.claim();
    live = false;  // destroyed meanwhile
    const before = drawn;
    reading.wait(fourth);
    assert.deepEqual([reading.returning, drawn], [false, before], 'a destroyed room is never drawn busy');
});

test('saved Review readiness distinguishes pending/error from confirmed absence and trails revisions', async () => {
    const reads = [], settled = [];
    const hydrator = createReviewHydrator({
        fetchDetail: id => new Promise((resolve, reject) => reads.push({ id, resolve, reject })),
        applyDetail: () => true,
        onSettled: id => settled.push([id, hydrator.ready(id)]),
    });
    const first = hydrator.hydrate('bookmark', 'a'.repeat(64));
    const unrelated = hydrator.hydrate('unrelated');
    assert.equal(hydrator.ready('bookmark'), false);
    assert.equal(hydrator.ready('no-detail-needed'), true);
    await Promise.resolve();
    reads[0].reject(new Error('temporarily unavailable'));
    await first;
    assert.equal(hydrator.ready('bookmark'), false, 'failure preserves the bookmark for Retry');
    const retry = hydrator.hydrate('bookmark', 'a'.repeat(64));
    const trailing = hydrator.hydrate('bookmark', 'b'.repeat(64));
    await Promise.resolve();
    reads[2].resolve({});
    await retry;
    assert.equal(hydrator.ready('bookmark'), false, 'a queued necessary revision is still pending');
    reads[3].resolve(null);
    await trailing;
    assert.equal(hydrator.ready('bookmark'), true, 'successful absence permits explicit fallback');
    assert.equal(hydrator.ready('unrelated'), false, 'unrelated detail is still held');
    assert.deepEqual(settled.filter(([id]) => id === 'bookmark'), [
        ['bookmark', false], ['bookmark', false], ['bookmark', true],
    ]);
    hydrator.clear();
    reads[1].resolve({});
    await unrelated;
    assert.equal(settled.some(([id]) => id === 'unrelated'), false, 'disposed requests cannot resume positioning');
});

test('a sideways finger (the drawer, a swipe to reply) is not reading; a vertical one past the slop is', t => {
    const f = fixture(t), navigation = [];
    f.reading.bindGestures(direction => navigation.push(direction));
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 600, clientHeight: 400 });
    f.reading.stick = true;
    // 80 px sideways with a thumb's 6 px of drift: the drawer's or the row swipe's, following stays on.
    f.gesture({ type: 'touchstart', touches: [{ clientX: 40, clientY: 300 }] });
    for (const [x, y] of [[44, 302], [60, 304], [90, 306], [120, 306]]) {
        f.gesture({ type: 'touchmove', touches: [{ clientX: x, clientY: y }] }); f.frame();
    }
    assert.equal(f.reading.stick, true, 'a sideways drag keeps following the newest message');
    // Decided sideways, it stays the drawer's even if the thumb then wanders down.
    f.gesture({ type: 'touchmove', touches: [{ clientX: 130, clientY: 380 }] }); f.frame();
    assert.equal(f.reading.stick, true);
    assert.deepEqual(navigation, []);
    // The same thumb moving down the feed past the slop reads back.
    f.gesture({ type: 'touchstart', touches: [{ clientX: 200, clientY: 300 }] });
    f.gesture({ type: 'touchmove', touches: [{ clientX: 202, clientY: 306 }] }); f.frame();
    assert.equal(f.reading.stick, true, 'the drift before the axis is decided is no reading');
    f.gesture({ type: 'touchmove', touches: [{ clientX: 203, clientY: 360 }] }); f.frame();
    assert.equal(f.reading.stick, false, 'a vertical move past the slop is reading back: following ends');
    assert.deepEqual(navigation, [-1]);
});

test('a second finger makes the touch a pinch: no reading, paging or dismissal until a new one-finger touch', t => {
    const f = fixture(t), navigation = [];
    f.reading.bindGestures(direction => navigation.push(direction));
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 600, clientHeight: 400 });
    f.reading.stick = true;
    const fingers = (...points) => points.map(([clientX, clientY]) => ({ clientX, clientY }));
    // Fingers spread apart vertically: the first one moves down the feed as a reading swipe would.
    f.gesture({ type: 'touchstart', touches: fingers([200, 300]) });
    f.gesture({ type: 'touchstart', touches: fingers([200, 300], [200, 420]) });
    for (const [a, b] of [[290, 440], [270, 470], [240, 500]]) {
        f.gesture({ type: 'touchmove', touches: fingers([200, a], [200, b]) }); f.frame();
    }
    // One finger lifted: the rest of the pinch is still no reading.
    f.gesture({ type: 'touchmove', touches: fingers([200, 300]) }); f.frame();
    f.gesture({ type: 'touchmove', touches: fingers([200, 380]) }); f.frame();
    assert.equal(f.reading.stick, true, 'a pinch only magnifies: following stays');
    assert.deepEqual(navigation, []);
    // A pinch that began as a one-finger read ends that read where the second finger lands.
    f.gesture({ type: 'touchstart', touches: fingers([200, 300]) });
    f.gesture({ type: 'touchmove', touches: fingers([200, 306]) }); f.frame();
    f.gesture({ type: 'touchmove', touches: fingers([200, 360]) }); f.frame();
    assert.deepEqual(navigation, [-1], 'a new one-finger touch reads again');
    f.reading.stick = true;
    f.gesture({ type: 'touchstart', touches: fingers([200, 360], [260, 200]) });
    f.gesture({ type: 'touchmove', touches: fingers([200, 420], [260, 150]) }); f.frame();
    assert.deepEqual(navigation, [-1]);
    assert.equal(f.reading.stick, true);
});

test('a write between a feed box change and its reflow keeps a reader following the newest message', async () => {
    // A keyboard room that shrinks leaves the feed short of its newest row until the resize's reflow, a frame later. A
    // write landing in that frame (a state poll's) found the reader "away" and ended the following for good.
    const { createChatReadingPosition: create } = await import('../modules/chat_reading_position.js');
    const feed = { clientWidth: 390, offsetWidth: 390, clientHeight: 253, scrollHeight: 1000, top: 0,
        get scrollTop() { return this.top; },
        set scrollTop(value) { this.top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); } };
    const state = create({ feed, visible: () => true, alive: () => true, ready: () => true, changed() {}, afterWrite() {},
        activity() {}, updateButton() {}, anchors: { capture: () => feed.scrollTop, restore: () => true, serialize: () => null } });
    state.reflow();
    assert.deepEqual([state.stick, feed.scrollTop], [true, 747], 'the room opens following its newest message');
    feed.clientHeight = 60;  // the room shrinks: the newest row is 193 px below until the reflow
    assert.equal(state.nearBottom(), false);
    assert.equal(state.mutate(() => true), true);
    assert.equal(state.stick, true, 'the write is no reading back');
    state.reflow();
    assert.deepEqual([state.stick, feed.scrollTop], [true, 940], 'the reflow brings the newest row back');
    // In the box the last reflow placed, a reader found away from the newest row is away: a write keeps the place.
    feed.scrollTop = 300;
    state.mutate(() => true);
    assert.deepEqual([state.stick, feed.scrollTop], [false, 300]);
});

for (const supersede of [false, true]) for (const readBack of [false, true]) {
    test(`a return ${supersede ? 'superseded by ↓' : 'completed'} settles its box before ${readBack ? 'focus reads back' : 'more rows arrive'}`, t => {
        const previous = globalThis.requestAnimationFrame, frames = [];
        globalThis.requestAnimationFrame = fn => frames.push(fn);
        t.after(() => { globalThis.requestAnimationFrame = previous; });
        const frame = () => frames.splice(0).forEach(fn => fn());
        const feed = { clientWidth: 390, offsetWidth: 390, clientHeight: 0, scrollHeight: 1000, top: 0,
            get scrollTop() { return this.top; },
            set scrollTop(value) { this.top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); } };
        const state = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
            changed() {}, afterWrite() {}, activity() {}, updateButton() {},
            anchors: { capture: () => feed.scrollTop, restore: top => { feed.scrollTop = top; return true; }, serialize: () => null } });
        state.request(); // resume restores a follower even before its read finishes
        feed.clientHeight = 400;
        state.reflow(); // resize arrives while that restoration owns positioning
        if (supersede) { state.claim(); state.followAfterLayout(); }
        frame(); frame();
        assert.deepEqual([state.pending, feed.scrollTop], [false, 600]);
        // Tab can scroll a focused disclosure into view without a wheel/PageUp intent.
        if (readBack) { feed.scrollTop = 300; state.scroll(); }
        state.mutate(() => { feed.scrollHeight += 100; });
        const expected = [!readBack, readBack ? 300 : 700];
        assert.deepEqual([state.stick, feed.scrollTop], expected, 'the settled box distinguishes later reading from following');
        state.reflow();
        assert.deepEqual([state.stick, feed.scrollTop], expected, 'the following reflow keeps the same place');
    });
}

test('a 300 px disappearance above the viewport during a 60 px downward scroll keeps its unapplied shift across scroll()', t => {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = fn => frames.push(fn);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const frame = () => frames.splice(0).forEach(fn => fn());
    const heard = new Map(), docHeard = new Map();
    const doc = { body: {}, documentElement: {}, addEventListener: (k, fn) => docHeard.set(k, fn), removeEventListener() {} };
    const feed = { clientWidth: 400, offsetWidth: 400, clientHeight: 500, offsetHeight: 500, scrollHeight: 3000, scrollTop: 1000,
        ownerDocument: doc, addEventListener: (k, fn) => heard.set(k, fn), removeEventListener() {},
        getBoundingClientRect: () => ({ top: 0, bottom: 500, left: 0, right: 400, width: 400, height: 500 }),
        contains: node => node === feed || Boolean(node?.inFeed) };
    let rowY = 1390; // starts at +390 px when scrollTop = 1000
    const row = { inFeed: true, parentElement: feed, isConnected: true, historyId: 'chat:r',
        getBoundingClientRect: () => ({ top: rowY - feed.scrollTop, bottom: rowY - feed.scrollTop + 100, width: 400, height: 100 }) };
    const anchors = {
        serialize: () => null,
        capture: () => ({ node: row, historyId: row.historyId, offset: row.getBoundingClientRect().top }),
        restore: a => { if (!a?.node) return false; feed.scrollTop += a.node.getBoundingClientRect().top - a.offset; return true; },
    };
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.bindGestures(() => {});
    reading.stick = false; reading.remember(true);
    docHeard.get('pointerdown')({ type: 'pointerdown', target: { inFeed: true } });
    docHeard.get('keydown')({ type: 'keydown', key: 'PageDown', target: doc.body, timeStamp: 10 });
    frame();
    // While scrolling 60 px down, 300 px disappear above the viewport; the next scroll() updates viewportAnchor.
    reading.mutate(() => { rowY -= 300; feed.scrollHeight -= 300; return true; }, { remoteContent: true });
    assert.equal(feed.scrollTop, 1000, 'scrollTop is not written mid-animation');
    feed.scrollTop = 1060;
    reading.scroll();
    heard.get('scrollend')({ type: 'scrollend' });
    frame(); frame();
    assert.equal(feed.scrollTop, 760);
    assert.equal(row.getBoundingClientRect().top, 330, 'row from +390 px lands at +330 px after 60 px scroll, not +30 px');
});

test('an unchanged repaint and two frames before End delivers its first scroll do not cancel End auto-follow', t => {
    const f = fixture(t, true, null);
    const body = f.feed.ownerDocument.body;
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    f.gesture({ type: 'pointerdown', target: { inFeed: true } });
    f.gesture({ type: 'keydown', key: 'End', target: body, timeStamp: 20 });
    // Repaint with no shift and two frames pass before the engine's first scroll event of End arrives.
    f.reading.mutate(() => true, { remoteContent: true });
    f.frame(); f.frame();
    for (const top of [350, 600]) { f.feed.scrollTop = top; f.reading.scroll(); }
    f.gesture({ type: 'scrollend' });
    f.frame(); f.frame();
    assert.equal(f.reading.stick, true, 'End reaches the live edge with auto-follow enabled');
    f.reading.mutate(() => { f.feed.scrollHeight = 1140; return true; }, { remoteContent: true });
    const clampedGap = Math.max(0, f.feed.scrollHeight - Math.min(f.feed.scrollTop, f.feed.scrollHeight - f.feed.clientHeight) - f.feed.clientHeight);
    assert.equal(clampedGap, 0, 'next 140 px reply leaves zero gap');
});

test('scrolling a nested card timeline by 60 px after PageDown and an empty repaint or reflow leaves the outer feed at 150 px', t => {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = fn => frames.push(fn);
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const frame = () => frames.splice(0).forEach(fn => fn());
    const feedHeard = new Map(), docHeard = new Map();
    const doc = { body: {}, documentElement: {}, defaultView: { getComputedStyle: () => ({ overflowY: 'auto' }) },
        addEventListener: (k, fn) => docHeard.set(k, fn), removeEventListener() {} };
    const feed = { clientWidth: 400, offsetWidth: 400, clientHeight: 500, offsetHeight: 500, scrollHeight: 2000, scrollTop: 150,
        ownerDocument: doc, addEventListener: (k, fn) => feedHeard.set(k, fn), removeEventListener() {},
        getBoundingClientRect: () => ({ top: 0, bottom: 500, left: 0, right: 400, width: 400, height: 500 }),
        contains: node => node === feed || node === timeline || node === line };
    const timeline = { parentElement: feed, isConnected: true, clientHeight: 200, scrollHeight: 600, scrollTop: 0,
        getBoundingClientRect: () => ({ top: 180 - feed.scrollTop, bottom: 380 - feed.scrollTop, width: 400, height: 200 }) };
    const line = { parentElement: timeline, isConnected: true,
        getBoundingClientRect: () => ({ top: 180 - feed.scrollTop + 20 - timeline.scrollTop, width: 400, height: 40 }) };
    const anchors = {
        serialize: () => null,
        capture: () => ({ node: line, offset: line.getBoundingClientRect().top }),
        restore: a => { if (!a?.node) return false; feed.scrollTop += a.node.getBoundingClientRect().top - a.offset; return true; },
    };
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors, changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.bindGestures(() => {});
    reading.stick = false; reading.remember(true);

    for (const emptyStep of [() => reading.mutate(() => false), () => reading.reflow()]) {
        feed.scrollTop = 150; timeline.scrollTop = 0; reading.remember(true);
        docHeard.get('pointerdown')({ type: 'pointerdown', target: feed });
        docHeard.get('pointerup')({ type: 'pointerup', target: feed });
        docHeard.get('pointerdown')({ type: 'pointerdown', target: { inFeed: true, parentElement: feed } });
        docHeard.get('keydown')({ type: 'keydown', key: 'PageDown', target: doc.body, timeStamp: 100 });
        emptyStep();
        // Reader scrolls the inner card timeline by 60 px.
        feedHeard.get('wheel')({ type: 'wheel', deltaY: 60, target: line, timeStamp: 116 });
        timeline.scrollTop += 60;
        feedHeard.get('scroll')({ type: 'scroll', target: timeline, timeStamp: 118 });
        frame(); frame(); frame();
        assert.equal(feed.scrollTop, 150, 'outer feed stays at 150 px instead of shifting to 90 px');
    }
});

function runGestureSeries({ kind, count, spacingMs, withScrollEnd = true, setupFeed, mutateAfterFirst }) {
    const previous = globalThis.requestAnimationFrame, frames = [];
    globalThis.requestAnimationFrame = fn => frames.push(fn);
    try {
        const frame = () => frames.splice(0).forEach(fn => fn());
        const feedHeard = new Map(), docHeard = new Map();
        const doc = { body: {}, documentElement: {}, addEventListener: (k, fn) => docHeard.set(k, fn), removeEventListener() {} };
        const { feed, anchors, targetRow } = setupFeed(doc, feedHeard);
        const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
            anchors: { serialize: () => null, capture: anchors.captureVisibleTimelineAnchor, restore: anchors.restoreVisibleTimelineAnchor },
            changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
        reading.bindGestures(() => {});
        reading.stick = false; reading.remember(true);

        const queue = [];
        let touchY = 500;
        if (kind === 'touch') {
            docHeard.get('pointerdown')({ type: 'pointerdown', target: targetRow, timeStamp: 98 });
            feedHeard.get('touchstart')({ type: 'touchstart', target: targetRow, touches: [{ clientX: 200, clientY: touchY }], timeStamp: 99 });
        }
        for (let i = 0; i < count; i++) {
            const at = i * spacingMs;
            queue.push({ at, order: 1, run: () => {
                if (kind === 'wheel') {
                    feedHeard.get('wheel')({ type: 'wheel', deltaY: 60, target: targetRow, timeStamp: 100 + at });
                } else {
                    touchY -= 60;
                    feedHeard.get('touchmove')({ type: 'touchmove', target: targetRow, touches: [{ clientX: 200, clientY: touchY }], timeStamp: 100 + at });
                }
                if (i === 0) {
                    const topBeforeMutate = feed.scrollTop;
                    mutateAfterFirst(reading);
                    assert.equal(feed.scrollTop, topBeforeMutate, 'mutate does not write scrollTop mid-gesture');
                    if (kind === 'touch') docHeard.get('pointercancel')({ type: 'pointercancel', target: targetRow, timeStamp: 100 + at });
                }
            } });
            queue.push({ at: at + 32, order: 2, run: () => { feed.scrollTop += 60; reading.scroll(); } });
        }
        const lastScrollAt = (count - 1) * spacingMs + 32;
        if (kind === 'touch') queue.push({ at: lastScrollAt, order: 3, run: () => docHeard.get('pointerup')({ type: 'pointerup', target: {}, timeStamp: 100 + lastScrollAt }) });
        if (withScrollEnd) queue.push({ at: lastScrollAt + 8, order: 4, run: () => feedHeard.get('scrollend')({ type: 'scrollend' }) });
        for (let t = 16; t <= lastScrollAt + 160; t += 16) queue.push({ at: t, order: 0, run: frame });
        queue.sort((a, b) => (a.at - b.at) || (a.order - b.order));
        for (const item of queue) item.run();
        return feed;
    } finally {
        globalThis.requestAnimationFrame = previous;
    }
}

function makeSeriesDom(doc, feedHeard, { scrollTop, scrollHeight = 5000, clientHeight = 750 }) {
    const children = [];
    const feed = {
        clientWidth: 400, offsetWidth: 400, clientHeight, offsetHeight: clientHeight, scrollHeight, scrollTop, children,
        ownerDocument: doc, addEventListener: (k, fn) => feedHeard.set(k, fn), removeEventListener() {},
        getBoundingClientRect: () => ({ top: 0, bottom: clientHeight, left: 0, right: 400, width: 400, height: clientHeight }),
        contains: node => node === feed || Boolean(node?.inFeed),
        querySelectorAll: () => children.filter(c => c.dataset?.historyId),
    };
    const addRow = ({ classes = ['chat-bubble'], historyId = '', y, height }) => {
        const set = new Set(classes);
        const row = {
            inFeed: true, isConnected: true, parentElement: feed, y, height,
            dataset: historyId ? { historyId } : {},
            classList: { contains: c => set.has(c) },
            getBoundingClientRect: () => ({ top: row.y - feed.scrollTop, bottom: row.y - feed.scrollTop + row.height, left: 0, right: 400, width: 400, height: row.height }),
            getClientRects: () => (row.height > 0 ? [row.getBoundingClientRect()] : []),
            closest: sel => (sel === '[data-history-id]' && row.dataset.historyId ? row : null),
            querySelectorAll: () => [],
        };
        children.push(row);
        return row;
    };
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: new Map() });
    return { feed, anchors, addRow };
}

test('wheel series of 2 and 5 events at 16-60 ms spacing keep owedShift when 300 px disappear above the reading line', () => {
    for (const count of [2, 5]) for (const spacingMs of [16, 48, 60]) for (const withScrollEnd of [true, false]) {
        let readingLine;
        runGestureSeries({
            kind: 'wheel', count, spacingMs, withScrollEnd,
            setupFeed: (doc, feedHeard) => {
                const { feed, anchors, addRow } = makeSeriesDom(doc, feedHeard, { scrollTop: 1000, clientHeight: 500, scrollHeight: 3500 });
                const topRow = addRow({ historyId: 'chat:top', y: 1020, height: 120 });
                readingLine = addRow({ historyId: 'chat:read', y: 1390, height: 100 }); // +390 px before scrolling
                readingLine._topRow = topRow;
                return { feed, anchors, targetRow: readingLine };
            },
            mutateAfterFirst: reading => reading.mutate(() => {
                readingLine._topRow.y -= 300;
                readingLine.y -= 300;
                return true;
            }, { remoteContent: true }),
        });
        const expected = 390 - count * 60; // +270 px for 2 events, +90 px for 5 events
        assert.equal(readingLine.getBoundingClientRect().top, expected,
            `wheel count=${count} spacing=${spacingMs}ms scrollend=${withScrollEnd}`);
    }
});

test('wheel series of 2 and 5 events at 16-60 ms spacing keep owedShift across a 400 px card face change', () => {
    for (const count of [2, 5]) for (const spacingMs of [16, 48, 60]) for (const withScrollEnd of [true, false]) {
        let card, nextRow, lowerRow, domFeed;
        runGestureSeries({
            kind: 'wheel', count, spacingMs, withScrollEnd,
            setupFeed: (doc, feedHeard) => {
                const { feed, anchors, addRow } = makeSeriesDom(doc, feedHeard, { scrollTop: 2000, clientHeight: 750, scrollHeight: 5000 });
                domFeed = feed;
                card = addRow({ classes: ['chat-bubble', 'chat-card-bubble'], historyId: 'card:face', y: 1500, height: 650 }); // [-500..+150]
                nextRow = addRow({ historyId: 'chat:next', y: 2154, height: 120 }); // +154 px
                lowerRow = addRow({ historyId: 'chat:lower', y: 2334, height: 120 }); // +334 px
                return { feed, anchors, targetRow: nextRow };
            },
            mutateAfterFirst: reading => reading.mutate(() => {
                card.height = 250; // 400 px face shrink
                nextRow.y -= 400;
                lowerRow.y -= 400;
                domFeed.scrollHeight -= 400;
                return true;
            }, { remoteContent: true }),
        });
        assert.equal(nextRow.getBoundingClientRect().top, 154 - count * 60,
            `nextRow wheel count=${count} spacing=${spacingMs}ms scrollend=${withScrollEnd}`);
        assert.equal(lowerRow.getBoundingClientRect().top, 334 - count * 60,
            `lowerRow wheel count=${count} spacing=${spacingMs}ms scrollend=${withScrollEnd}`);
    }
});

test('touchmove series of 2 and 5 events at 16-60 ms spacing keep owedShift for 300 px removal and 400 px card face change', () => {
    for (const count of [2, 5]) for (const spacingMs of [16, 48, 60]) {
        let readingLine;
        runGestureSeries({
            kind: 'touch', count, spacingMs, withScrollEnd: false,
            setupFeed: (doc, feedHeard) => {
                const { feed, anchors, addRow } = makeSeriesDom(doc, feedHeard, { scrollTop: 1000, clientHeight: 500, scrollHeight: 3500 });
                const topRow = addRow({ historyId: 'chat:top', y: 1020, height: 120 });
                readingLine = addRow({ historyId: 'chat:read', y: 1390, height: 100 });
                readingLine._topRow = topRow;
                return { feed, anchors, targetRow: readingLine };
            },
            mutateAfterFirst: reading => reading.mutate(() => {
                readingLine._topRow.y -= 300;
                readingLine.y -= 300;
                return true;
            }, { remoteContent: true }),
        });
        assert.equal(readingLine.getBoundingClientRect().top, 390 - count * 60,
            `touch 300px removal count=${count} spacing=${spacingMs}ms`);

        let card, nextRow, lowerRow, domFeed;
        runGestureSeries({
            kind: 'touch', count, spacingMs, withScrollEnd: false,
            setupFeed: (doc, feedHeard) => {
                const { feed, anchors, addRow } = makeSeriesDom(doc, feedHeard, { scrollTop: 2000, clientHeight: 750, scrollHeight: 5000 });
                domFeed = feed;
                card = addRow({ classes: ['chat-bubble', 'chat-card-bubble'], historyId: 'card:face', y: 1500, height: 650 });
                nextRow = addRow({ historyId: 'chat:next', y: 2154, height: 120 });
                lowerRow = addRow({ historyId: 'chat:lower', y: 2334, height: 120 });
                return { feed, anchors, targetRow: nextRow };
            },
            mutateAfterFirst: reading => reading.mutate(() => {
                card.height = 250;
                nextRow.y -= 400;
                lowerRow.y -= 400;
                domFeed.scrollHeight -= 400;
                return true;
            }, { remoteContent: true }),
        });
        assert.equal(nextRow.getBoundingClientRect().top, 154 - count * 60,
            `touch 400px card face nextRow count=${count} spacing=${spacingMs}ms`);
        assert.equal(lowerRow.getBoundingClientRect().top, 334 - count * 60,
            `touch 400px card face lowerRow count=${count} spacing=${spacingMs}ms`);
    }
});

test('an opening\'s saved-preview follow yields to a Reply that claimed the room and left the newest message', t => {
    const f = fixture(t);
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 2000, scrollTop: 0, clientHeight: 400 });
    // A Reply landed here first (chat_reply.reveal): its claim, and the newest message left.
    const reply = f.reading.claim(); f.reading.stick = false;
    f.reading.followAfterLayout(true); f.frame(); f.frame();
    assert.deepEqual([f.feed.scrollTop, f.reading.stick, reply()], [0, false, true], 'the opening moves nothing of the landing');
    // With no navigation, or one that keeps following (an unread opening, claim(true)), the opening follows.
    for (const stick of [true, false]) {
        f.feed.scrollTop = 0; f.reading.cancel(); f.reading.stick = stick;
        f.reading.followAfterLayout(true); f.frame(); f.frame();
        assert.equal(f.feed.scrollTop, f.feed.scrollHeight, `no navigation in charge (stick ${stick}): the opening follows`);
    }
    f.feed.scrollTop = 0; f.reading.claim(true);
    f.reading.followAfterLayout(true); f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, f.feed.scrollHeight, 'an unread opening still starts from the newest message');
    // A follow asked for (↓, a sent message) never yields.
    f.feed.scrollTop = 0; f.reading.claim(); f.reading.stick = false;
    f.reading.followAfterLayout(); f.frame(); f.frame();
    assert.equal(f.feed.scrollTop, f.feed.scrollHeight);
});

// A reader placed on one row of a feed whose own layout and parents the test moves: `anchors` counts the searches
// for the row read and records where each restore puts it.
function placedFixture(t) {
    const previous = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = () => 0;  // a width that moves once settles in frames this test does not run
    t.after(() => { globalThis.requestAnimationFrame = previous; });
    const page = { offsetTop: 0, offsetParent: null }, room = { offsetTop: 0, offsetParent: page };
    const feed = { scrollTop: 400, scrollHeight: 3000, clientHeight: 500, clientWidth: 375, offsetWidth: 375, offsetTop: 0,
        offsetParent: room, get offsetHeight() { return this.clientHeight; }, contains: node => node?.parentElement === feed,
        getBoundingClientRect() { return { top: 0, bottom: this.clientHeight, height: this.clientHeight }; } };
    const row = { isConnected: true, parentElement: feed, getBoundingClientRect: () => ({ top: 100, bottom: 160, width: 300, height: 60 }) };
    let searches = 0;
    const restored = [];
    const anchors = { capture: () => { searches++; return { node: row, offset: 100, historyId: 'chat:1' }; },
        restore: anchor => { restored.push(anchor.offset); return true; }, serialize: () => null };
    const reading = createChatReadingPosition({ feed, anchors, visible: () => true, alive: () => true, ready: () => true,
        changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    reading.stick = false; reading.remember(true);
    return { reading, feed, row, page, room, restored, searches: () => searches };
}

test('a row put back stays the row read through a passing layout; a row gone is searched again', t => {
    // A rotation: the width changes and changes back. The row read before it is the one put back after it, not a
    // fresh search of the band, which would land the way back a line off.
    const f = placedFixture(t);
    f.feed.clientWidth = f.feed.offsetWidth = 667; f.reading.reflow();
    f.feed.clientWidth = f.feed.offsetWidth = 375; f.reading.reflow();
    assert.deepEqual([f.searches(), f.restored], [1, [100, 100]], 'the same row, at its own offset, both ways');
    f.row.isConnected = false;
    f.feed.clientWidth = f.feed.offsetWidth = 667; f.reading.reflow();
    assert.equal(f.searches(), 2, 'a row no longer placed yields to a fresh search');
});

test('the feed moved by a bar entering the flow outside its own parent shifts the place; an unmoved feed does not', t => {
    const f = placedFixture(t);
    f.reading.reflow();
    assert.deepEqual(f.restored, [100], 'nothing moved: the row is put back where it was');
    f.page.offsetTop = 56;  // a Project room's bar enters the flow above the room, outside the feed's own offsetParent
    f.reading.reflow();
    assert.deepEqual(f.restored, [100, 44], 'the feed moved down 56 px, so the row goes 56 px up in it and stays on the screen');
});
