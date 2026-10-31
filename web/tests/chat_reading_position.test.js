import assert from 'node:assert/strict';
import test from 'node:test';
import { createReviewHydrator } from '../modules/review_presentation.js';
import { createChatReadingPosition } from '../modules/chat_reading_position.js';

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
        contains: node => node === feed || Boolean(node?.inFeed) };
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
    const body = { inFeed: true, parentElement: card, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400 };
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
    for (const top of [400, 580]) { f.feed.scrollTop = top; f.reading.scroll(); f.frame(); }
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
    const body = { inFeed: true, parentElement: f.feed, overflowY: 'auto', scrollTop: 200, scrollHeight: 1000, clientHeight: 400 };
    const text = { inFeed: true, parentElement: body };
    f.reading.bindGestures(() => {});
    f.reading.cancel();
    Object.assign(f.feed, { scrollHeight: 1000, scrollTop: 100, clientHeight: 400 });
    const current = f.reading.claim(); // ↓ pressed; its latest page is still loading
    assert.equal(current(), true);
    f.gesture({ type: 'wheel', deltaY: 1, target: text, timeStamp: 1000 }); f.frame();
    assert.equal(current(), false, 'reading on inside a box keeps the reader where they are');
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
