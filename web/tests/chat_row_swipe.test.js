// Swipe a message to the left to reply (chat_row_swipe.js, owner 19.10 4A): the row follows the finger leftwards, the
// reply mark fills toward the reply point, crossing it plays one haptic per crossing, releasing past it replies as the
// row's Reply does, and the row settles back either way. The moves are cancelled once the axis is decided, by the
// gesture's own listener and by the feed's, registered up front and only in the stack layout under a coarse pointer.
// Real touch, layout and paint are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { SWIPE, createRowSwipe, pull } from '../modules/chat_row_swipe.js';
import { installDomDouble } from './dom_double.js';

function fixture({ stack = true, coarse = true, blocked = false, selection = '' } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    doc.body.append(feed);
    feed.rect = { left: 0, top: 60, right: 390, bottom: 700, width: 390, height: 640 };
    const words = el('div', { className: 'message', text: 'words' });
    const code = el('pre', { text: 'wide code' });
    const field = el('textarea');
    const bubble = el('div', { className: 'chat-bubble assistant', dataset: { rowAddress: 'row:1@t#a' } }, words, code, field);
    bubble.rect = { left: 20, top: 200, right: 370, bottom: 280, width: 350, height: 80 };
    const echo = el('div', { className: 'chat-bubble user' }, el('div', { className: 'message', text: 'unsaved echo' }));
    feed.append(bubble, echo);
    const replied = [];
    const haptics = [];
    const layoutHandlers = [];
    const state = { stack, coarse, blocked, selection };
    globalThis.window.Telegram = { WebApp: { initData: 'x', isVersionAtLeast: () => true,
        HapticFeedback: { impactOccurred: (style) => haptics.push(style) } } };
    globalThis.window.getSelection = () => ({ toString: () => state.selection });
    const dispose = createRowSwipe({
        feed, icon: '<svg></svg>', reply: (ref) => replied.push(ref.address),
        replyRef: (row) => (row.dataset.rowAddress ? { address: row.dataset.rowAddress } : null),
        isStack: () => state.stack, isCoarse: () => state.coarse, isBlocked: () => state.blocked,
        onLayout: (sync) => { layoutHandlers.push(sync); return () => layoutHandlers.splice(layoutHandlers.indexOf(sync), 1); },
        win: globalThis.window, doc,
    });
    const touches = (x, y) => [{ identifier: 1, clientX: x, clientY: y }];
    const start = (target, x, y) => target.emit('touchstart', { touches: touches(x, y), changedTouches: touches(x, y) });
    const move = (target, x, y, extra = {}) => target.emit('touchmove', { touches: touches(x, y), changedTouches: touches(x, y), cancelable: true, ...extra });
    const lift = (target, x, y) => target.emit('touchend', { touches: [], changedTouches: touches(x, y) });
    const cancel = (target, x, y) => target.emit('touchcancel', { touches: [], changedTouches: touches(x, y) });
    const mark = () => doc.body.children.find((node) => node.classList.contains('chat-row-reply-mark')) || null;
    return { ...env, feed, bubble, echo, words, code, field, replied, haptics, state, layoutHandlers, dispose, start, move, lift, cancel, mark,
        close() { dispose(); delete globalThis.window.Telegram; env.restore(); } };
}

// Moves a finger from x=300 to the left by `steps`, after the axis locked at the first move past the slop.
function swipe(f, target, steps, { y = 220 } = {}) {
    f.start(target, 300, y);
    const moves = [];
    for (const dx of steps) moves.push(f.move(target, 300 - dx, y + dx * 0.1));
    return moves;
}

test('the row follows a leftward finger 1:1 to the reply point, then gives like a rubber band', () => {
    assert.equal(pull(0), 0);
    assert.equal(pull(30), 30);
    assert.equal(pull(SWIPE.replyAt), SWIPE.replyAt);
    const give = [SWIPE.replyAt + 20, SWIPE.replyAt + 60, SWIPE.replyAt + 400].map(pull);
    assert.ok(give[0] > SWIPE.replyAt && give[1] > give[0] && give[2] > give[1], 'it keeps giving with every pixel');
    assert.ok(give[2] < SWIPE.replyAt + SWIPE.giveMax, 'and never reaches the limit');
});

test('a swipe past the reply point replies as the row\'s Reply does; the mark fills and the row settles back', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        const moves = swipe(f, f.words, [12, 30, 60]);
        assert.equal(moves[0].defaultPrevented, true, 'the lock move is cancelled: iOS cannot hand the finger to the scroll');
        assert.equal(f.bubble.dataset.rowSwipe, 'drag');
        assert.equal(f.bubble.css.get('--row-swipe'), '48px', 'from the lock point, 1:1');
        assert.equal(f.mark().css.get('--reply-k'), String(48 / SWIPE.replyAt));
        assert.equal(f.mark().dataset.armed, undefined);
        assert.deepEqual(f.haptics, []);
        const past = f.move(f.words, 300 - 80, 222);
        assert.equal(past.defaultPrevented, true);
        assert.equal(f.mark().dataset.armed, 'true');
        assert.deepEqual(f.haptics, ['medium'], 'one haptic as the reply point is crossed');
        f.move(f.words, 300 - 90, 222);
        assert.deepEqual(f.haptics, ['medium'], 'not again while it stays past it');
        f.lift(f.words, 300 - 90, 222);
        assert.deepEqual(f.replied, ['row:1@t#a']);
        assert.equal(f.bubble.dataset.rowSwipe, 'settle');
        assert.equal(f.bubble.css.get('--row-swipe'), '0px');
        assert.equal(f.mark().dataset.leaving, 'true');
        t.mock.timers.tick(SWIPE.settle + 40);
        assert.equal(f.bubble.dataset.rowSwipe, undefined, 'at rest again');
        assert.equal(f.bubble.css.has('--row-swipe'), false);
        assert.equal(f.mark(), null, 'the mark has gone');
    } finally { f.close(); }
});

test('pulling back under the point disarms; crossing again plays again; a release short of it replies nothing', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        swipe(f, f.words, [12, 80]);
        assert.deepEqual(f.haptics, ['medium']);
        f.move(f.words, 300 - 40, 220);
        assert.equal(f.mark().dataset.armed, 'false');
        f.move(f.words, 300 - 90, 220);
        assert.deepEqual(f.haptics, ['medium', 'medium'], 'a second crossing is felt again');
        f.move(f.words, 300 - 30, 220);
        f.lift(f.words, 300 - 30, 220);
        assert.deepEqual(f.replied, [], 'released short of the point');
        assert.equal(f.bubble.dataset.rowSwipe, 'settle');
    } finally { f.close(); }
});

test('the system taking the touch settles the row back and replies nothing', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        swipe(f, f.words, [12, 90]);
        f.cancel(f.words, 300 - 90, 220);
        assert.deepEqual(f.replied, []);
        assert.equal(f.bubble.dataset.rowSwipe, 'settle');
    } finally { f.close(); }
});

test('a rightward finger stays the drawer\'s and a vertical one scrolls: neither moves the row', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.start(f.words, 100, 220);
        const right = f.move(f.words, 130, 222);
        assert.equal(right.defaultPrevented, undefined);
        f.move(f.words, 40, 222);
        assert.equal(f.bubble.dataset.rowSwipe, undefined, 'once rightwards, the gesture is not a reply');
        f.lift(f.words, 40, 222);
        f.start(f.words, 300, 220);
        const down = f.move(f.words, 296, 260);
        assert.equal(down.defaultPrevented, undefined, 'the feed scrolls');
        f.move(f.words, 200, 262);
        assert.equal(f.bubble.dataset.rowSwipe, undefined);
        assert.equal(f.mark(), null);
    } finally { f.close(); }
});

test('a field, something that scrolls sideways, selected words, an open menu and a row without Reply keep the finger', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        for (const target of [f.field, f.code, f.echo]) {
            swipe(f, target, [12, 90]);
            f.lift(target, 210, 220);
        }
        f.state.selection = 'some words';
        swipe(f, f.words, [12, 90]);
        f.lift(f.words, 210, 220);
        f.state.selection = '';
        f.state.blocked = true;
        swipe(f, f.words, [12, 90]);
        f.lift(f.words, 210, 220);
        assert.deepEqual(f.replied, []);
        assert.equal(f.bubble.dataset.rowSwipe, undefined);
        // A menu a long press opened under the still finger is caught at the lock too.
        f.state.blocked = false;
        f.start(f.words, 300, 220);
        f.state.blocked = true;
        const lock = f.move(f.words, 280, 221);
        assert.equal(lock.defaultPrevented, undefined);
        f.move(f.words, 200, 221);
        f.lift(f.words, 200, 221);
        assert.deepEqual(f.replied, []);
    } finally { f.close(); }
});

test('a lock move the browser no longer lets the page cancel is the browser\'s scroll, never a swipe', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.start(f.words, 300, 220);
        f.move(f.words, 280, 221, { cancelable: false });
        f.move(f.words, 200, 221);
        f.lift(f.words, 200, 221);
        assert.deepEqual(f.replied, []);
        assert.equal(f.bubble.dataset.rowSwipe, undefined);
    } finally { f.close(); }
});

test('the feed\'s cancelling listener exists only in the stack layout under a coarse pointer and follows the layout', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture({ stack: false });
    try {
        assert.equal(f.feed.listeners.get('touchmove')?.size || 0, 0, 'docked: a touch on the feed never waits for the page');
        swipe(f, f.words, [12, 90]);
        f.lift(f.words, 210, 220);
        assert.deepEqual(f.replied, [], 'and no swipe');
        f.state.stack = true;
        f.layoutHandlers.forEach((sync) => sync());
        assert.equal(f.feed.listeners.get('touchmove')?.size, 1, 'mounted when the layout switches, before any touch');
        // A move bubbling from a row the gesture locked is cancelled by the feed as well.
        f.start(f.words, 300, 220);
        f.move(f.words, 280, 221);
        const bubbled = f.feed.emit('touchmove', { touches: [{ identifier: 1, clientX: 250, clientY: 221 }], cancelable: true });
        assert.equal(bubbled.defaultPrevented, true);
        f.lift(f.words, 250, 221);
    } finally { f.close(); }
});

test('the click that ends a swipe is not a tap; dispose releases every listener and the mark', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        swipe(f, f.words, [12, 90]);
        f.lift(f.words, 210, 220);
        const click = f.words.emit('click', { target: f.words });
        assert.equal(click.defaultPrevented, true);
        assert.equal(f.words.emit('click', { target: f.words }).defaultPrevented, undefined, 'only that one click');
        swipe(f, f.words, [12, 40]);
        f.dispose();
        assert.equal(f.words.listenerCount(), 0, 'the gesture\'s own listeners are gone');
        assert.equal(f.feed.listenerCount(), 0);
        assert.equal(f.doc.captures.get('click')?.size || 0, 0);
        assert.equal(f.mark(), null);
        assert.equal(f.bubble.dataset.rowSwipe, undefined);
    } finally { f.restore(); delete globalThis.window?.Telegram; }
});

test('the element the touch started on keeps the finger after a re-render removes it from the feed', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.start(f.words, 300, 220);
        f.bubble.remove();  // a live update redraws the row under the still finger
        const lock = f.move(f.words, 288, 221);
        assert.equal(lock.defaultPrevented, true, 'its own listener cancels: the move no longer reaches the feed');
        const drag = f.move(f.words, 200, 230);
        assert.equal(drag.defaultPrevented, true);
        f.lift(f.words, 200, 230);
        assert.deepEqual(f.replied, ['row:1@t#a'], 'the reply still names the row the finger took');
    } finally { f.close(); }
});

test('a layout that leaves the stack mid-swipe ends it without a reply', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        swipe(f, f.words, [12, 90]);
        assert.equal(f.mark().dataset.armed, 'true');
        f.state.stack = false;  // a phone turned into a docked window, the finger still down
        f.layoutHandlers.forEach((sync) => sync());
        assert.equal(f.bubble.dataset.rowSwipe, 'settle', 'the row goes back at once');
        assert.equal(f.mark().dataset.leaving, 'true');
        f.lift(f.words, 210, 220);
        assert.deepEqual(f.replied, [], 'the docked room was not asked to reply');
        assert.equal(f.words.listenerCount(), 0);
    } finally { f.close(); }
});
