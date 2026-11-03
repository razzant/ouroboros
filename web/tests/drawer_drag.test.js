import test from 'node:test';
import assert from 'node:assert/strict';
import { BLOCKING_SELECTOR, DRAG, DRAWER_DRAG, FIELD_SELECTOR, SIDEWAYS_SELECTOR, bindDrawerDrag, settleMs, stretch } from '../modules/drawer_drag.js';

// The drawer's finger (owner Q1=A, 19.10): eligibility, the axis lock, the cancelled page scroll, tracking by one
// number on the element the touch started on, the commit rules, the settle at the finger's speed, the restores
// that leave no trace, the scrim that fades before it goes, the swallowed ending click, and the one-line switch.

function listenerMap() {
    const map = new Map();
    return {
        map,
        add: (type, fn, options) => { map.set(type, [...(map.get(type) || []), { fn, options }]); },
        remove: (type, fn) => { map.set(type, (map.get(type) || []).filter((x) => x.fn !== fn)); },
        fire: (type, event) => (map.get(type) || []).slice().forEach(({ fn }) => fn(event)),
        count: () => [...map.values()].reduce((n, fns) => n + fns.length, 0),
    };
}

function fakeEl(name) {
    const classes = new Set();
    const props = new Map();
    const own = listenerMap();
    return {
        name, hidden: true, parentElement: null, scrollWidth: 0, clientWidth: 0, style: null, attached: true, own,
        classList: { toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)), contains: (c) => classes.has(c), add: (c) => classes.add(c) },
        styleProps: props,
        addEventListener: own.add, removeEventListener: own.remove,
        contains(other) { for (let x = other; x; x = x.parentElement) if (x === this) return true; return false; },
        closest() { return null; },
        getBoundingClientRect: () => ({ width: 320 }),
    };
}
function withStyle(el) { el.style = { setProperty: (k, v) => el.styleProps.set(k, v), removeProperty: (k) => el.styleProps.delete(k) }; return el; }

function world({ open = false, stack = true, coarse = true, enabled = DRAWER_DRAG, blockers = null, also = null } = {}) {
    const winL = listenerMap();
    const docL = listenerMap();
    const rootL = listenerMap();
    const styles = new Map();
    const clock = { t: 1000 };
    const timers = new Map();
    let timerId = 0;
    const doc = { body: fakeEl('body'), addEventListener: docL.add, removeEventListener: docL.remove, listeners: docL,
        documentElement: { addEventListener: rootL.add, removeEventListener: rootL.remove, listeners: rootL },
        querySelectorAll: () => blockers || [] };
    const win = { addEventListener: winL.add, removeEventListener: winL.remove, listeners: winL, innerWidth: 390,
        getComputedStyle: (el) => styles.get(el) || { touchAction: 'auto', overflowX: 'visible' },
        performance: { now: () => clock.t },
        setTimeout: (fn, ms) => { timerId += 1; timers.set(timerId, { fn, at: clock.t + ms }); return timerId; },
        clearTimeout: (id) => { timers.delete(id); } };
    const sidebar = withStyle(fakeEl('sidebar'));
    const scrim = withStyle(fakeEl('scrim'));
    const room = fakeEl('room'); room.parentElement = doc.body;
    const state = { open, set: [], grabs: 0, snaps: [], order: [] };
    const flags = { field: new Set(), sideways: new Set(), blocked: false, selection: false };
    const env = { stack, coarse, layoutHandlers: new Set() };
    const dispose = bindDrawerDrag({
        sidebar, scrim, isOpen: () => state.open,
        setOpen: (v) => { state.set.push(v); state.order.push(`set:${v}`); state.open = v; scrim.hidden = !v; },
        isStack: () => env.stack, isCoarse: () => env.coarse, onGrab: () => { state.grabs += 1; }, onSnap: (v) => { state.snaps.push(v); state.order.push(`snap:${v}`); }, win, doc, enabled,
        onLayout: (fn) => { env.layoutHandlers.add(fn); return () => env.layoutHandlers.delete(fn); },
        isField: (el) => flags.field.has(el), isSideways: (el) => flags.sideways.has(el),
        ...(blockers ? {} : { isBlocked: () => flags.blocked }), hasSelection: () => flags.selection,
        ...(also ? { also } : {}),
    });
    // A touch event is dispatched at the element the touch started on; it bubbles to the root element and the
    // document only while that element is still in the page.
    const down = new Map();
    const dispatch = (type, target, event) => {
        target.own.fire(type, event);
        if (target.attached) { rootL.fire(type, event); docL.fire(type, event); }
    };
    const touchEvent = (type, { x, y, id = 1, target, dt = 16, cancelable = true }) => {
        clock.t += dt;
        const point = { identifier: id, clientX: x, clientY: y };
        if (type === 'touchstart') down.set(id, { point, target });
        else if (down.has(id)) down.get(id).point = point;
        const touches = [...down.values()].map((d) => d.point);
        if (type === 'touchend' || type === 'touchcancel') down.delete(id);
        const event = { type, target, touches: type === 'touchend' || type === 'touchcancel' ? [...down.values()].map((d) => d.point) : touches,
            changedTouches: [point], timeStamp: clock.t, cancelable, prevented: false,
            preventDefault() { this.prevented = true; } };
        dispatch(type, target, event);
        return event;
    };
    const touch = (type, opts) => touchEvent(type, { ...opts, target: opts.target || down.get(opts.id || 1)?.target || room });
    const click = () => { const e = { stopped: false, prevented: false, stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; } }; docL.fire('click', e); return e; };
    const progress = () => Number(sidebar.styleProps.get('--drawer-p'));
    const dragging = () => sidebar.classList.contains('dragging') && scrim.classList.contains('dragging');
    const runTimers = (ms = 10_000) => {
        clock.t += ms;
        for (const [id, timer] of [...timers]) if (timer.at <= clock.t) { timers.delete(id); timer.fn(); }
    };
    return { doc, root: doc.documentElement, win, env, sidebar, scrim, room, state, flags, styles, dispose, touch, click, progress, dragging, clock, runTimers, timers };
}

// A drag from `from`: the first move locks the axis one slop in (12px), then `steps` moves reach `to`.
// Progress is measured from the lock point, so p = (to - from - 12) / 320.
const open = (w, { from = 100, to = 300, steps = 4, dt = 16, up = true, target } = {}) => {
    const moves = [];
    w.touch('touchstart', { x: from, y: 300, target });
    const lock = from + 12;
    moves.push(w.touch('touchmove', { x: lock, y: 300, dt }));
    for (let i = 1; i <= steps; i += 1) moves.push(w.touch('touchmove', { x: lock + ((to - lock) * i) / steps, y: 300, dt }));
    if (up) w.touch('touchend', { x: to, y: 300, dt });
    return moves;
};

test('the switch: DRAWER_DRAG off binds nothing', () => {
    assert.equal(typeof DRAWER_DRAG, 'boolean');
    const w = world({ enabled: false });
    assert.equal(w.doc.listeners.count() + w.root.listeners.count(), 0);
    w.dispose();
});

test('a start is eligible anywhere on the room, the edge included, but a field, with nothing over it, from a touch', () => {
    const w = world();
    const tryOpen = (opts = {}) => { open(w, opts); w.runTimers(); const set = w.state.set.slice(); w.state.set.length = 0; return set; };
    assert.deepEqual(tryOpen(), [true], 'the plain case opens');
    w.state.open = false; w.scrim.hidden = true;
    assert.deepEqual(tryOpen({ from: 2, to: 300 }), [true], 'from the screen\'s very edge too (owner 19.10)');
    w.state.open = false; w.scrim.hidden = true;
    w.flags.field.add(w.room);
    assert.deepEqual(tryOpen(), [], 'a field keeps its touch');
    w.flags.field.clear(); w.flags.sideways.add(w.room);
    assert.deepEqual(tryOpen(), [], 'a sideways scroller keeps its touch');
    w.flags.sideways.clear(); w.flags.selection = true;
    assert.deepEqual(tryOpen(), [], 'selected text keeps its touch');
    w.flags.selection = false; w.flags.blocked = true;
    assert.deepEqual(tryOpen(), [], 'a dialog, menu, reader or Shade owns the finger');
    w.flags.blocked = false;
    w.styles.set(w.room, { touchAction: 'none', overflowX: 'visible' });
    assert.deepEqual(tryOpen(), [], 'touch-action: none (the effort strip) keeps its touch');
    w.styles.set(w.room, { touchAction: 'auto', overflowX: 'auto' }); w.room.scrollWidth = 900; w.room.clientWidth = 300;
    assert.deepEqual(tryOpen(), [], 'a horizontal scroller keeps its touch');
    w.styles.delete(w.room); w.room.scrollWidth = 0;
    w.dispose();
    const docked = world({ stack: false });
    open(docked); assert.deepEqual(docked.state.set, [], 'docked: no drawer to drag'); docked.dispose();
    const fine = world({ coarse: false });
    open(fine); assert.deepEqual(fine.state.set, [], 'a fine pointer: no drag'); fine.dispose();
    // The real blocker test reads what is shown: a closed menu that stays mounted (Files' context menu, a model
    // list) does not block; an open one does.
    const mounted = world({ blockers: [{ getClientRects: () => [] }] });
    open(mounted); assert.deepEqual(mounted.state.set, [true], 'a mounted, hidden menu does not block'); mounted.dispose();
    const shown = world({ blockers: [{ getClientRects: () => [] }, { getClientRects: () => [{}] }] });
    open(shown); assert.deepEqual(shown.state.set, [], 'an open menu owns the finger'); shown.dispose();
    assert.match(FIELD_SELECTOR, /input, textarea/); assert.match(SIDEWAYS_SELECTOR, /\[data-scroll-x\]/); assert.match(BLOCKING_SELECTOR, /dialog\[open\]/);
    // A closed menu's leaving picture and a folding Shade no longer own the finger; their live forms do.
    for (const live of ['.ui-popup', '.chat-photo-menu']) assert.ok(BLOCKING_SELECTOR.includes(`${live}:not([data-leaving])`), live);
    assert.ok(BLOCKING_SELECTOR.includes('.rw-shade:not([hidden]):not([inert])'));
    // A button, a link or a summary under the finger does not stop a drag: the click that ends it is swallowed.
    for (const control of ['button', 'a[href]', 'summary', 'label', 'details', '[role="button"]']) {
        assert.equal(FIELD_SELECTOR.split(', ').includes(control), false, control);
    }
});

test('the axis locks after the slop; from then on no move may scroll the page, so the finger stays the drawer\'s', () => {
    const w = world();
    w.touch('touchstart', { x: 100, y: 300 });
    const inSlop = w.touch('touchmove', { x: 104, y: 306 });
    assert.equal(w.dragging(), false, 'inside the slop nothing is decided');
    assert.equal(inSlop.prevented, false, 'and the page may still scroll');
    const vertical = w.touch('touchmove', { x: 110, y: 330 });
    w.touch('touchmove', { x: 300, y: 330 });
    assert.equal(w.dragging(), false, 'a vertical start is a scroll, never a drag');
    assert.equal(vertical.prevented, false, 'a scroll is never cancelled');
    w.touch('touchend', { x: 300, y: 330 });
    assert.deepEqual(w.state.set, []);
    assert.equal(w.room.own.count(), 0, 'a gesture that ended leaves no listener on its element');
    // Sideways past the slop, with |dx| > |dy|: the drawer is revealed at 0 and follows; every move is cancelled.
    w.touch('touchstart', { x: 100, y: 300 });
    const lock = w.touch('touchmove', { x: 112, y: 304 });
    assert.equal(w.dragging(), true);
    assert.equal(lock.prevented, true, 'the move that locks the axis is cancelled');
    assert.equal(w.scrim.hidden, false, 'the scrim is painted during the drag');
    assert.equal(w.progress(), 0);
    const drift = w.touch('touchmove', { x: 112 + 160, y: 304 + 80 });
    assert.equal(drift.prevented, true, 'a thumb drifting down mid-drag cannot start a scroll (iOS would cancel the finger)');
    assert.equal(w.progress(), 0.5, 'progress is travel over the drawer width from the lock point');
    assert.equal(w.scrim.styleProps.get('--drawer-p'), '0.5');
    w.touch('touchmove', { x: 112 + 900, y: 304 });
    assert.ok(w.progress() > 1 && w.progress() < 1.25, 'past fully open it gives, damped (a rubber band, not a wall)');
    w.touch('touchend', { x: 112 + 900, y: 304 });
    assert.deepEqual(w.state.set, [true]);
    assert.equal(w.dragging(), false);
    assert.equal(w.sidebar.styleProps.has('--drawer-p'), false, 'the number is the finger\'s; the state owns the rest');
    w.state.open = false; w.scrim.hidden = true; w.state.set.length = 0; w.runTimers();
    // A thumb's arc: 12px sideways and 10 up at the slop is a drag, not a scroll.
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 112, y: 290 });
    assert.equal(w.dragging(), true, 'a diagonal thumb is the drawer\'s');
    w.touch('touchmove', { x: 300, y: 230 });
    w.touch('touchend', { x: 300, y: 230 });
    assert.deepEqual(w.state.set, [true]);
    w.state.open = false; w.scrim.hidden = true; w.state.set.length = 0; w.runTimers();
    // The browser already scrolls (its move is not cancelable): the finger is the scroll's, never half a drag.
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 114, y: 302, cancelable: false });
    assert.equal(w.dragging(), false);
    w.touch('touchmove', { x: 300, y: 302 });
    assert.equal(w.dragging(), false);
    w.touch('touchend', { x: 300, y: 302 });
    assert.deepEqual(w.state.set, []);
    w.dispose();
});

test('the page listener cancels a move only while a drawer is dragged, and it is in place before any touch', () => {
    const w = world();
    const page = w.root.listeners.map.get('touchmove') || [];
    assert.equal(page.length, 1);
    assert.equal(page[0].options?.passive, false, 'a non-passive page listener, registered up front (iOS decides at the touch\'s start)');
    assert.equal(w.doc.listeners.map.get('touchmove')?.length || 0, 0, 'on the root element, not the document (iOS makes those passive)');
    const start = w.root.listeners.map.get('touchstart') || [];
    assert.equal(start[0].options?.passive, true, 'a start never blocks scrolling');
    const move = () => ({ cancelable: true, prevented: false, preventDefault() { this.prevented = true; } });
    const idle = move();
    w.root.listeners.fire('touchmove', idle);
    assert.equal(idle.prevented, false, 'no drag, no cancel: scrolling is untouched');
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 112, y: 300 });
    const during = move();
    w.root.listeners.fire('touchmove', during);
    assert.equal(during.prevented, true, 'while the drawer is dragged the page listener itself cancels a move');
    w.dispose();
});

test('the element the touch started on carries it through a re-render that removes it from the page', () => {
    const w = world();
    const row = fakeEl('row'); row.parentElement = w.doc.body;
    w.touch('touchstart', { x: 100, y: 300, target: row });
    assert.equal(row.own.map.get('touchmove')?.[0]?.options?.passive, false, 'its own move listener can cancel too');
    w.touch('touchmove', { x: 112, y: 300 });
    assert.equal(w.dragging(), true);
    assert.equal(w.state.grabs, 1, 'the drag tells the scrim\'s owner it took the scrim');
    row.attached = false;  // a live card re-rendered: its moves reach only the old element, never the document
    const moved = w.touch('touchmove', { x: 112 + 160, y: 300 });
    assert.equal(w.progress(), 0.5, 'the drawer still follows');
    assert.equal(moved.prevented, true);
    w.touch('touchend', { x: 112 + 200, y: 300 });
    assert.deepEqual(w.state.set, [true], 'and the release commits');
    assert.equal(row.own.count(), 0, 'its listeners go with the gesture');
    w.dispose();
});

test('a slow release commits past half (open) or 30% (close); a fast one commits after 24px; else it springs back', () => {
    let w = world();
    open(w, { from: 100, to: 100 + 12 + 150, steps: 5, dt: 200 });  // slow: 30px per 200ms, p = 0.47
    assert.deepEqual(w.state.set, [], 'under half, slow: springs back');
    assert.equal(w.scrim.hidden, false, 'the scrim fades back out with the drawer…');
    w.runTimers();
    assert.equal(w.scrim.hidden, true, '…and is put away once it has');
    open(w, { from: 100, to: 100 + 12 + 170, steps: 5, dt: 200 });  // slow, p = 0.53
    assert.deepEqual(w.state.set, [true]);
    w.dispose();
    w = world();
    open(w, { from: 100, to: 100 + 12 + 40, steps: 2, dt: 16 });  // fast: 20px per 16ms, 52px of travel
    assert.deepEqual(w.state.set, [true], 'a fling after 24px opens');
    w.dispose();
    w = world();
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 115, y: 300, dt: 1 });
    w.touch('touchmove', { x: 122, y: 300, dt: 1 });
    w.touch('touchend', { x: 122, y: 300, dt: 1 });
    assert.deepEqual(w.state.set, [], 'a fast flick under 24px of travel is not a commit');
    w.dispose();
    // Closing: a drag on the open drawer, leftwards.
    w = world({ open: true });
    w.scrim.hidden = false;
    const close = (to, dt) => {
        w.touch('touchstart', { x: 300, y: 300, target: w.sidebar });
        w.touch('touchmove', { x: 288, y: 300, dt });
        w.touch('touchmove', { x: to, y: 300, dt });
        w.touch('touchend', { x: to, y: 300, dt });
    };
    close(288 - 60, 200);
    assert.deepEqual(w.state.set, [], 'less than 30% gone, slow: stays open');
    close(288 - 110, 200);
    assert.deepEqual(w.state.set, [false], 'more than 30% gone: closes');
    w.state.open = true; w.state.set.length = 0;
    w.touch('touchstart', { x: 300, y: 300, target: w.scrim });
    w.touch('touchmove', { x: 320, y: 300 });
    assert.equal(w.dragging(), false, 'the open drawer dragged rightwards has nowhere to go');
    w.touch('touchend', { x: 320, y: 300 });
    w.touch('touchstart', { x: 300, y: 300, target: w.room });
    w.touch('touchmove', { x: 200, y: 300 });
    assert.equal(w.dragging(), false, 'a start outside the open drawer and its scrim is not a close');
    w.touch('touchend', { x: 200, y: 300 });
    assert.deepEqual(w.state.set, []);
    w.dispose();
});

test('a thumb that rests before it lifts is no fling; the same drag lifted while moving is', () => {
    let w = world();
    open(w, { from: 100, to: 100 + 12 + 80, steps: 4, dt: 16, up: false });  // fast, p = 0.25
    w.touch('touchend', { x: 192, y: 300, dt: 150 });  // held still for 150 ms first
    assert.deepEqual(w.state.set, [], 'under half and at rest: it springs back');
    w.dispose();
    w = world();
    open(w, { from: 100, to: 100 + 12 + 80, steps: 4, dt: 16 });  // lifted 16 ms after its last move
    assert.deepEqual(w.state.set, [true], 'lifted in motion it is a fling: it opens');
    w.dispose();
});

test('the release settles at the finger\'s speed: a fling is quick, a slow release far from the end takes longer', () => {
    assert.equal(settleMs(0, 0), DRAG.settleMin, 'nothing left to travel: the shortest settle');
    assert.equal(settleMs(300, 0), DRAG.settleMax, 'a slow release far from the end: the longest');
    assert.ok(settleMs(160, 3) < settleMs(160, 0.2), 'a faster finger settles sooner');
    assert.equal(settleMs(160, -3), settleMs(160, 3), 'either direction');
    const w = world();
    open(w, { from: 100, to: 100 + 12 + 40, steps: 2, dt: 16 });  // a fling: 20px per 16ms
    const fling = w.sidebar.styleProps.get('--drawer-settle');
    assert.match(fling, /^\d+ms$/);
    assert.equal(w.scrim.styleProps.get('--drawer-settle'), fling, 'the scrim settles with the drawer');
    assert.equal(Number.parseInt(fling, 10), settleMs((1 - 40 / 320) * 320, 1.25));
    w.runTimers();
    assert.equal(w.sidebar.styleProps.has('--drawer-settle'), false, 'afterwards ☰ and × keep style.css\'s own timing');
    w.dispose();
});

test('a second finger or a new width restore the start with no commit; a height change keeps the drag', () => {
    const w = world();
    const start = () => { w.touch('touchstart', { x: 100, y: 300 }); w.touch('touchmove', { x: 112, y: 300 }); w.touch('touchmove', { x: 260, y: 300 }); assert.equal(w.dragging(), true); };
    start(); w.touch('touchstart', { x: 200, y: 400, id: 2, target: w.room });
    assert.equal(w.dragging(), false, 'a second finger');
    w.touch('touchend', { x: 200, y: 400, id: 2 }); w.touch('touchend', { x: 260, y: 300 });
    w.runTimers(); assert.equal(w.scrim.hidden, true);
    const resize = () => w.win.listeners.fire('resize', {});
    start(); resize();
    assert.equal(w.dragging(), true, 'a height-only change (a browser bar, the keyboard) keeps the drag');
    w.win.innerWidth = 844; resize();
    assert.equal(w.dragging(), false, 'a new width (the phone turned) ends it');
    assert.deepEqual(w.state.set, [], 'no state change, so no history entry');
    w.touch('touchend', { x: 260, y: 300 });
    assert.deepEqual(w.state.set, []);
    w.dispose();
});

test('the click that ends a drag is swallowed once; a tap is not; dispose releases every listener', () => {
    const w = world();
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchend', { x: 100, y: 300 });
    assert.equal(w.click().stopped, false, 'a tap keeps its click');
    open(w, { from: 100, to: 400 });
    const ending = w.click();
    assert.equal(ending.stopped && ending.prevented, true);
    assert.equal(w.click().stopped, false, 'only the one ending click');
    w.clock.t += DRAG.clickGuard + 1;
    w.touch('touchstart', { x: 100, y: 300 });  // a gesture in flight when the app goes
    w.dispose();
    assert.equal(w.doc.listeners.count() + w.root.listeners.count(), 0, 'dispose releases every page listener');
    assert.equal(w.win.listeners.count(), 0);
    assert.equal(w.room.own.count(), 0, 'and the gesture\'s own');
});

test('when the system takes the touch the drawer settles from where the finger left it, never stranded halfway', () => {
    const w = world();
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 112, y: 300 });
    w.touch('touchmove', { x: 112 + 200, y: 300, dt: 200 });  // p = 0.625, slow
    w.touch('touchcancel', { x: 112 + 200, y: 300 });
    assert.equal(w.dragging(), false);
    assert.deepEqual(w.state.set, [true], 'past half: it opens');
    w.state.open = false; w.scrim.hidden = true; w.state.set.length = 0; w.runTimers();
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 112, y: 300 });
    w.touch('touchmove', { x: 112 + 90, y: 300, dt: 200 });  // p = 0.28, slow
    w.touch('touchcancel', { x: 112 + 90, y: 300 });
    assert.deepEqual(w.state.set, [], 'under half: it goes back');
    assert.equal(w.scrim.hidden, false, 'the scrim fades with it…');
    w.runTimers();
    assert.equal(w.scrim.hidden, true, '…and goes');
    assert.equal(w.room.own.count(), 0);
    w.dispose();
});

test('the rubber band: nothing changes up to fully open; past it the give grows ever slower', () => {
    assert.equal(stretch(0.5, 320), 0.5);
    assert.equal(stretch(-0.3, 320), 0);
    assert.equal(stretch(1, 320), 1);
    const a = stretch(1.1, 320) - 1;
    const b = stretch(1.5, 320) - 1;
    const c = stretch(3, 320) - 1;
    assert.ok(a > 0 && b > a && c > b, [a, b, c]);
    assert.ok((c - b) / 1.5 < (b - a) / 0.4, 'each further pixel of pull gives less');
    assert.ok(c < (3 - 1) / 8, 'a damped give: at most an eighth of the pull past open');
    for (const width of [320, 360]) {
        assert.ok((stretch(10, width) - 1) * width < DRAG.rubberMax, 'never past the painted cover, however far the pull');
        // Every pixel of pull still moves the drawer: out to a far pull and back, never a flat stretch (a wall).
        const gives = [100, 250, 400, 550, 700].map((d) => (stretch(1 + d / width, width) - 1) * width);
        assert.ok(gives.every((g, i) => i === 0 || g > gives[i - 1] + 0.5), gives);
    }
    assert.ok(DRAG.rubberMax < 64, 'style.css paints the drawer\'s surface 64 px past its edge');
});

test('the blocking page listener exists only where a drag can start, and follows the layout before any touch', () => {
    const docked = world({ stack: false });
    const moves = () => docked.root.listeners.map.get('touchmove')?.length || 0;
    const switchTo = (stack) => { docked.env.stack = stack; docked.env.layoutHandlers.forEach((fn) => fn(stack ? 'stack' : 'docked')); };
    assert.equal(moves(), 0, 'a docked window never waits for the page');
    switchTo(true);
    assert.equal(moves(), 1, 'the layout owner\'s switch to the phone layout mounts it, before any touch');
    switchTo(false);
    assert.equal(moves(), 0, 'and its switch back takes it off again');
    docked.env.stack = true;
    docked.win.listeners.fire('resize', {});
    assert.equal(moves(), 1, 'a resize is a fallback');
    docked.env.stack = false;
    docked.touch('touchstart', { x: 100, y: 300 });
    docked.touch('touchend', { x: 100, y: 300 });
    assert.equal(moves(), 0, 'and so is the next touch, for a change nobody announced');
    docked.dispose();
    assert.equal(docked.env.layoutHandlers.size, 0, 'dispose stops following the layout');
    const fine = world({ coarse: false });
    assert.equal(fine.root.listeners.map.get('touchmove')?.length || 0, 0, 'a fine pointer: none either');
    fine.dispose();
});

test('a menu that a long press opened under the still finger keeps it: a sideways move after it is not a drag', () => {
    const w = world();
    w.touch('touchstart', { x: 100, y: 300 });
    w.flags.blocked = true;  // 450 ms later the row's menu is open; the finger has not moved yet
    const moved = w.touch('touchmove', { x: 114, y: 300 });
    assert.equal(w.dragging(), false, 'no drawer under an open menu');
    assert.equal(moved.prevented, false, 'and the move is not cancelled');
    w.touch('touchend', { x: 114, y: 300 });
    assert.deepEqual(w.state.set, []);
    w.flags.blocked = false;
    const o = world({ open: true });
    o.scrim.hidden = false;
    o.touch('touchstart', { x: 300, y: 300, target: o.sidebar });
    o.flags.blocked = true;  // a row's menu inside the open drawer
    o.touch('touchmove', { x: 286, y: 300 });
    assert.equal(o.dragging(), false, 'nor a close drag under a menu inside the drawer');
    o.touch('touchend', { x: 286, y: 300 });
    assert.deepEqual(o.state.set, []);
    w.dispose(); o.dispose();
});

test('a finger that latches the drawer open or closed snaps once (the haptic); a spring back or the system does not', () => {
    const w = world();
    open(w, { from: 100, to: 100 + 12 + 150, steps: 5, dt: 200 });  // slow, under half: springs back
    assert.deepEqual(w.state.snaps, [], 'it went back where it was: nothing latched');
    w.runTimers();
    open(w);
    assert.deepEqual(w.state.snaps, [true], 'released, it latched open');
    w.touch('touchstart', { x: 300, y: 300, target: w.sidebar });
    w.touch('touchmove', { x: 288, y: 300, dt: 200 });
    w.touch('touchmove', { x: 288 - 110, y: 300, dt: 200 });
    w.touch('touchend', { x: 288 - 110, y: 300, dt: 200 });
    assert.deepEqual(w.state.snaps, [true, false], 'and latched closed');
    assert.deepEqual(w.state.order, ['snap:true', 'set:true', 'snap:false', 'set:false'],
        'each latch is felt as the finger leaves, before the state commits');
    w.runTimers();
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 112, y: 300 });
    w.touch('touchmove', { x: 112 + 200, y: 300, dt: 200 });
    w.touch('touchcancel', { x: 112 + 200, y: 300 });
    assert.deepEqual(w.state.set.at(-1), true, 'the system took the touch and the drawer opened…');
    assert.deepEqual(w.state.snaps, [true, false], '…but that was not the finger\'s latch');
    assert.equal(w.state.order.at(-1), 'set:true', 'the system\'s latch commits without a haptic');
    w.dispose();
});

test('what moves with the drawer (the room card, its veil) follows its number and settles with it, read when a drag takes hold', () => {
    const card = withStyle(fakeEl('card'));
    const veil = withStyle(fakeEl('veil'));
    let moving = [card];
    const w = world({ also: () => moving });
    moving = [card, veil, null];  // a room opened since the bind joins at the next drag; nothing is skipped quietly
    open(w, { to: 200, up: false });
    for (const el of [card, veil]) {
        assert.ok(el.classList.contains('dragging'), el.name);
        assert.equal(el.styleProps.get('--drawer-p'), w.sidebar.styleProps.get('--drawer-p'), el.name);
    }
    assert.ok(w.progress() > 0 && w.progress() < 1, 'halfway');
    w.touch('touchend', { x: 200, y: 300 });
    for (const el of [card, veil]) {
        assert.equal(el.classList.contains('dragging'), false, el.name);
        assert.equal(el.styleProps.has('--drawer-p'), false, el.name);
        assert.match(el.styleProps.get('--drawer-settle'), /^\d+ms$/, el.name);
    }
    w.runTimers();
    for (const el of [card, veil]) assert.equal(el.styleProps.has('--drawer-settle'), false, el.name);
    // A touch that never takes hold (the page scrolls) moves nothing, the card included.
    w.touch('touchstart', { x: 100, y: 300 });
    w.touch('touchmove', { x: 102, y: 340 });
    w.touch('touchend', { x: 102, y: 340 });
    assert.equal(card.classList.contains('dragging'), false);
    assert.equal(card.styleProps.has('--drawer-p'), false);
    w.dispose();
});

test('a finger that rested longer than the sample window before it lifted has no speed: where it stopped decides', () => {
    // Quickly to 30% open, then still: under half, it springs back closed (no fling read from moves long gone).
    let w = world();
    open(w, { from: 100, to: 100 + 12 + 96, steps: 4, dt: 16, up: false });  // 24px per 16ms, p = 0.3
    w.touch('touchend', { x: 208, y: 300, dt: DRAG.sampleWindow + 300 });
    assert.deepEqual(w.state.set, [], 'rested under half: springs back');
    w.dispose();
    // The open drawer quickly 12% of the way closed, then still: less than 30% gone, it stays open.
    w = world({ open: true });
    w.scrim.hidden = false;
    w.touch('touchstart', { x: 300, y: 300, target: w.sidebar });
    w.touch('touchmove', { x: 288, y: 300 });
    w.touch('touchmove', { x: 268, y: 300 });
    w.touch('touchmove', { x: 250, y: 300 });
    w.touch('touchend', { x: 250, y: 300, dt: DRAG.sampleWindow + 300 });
    assert.deepEqual(w.state.set, [], 'rested open: stays open');
    w.dispose();
    // The system taking a rested finger reads its place too.
    w = world();
    open(w, { from: 100, to: 100 + 12 + 96, steps: 4, dt: 16, up: false });
    w.touch('touchcancel', { x: 208, y: 300, dt: DRAG.sampleWindow + 300 });
    assert.deepEqual(w.state.set, [], 'taken after a rest under half: back closed');
    w.dispose();
    // Lifted while still moving (within the window), the same drag keeps the finger's speed: a fling.
    w = world();
    open(w, { from: 100, to: 100 + 12 + 96, steps: 4, dt: 16 });
    assert.deepEqual(w.state.set, [true], 'lifted moving: a fling opens');
    w.dispose();
});
