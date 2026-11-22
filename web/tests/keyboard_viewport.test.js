import assert from 'node:assert/strict';
import test from 'node:test';
import { bindKeyboardViewport } from '../modules/keyboard_viewport.js';

class Target {
    listeners = new Map();
    addEventListener(type, fn, options) {
        const key = `${type}:${typeof options === 'object' ? Boolean(options?.capture) : Boolean(options)}`;
        if (!this.listeners.has(key)) this.listeners.set(key, new Set());
        this.listeners.get(key).add(fn);
    }
    removeEventListener(type, fn, options) {
        this.listeners.get(`${type}:${typeof options === 'object' ? Boolean(options?.capture) : Boolean(options)}`)?.delete(fn);
    }
    count(type, capture) {
        return [...this.listeners].filter(([key]) => (!type || key.startsWith(`${type}:`))
            && (capture === undefined || key.endsWith(`:${capture}`))).reduce((n, [, set]) => n + set.size, 0);
    }
    emit(type, init = {}) {
        const event = { type, currentTarget: this, target: this, ...init };
        for (const [key, set] of this.listeners) if (key.startsWith(`${type}:`)) for (const fn of [...set]) fn(event);
        return event;
    }
}

const classList = () => {
    const names = new Set();
    return {
        names,
        contains: (n) => names.has(n),
        remove: (n) => names.delete(n),
        toggle: (n, on = !names.has(n)) => { if (on) names.add(n); else names.delete(n); return on; },
    };
};

function world() {
    const frames = new Map(), timers = new Map();
    let now = 0, ids = 0, reads = 0;
    class Element {}
    const makeField = () => Object.assign(new Element(), {
        closest() { return this; }, matches: () => true, disabled: false, readOnly: false, blur() { doc.activeElement = doc.body; },
    });
    const field = makeField(), otherField = makeField();
    const head = { children: [], appendChild(node) { this.children.push(node); node.parent = this; } };
    const doc = Object.assign(new Target(), {
        head,
        body: { classList: classList() },
        activeElement: null,
        documentElement: {
            dataset: { layout: 'stack', pointer: 'coarse' },
            classList: classList(),
            clientWidth: 390,
            clientHeight: 844,
        },
        createElement: () => ({ textContent: '', remove() { head.children = head.children.filter((n) => n !== this); } }),
    });
    let vvHeight = 844;
    const vv = Object.assign(new Target(), { width: 390 });
    Object.defineProperty(vv, 'height', { get() { reads++; return vvHeight; }, set(v) { vvHeight = v; } });
    const win = Object.assign(new Target(), {
        innerHeight: 844,
        innerWidth: 390,
        scrollY: 0,
        visualViewport: vv,
        document: doc,
        scrollTo(_x, y) { this.scrollY = y; },
    });
    const globals = {
        window: win,
        document: doc,
        Element,
        requestAnimationFrame: (fn) => { frames.set(++ids, fn); return ids; },
        cancelAnimationFrame: (id) => frames.delete(id),
        setTimeout: (fn, ms) => { timers.set(++ids, { fn, at: now + ms }); return ids; },
        clearTimeout: (id) => timers.delete(id),
        performance: { now: () => now },
    };
    const prior = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    for (const [key, value] of Object.entries(globals)) {
        Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
    }
    const restore = () => {
        for (const [key, descriptor] of Object.entries(prior)) {
            if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
        }
    };
    const frame = () => {
        for (const [id, fn] of [...frames]) { frames.delete(id); fn(); }
    };
    const run = (ms) => {
        frame();
        frame();
        const end = now + ms;
        for (let next; (next = [...timers].sort((a, b) => a[1].at - b[1].at)[0]) && next[1].at <= end;) {
            timers.delete(next[0]);
            now = next[1].at;
            next[1].fn();
            frame();
            frame();
        }
        now = end;
    };
    const focus = (on, relatedTarget = null) => {
        doc.activeElement = on ? field : (relatedTarget || doc.body);
        doc.emit(on ? 'focusin' : 'focusout', { target: field, relatedTarget });
    };
    return { win, doc, vv, head, field, otherField, frames, timers, frame, run, focus, reads: () => reads, restore };
}

const postBlurReads = (doc) => ['pointerdown', 'touchstart', 'scroll'].reduce((n, type) => n + doc.count(type, true), 0);

const read = (w) => {
    const css = w.head.children[0]?.textContent || '';
    const px = (name) => Number(css.match(new RegExp(`${name}:(\\d+)px`))?.[1]);
    return { css, vvh: px('--vvh'), cap: px('--chat-viewport-input-cap'), open: w.doc.body.classList.contains('keyboard-open') };
};

test('teardown releases every listener, layout subscription, pending settle and mounted style', () => {
    const w = world();
    try {
        const release = bindKeyboardViewport({ shownRoom: () => null });
        assert.equal(typeof release, 'function');
        assert.equal(w.head.children.length, 1);
        w.focus(true);
        w.vv.height = 508;
        w.vv.emit('resize');
        assert.ok(w.doc.documentElement.classList.contains('keyboard-open'));
        assert.equal(w.doc.count('touchmove'), 1);
        w.focus(false);
        assert.equal(postBlurReads(w.doc), 3);
        assert.ok(w.frames.size > 0);
        w.frame();
        w.frame();
        assert.ok(w.timers.size > 0);
        release.dispose();
        assert.equal(w.win.count() + w.doc.count() + w.vv.count(), 0);
        assert.equal(w.frames.size + w.timers.size, 0);
        assert.equal(w.head.children.length, 0);
        assert.equal(w.doc.body.classList.names.size + w.doc.documentElement.classList.names.size, 0);
        release.dispose();
        assert.equal(w.win.count() + w.doc.count() + w.vv.count(), 0);
    } finally { w.restore(); }
});

test('room height follows visualViewport across events, two animation frames, one settle timer, and the first post-blur input', () => {
    for (const signal of ['pointerdown', 'touchstart', 'scroll']) {
        const w = world();
        try {
            const release = bindKeyboardViewport({ shownRoom: () => null });
            assert.deepEqual(read(w), { css: ':root{--vvh:844px;--chat-viewport-input-cap:140px}', vvh: 844, cap: 140, open: false });

            // Focus schedules two animation frames and one settle timer.
            w.focus(true);
            w.vv.height = 520;
            w.frame();
            assert.equal(read(w).vvh, 520, 'first animation frame re-reads');
            w.vv.height = 508;
            w.frame();
            assert.equal(read(w).vvh, 508, 'second animation frame re-reads');
            assert.equal(w.frames.size, 0);
            w.vv.height = 500;
            w.run(300);
            assert.equal(read(w).vvh, 500, 'short settle timer re-reads');
            assert.equal(w.timers.size, 0);
            assert.equal(read(w).open, true);

            // Blur keeps keyboard-open for the synchronous click task, then clears on frame 1, runs frame 2 + settle timer,
            // and waits for one post-blur input.
            w.focus(false);
            assert.equal(read(w).open, true, 'tap targets do not shift mid-press before click lands');
            w.frame();
            assert.equal(read(w).open, false, 'cleared on the next animation frame');
            w.run(300);
            assert.equal(w.frames.size + w.timers.size, 0);
            assert.equal(postBlurReads(w.doc), 3);

            w.vv.height = 844; // silent restore after the settle timer
            const before = w.reads();
            w.doc.emit(signal);
            assert.equal(read(w).vvh, 844, signal);
            assert.equal(postBlurReads(w.doc), 0, `${signal} disarms the one-shot listener`);
            const afterFirst = w.reads();
            w.doc.emit(signal);
            assert.equal(w.reads(), afterFirst, 'subsequent touches do not re-read on their own');
            assert.ok(afterFirst > before);
            release.dispose();
        } finally { w.restore(); }
    }
});

test('document scroll resets to 0 on stack viewport events while leaving pinch zoom untouched', () => {
    const w = world();
    try {
        const release = bindKeyboardViewport({ shownRoom: () => null });
        for (const [target, event] of [
            [w.win, 'scroll'],
            [w.vv, 'scroll'],
            [w.vv, 'resize'],
            [w.win, 'orientationchange'],
            [w.win, 'ouro:host-viewport'],
            [w.win, 'ouro:host-chrome'],
        ]) {
            w.win.scrollY = 220;
            target.emit(event);
            assert.equal(w.win.scrollY, 0, event);
        }

        // Magnified pinch keeps its pan and room height, while still releasing keyboard-open on blur.
        w.focus(true);
        w.vv.height = 508;
        w.vv.emit('resize');
        assert.equal(read(w).open, true);
        w.vv.width = 195;
        w.win.scrollY = 90;
        w.vv.emit('resize');
        assert.equal(w.win.scrollY, 90, 'pinch pan is preserved');
        assert.equal(read(w).vvh, 508, 'room height stays untouched during pinch');
        w.focus(false);
        w.frame();
        assert.equal(read(w).open, false, 'blur clears cosmetic keyboard-open even while pinched');
        w.vv.width = 390;
        w.vv.emit('resize');
        w.vv.height = 844; // silent restore after pinch ends
        w.doc.emit('pointerdown');
        assert.equal(read(w).vvh, 844);
        release.dispose();
    } finally { w.restore(); }
});

test('keyboard-open is never set on a fine pointer and resets on width change, Telegram mode change, and pageshow', () => {
    const w = world();
    try {
        const release = bindKeyboardViewport({ shownRoom: () => null });

        // Fine pointer: never keyboard-open, even when visual or paired height shrinks.
        w.doc.documentElement.dataset.pointer = 'fine';
        w.focus(true);
        w.vv.height = 480;
        w.vv.emit('resize');
        assert.deepEqual(read(w), { css: ':root{--vvh:480px;--chat-viewport-input-cap:120px}', vvh: 480, cap: 120, open: false });

        // Coarse pointer: paired shrink opens, Telegram mode change resets the captured focus height.
        w.doc.documentElement.dataset.pointer = 'coarse';
        w.doc.documentElement.dataset.telegramFullscreen = 'active';
        w.win.innerHeight = w.vv.height = 844;
        w.win.emit('ouro:host-chrome');
        w.focus(false);
        w.focus(true);
        w.win.innerHeight = w.vv.height = 508;
        w.win.emit('resize');
        assert.equal(read(w).open, true);

        // Moving focus directly to another editable keeps the captured focus height.
        w.focus(false, w.otherField);
        w.doc.emit('focusin', { target: w.otherField });
        assert.equal(read(w).open, true);

        // Telegram mode change resets focusHeight: expanded at 760 is not open, and a subsequent shrink in expanded opens.
        w.doc.documentElement.dataset.telegramFullscreen = 'expanded';
        w.win.innerHeight = w.vv.height = 760;
        w.win.emit('ouro:host-chrome');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 760, open: false });
        w.win.innerHeight = w.vv.height = 440;
        w.win.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 440, open: true });

        // Android Back restores height while keeping focus, then width change (rotation) with hidden keyboard stays closed.
        w.win.innerHeight = w.vv.height = 760;
        w.win.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 760, open: false });
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 667;
        w.win.innerHeight = w.vv.height = 375;
        w.win.emit('orientationchange');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 375, open: false });

        // pageshow from bfcache resets focusHeight and reads current viewport.
        w.win.innerHeight = w.vv.height = 200;
        w.win.emit('resize');
        assert.equal(read(w).open, true);
        w.win.emit('pagehide', { persisted: true });
        assert.equal(w.frames.size + w.timers.size + postBlurReads(w.doc), 0);
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 390;
        w.win.innerHeight = w.vv.height = 844;
        w.win.emit('pageshow', { persisted: true });
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 844, open: false });

        // Docked layout uses 100dvh and clears keyboard-open.
        w.doc.documentElement.dataset.layout = 'docked';
        w.win.emit('ouro:layout-changed');
        assert.deepEqual(read(w), { css: ':root{--vvh:100dvh}', vvh: Number.NaN, cap: Number.NaN, open: false });

        release.dispose();
    } finally { w.restore(); }
});

test('shownRoom remeasures the composer and reflows the reading place synchronously on height and class changes', () => {
    const w = world();
    try {
        const calls = [];
        let shortAttr = false;
        const page = {
            ownerDocument: w.doc,
            getBoundingClientRect: () => ({ height: 210 }),
            querySelector: () => ({ getBoundingClientRect: () => ({ height: 93 }) }),
            hasAttribute: () => shortAttr,
            toggleAttribute: (_name, on) => { shortAttr = on; calls.push(`short:${on}`); },
        };
        const room = {
            page,
            composer: { remeasureInput(fn) { calls.push('remeasure:start'); fn(); calls.push('remeasure:end'); } },
            reflowViewport() { calls.push(`reflow:${read(w).vvh}:${read(w).open}`); },
        };
        const release = bindKeyboardViewport({ shownRoom: () => room });
        calls.length = 0;
        w.focus(true);
        w.vv.height = 210;
        w.vv.emit('resize');
        assert.deepEqual(calls, ['remeasure:start', 'short:true', 'remeasure:end', 'reflow:210:true']);

        // Identical geometry writes nothing and triggers no redundant reflow.
        calls.length = 0;
        w.vv.emit('resize');
        assert.deepEqual(calls, []);
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard sequence 1: rotating 360x640 -> 360x380 -> 640x180 with keyboard open keeps keyboard-open', () => {
    const w = world();
    try {
        w.doc.documentElement.dataset.telegramFullscreen = 'expanded';
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 640;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        w.focus(true);
        w.win.innerHeight = w.vv.height = 380;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 380, open: true });

        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 640;
        w.win.innerHeight = w.vv.height = 180;
        w.win.emit('resize');
        w.vv.emit('resize');
        w.win.emit('orientationchange');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 180, open: true });
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard sequence 2: switching Telegram expanded -> fullscreen (360x568 -> 360x216 -> 360x288) keeps keyboard-open', () => {
    const w = world();
    try {
        w.doc.documentElement.dataset.telegramFullscreen = 'expanded';
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 568;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        w.focus(true);
        w.win.innerHeight = w.vv.height = 216;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 216, open: true });

        w.doc.documentElement.dataset.telegramFullscreen = 'active';
        w.win.emit('ouro:host-chrome');
        w.win.innerHeight = w.vv.height = 288;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 288, open: true });
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard sequence 3: next-frame blur and refocus at 360x236 while keyboard stays up keeps keyboard-open', () => {
    const w = world();
    try {
        w.doc.documentElement.dataset.telegramFullscreen = 'active';
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 640;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        w.focus(true);
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 236, open: true });

        w.focus(false);
        w.frame(); // frame 1 clears the old focus session
        assert.equal(read(w).open, false);
        w.focus(true); // refocus on the next frame while IME height 236 still stands
        w.run(300);
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 236, open: true });
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard sequence 4: late Telegram SDK mode report and initial pageshow at 360x236 keep keyboard-open', () => {
    const w = world();
    try {
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 600;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        w.focus(true);
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 236, open: true });

        w.doc.documentElement.dataset.telegramFullscreen = 'expanded';
        w.win.emit('ouro:host-chrome');
        w.win.emit('pageshow', { persisted: false });
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 236, open: true });
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard negative 1: Android Back with focus kept (viewport grows back) clears keyboard-open', () => {
    const w = world();
    try {
        w.doc.documentElement.dataset.telegramFullscreen = 'active';
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 640;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        // Direct open then Android Back restore at 360x640.
        w.focus(true);
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        assert.equal(read(w).open, true);
        w.win.innerHeight = w.vv.height = 640;
        w.win.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 640, open: false });

        // Hold open across rotation (360x640 -> 360x380 -> 640x180), then Android Back restores 640x360 with focus kept.
        w.win.innerHeight = w.vv.height = 380;
        w.win.emit('resize');
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 640;
        w.win.innerHeight = w.vv.height = 180;
        w.win.emit('resize');
        assert.equal(read(w).open, true);

        w.win.innerHeight = w.vv.height = 360;
        w.win.emit('resize');
        w.vv.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 360, open: false });

        // Reopening at 640x180 still works without blurring.
        w.win.innerHeight = w.vv.height = 180;
        w.win.emit('resize');
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 180, open: true });
        release.dispose();
    } finally { w.restore(); }
});

test('paired keyboard negative 2: quick refocus after the keyboard really closed does not reopen', () => {
    const w = world();
    try {
        w.doc.documentElement.dataset.telegramFullscreen = 'active';
        w.win.innerWidth = w.vv.width = w.doc.documentElement.clientWidth = 360;
        w.win.innerHeight = w.vv.height = 640;
        const release = bindKeyboardViewport({ shownRoom: () => null });

        w.focus(true);
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        assert.equal(read(w).open, true);
        w.focus(false);
        w.run(50);
        w.win.innerHeight = w.vv.height = 640;
        w.win.emit('resize');
        w.focus(true);
        w.run(300);
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 640, open: false },
            'quick refocus after real keyboard close does not reopen');

        // Same negative when the viewport silently grew back before focusin fired.
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        assert.equal(read(w).open, true);
        w.focus(false);
        w.frame();
        w.win.innerHeight = w.vv.height = 640;
        w.focus(true);
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 640, open: false },
            'silent restore before quick refocus also drops recent-open memory');

        // Timeout (> 1000 ms), width change, or mode change also drop quick-refocus memory.
        w.win.innerHeight = w.vv.height = 236;
        w.win.emit('resize');
        assert.equal(read(w).open, true);
        w.focus(false);
        w.run(1100);
        w.focus(true);
        assert.deepEqual({ vvh: read(w).vvh, open: read(w).open }, { vvh: 236, open: false },
            'refocus after timeout starts a fresh baseline');

        release.dispose();
    } finally { w.restore(); }
});
