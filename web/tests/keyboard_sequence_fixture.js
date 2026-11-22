// Random event-order generator for keyboard_viewport.js across six host models (iOS/Android Telegram, iOS/Android
// browser, desktop fine pointer, tablet coarse pointer). Checks the contract invariants after every settle:
// I1: room height (--vvh) equals Math.ceil(visualViewport.height) in stack layout, 100dvh when docked;
// I2: window.scrollY is 0 in stack layout (unless pinch-zoomed);
// I3: unfocused is never keyboard-open, and a fine primary pointer is never keyboard-open.
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
    count() { return [...this.listeners.values()].reduce((n, set) => n + set.size, 0); }
    emit(type, init = {}) {
        const event = { type, currentTarget: this, target: this, ...init };
        for (const [key, set] of this.listeners) if (key.startsWith(`${type}:`)) for (const fn of [...set]) fn(event);
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

function random(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const PHONES = [
    { w: 390, h: 844, bar: [84, 84] },
    { w: 375, h: 667, bar: [84, 87] },
    { w: 360, h: 640, bar: [80, 82] },
    { w: 320, h: 568, bar: [76, 80] },
    { w: 414, h: 896, bar: [88, 90] },
];

export const HOSTS = ['H1', 'H2', 'H3i', 'H3a', 'H4f', 'H4c'];

export function sequence(kind, seed, { length = 24 } = {}) {
    const r = random(seed);
    const pick = (list) => list[Math.floor(r() * list.length)];
    const chance = (p) => r() < p;
    const telegram = kind === 'H1' || kind === 'H2';
    const desk = kind === 'H4f' || kind === 'H4c';
    const soft = !desk;
    const paired = kind === 'H2';
    const silentHost = kind === 'H1' || kind === 'H3i';
    const phone = pick(PHONES);
    const share = { p: 0.36 + r() * 0.06, l: 0.5 + r() * 0.12 };
    const T = {
        orient: chance(0.2) ? 'l' : 'p',
        mode: telegram ? pick(['active', 'expanded']) : '',
        k: 0,
        focused: false,
        scale: 1,
        winW: 300 + Math.floor(r() * 1100),
        winH: 300 + Math.floor(r() * 700),
    };

    const width = () => (desk ? T.winW : T.orient === 'p' ? phone.w : phone.h);
    const full = () => (desk ? T.winH : (T.orient === 'p' ? phone.h : phone.w)
        - (T.mode === 'expanded' ? phone.bar[T.orient === 'p' ? 0 : 1] : 0));
    const keyboard = () => Math.min(full() - 48, Math.round((T.orient === 'p' ? phone.h : phone.w) * share[T.orient]));
    const layoutOf = () => (paired ? full() - T.k : full());
    const visualOf = () => (full() - T.k) / T.scale;

    const frames = new Map(), timers = new Map();
    let now = 0, ids = 0, unreadSilent = false;
    class Element {}
    const field = Object.assign(new Element(), { closest() { return this; }, matches: () => true, disabled: false, readOnly: false });
    const head = { children: [], appendChild(node) { this.children.push(node); } };
    const root = {
        dataset: { layout: width() < 720 ? 'stack' : 'docked', pointer: kind === 'H4f' ? 'fine' : 'coarse' },
        classList: classList(),
        clientWidth: width(),
        clientHeight: layoutOf(),
    };
    if (telegram) root.dataset.telegramFullscreen = T.mode;
    const doc = Object.assign(new Target(), {
        head,
        body: { classList: classList() },
        activeElement: null,
        documentElement: root,
        createElement: () => ({ textContent: '', remove() { head.children = head.children.filter((n) => n !== this); } }),
    });
    const vv = Object.assign(new Target(), { width: width(), height: visualOf() });
    const win = Object.assign(new Target(), {
        innerHeight: layoutOf(),
        innerWidth: width(),
        scrollY: 0,
        visualViewport: vv,
        document: doc,
        scrollTo(_x, y) { this.scrollY = y; },
    });
    const globals = {
        window: win,
        document: doc,
        Element,
        performance: { now: () => now },
        requestAnimationFrame: (fn) => { frames.set(++ids, fn); return ids; },
        cancelAnimationFrame: (id) => frames.delete(id),
        setTimeout: (fn, ms) => { timers.set(++ids, { fn, at: now + ms }); return ids; },
        clearTimeout: (id) => timers.delete(id),
    };
    const prior = Object.fromEntries(Object.keys(globals).map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
    for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { configurable: true, writable: true, value: v });

    const run = (ms) => {
        const end = now + ms;
        do {
            for (let next; (next = [...timers].sort((a, b) => a[1].at - b[1].at)[0]) && next[1].at <= now;) {
                timers.delete(next[0]);
                next[1].fn();
            }
            for (const [id, fn] of [...frames]) { frames.delete(id); fn(); }
            if (now >= end) break;
            now = Math.min(end, now + 16);
        } while (now <= end);
    };

    const syncGeometry = (silent = false) => {
        const layoutMoved = win.innerHeight !== layoutOf() || win.innerWidth !== width();
        win.innerHeight = layoutOf();
        win.innerWidth = root.clientWidth = width();
        vv.height = visualOf();
        vv.width = width() / T.scale;
        if (silent) { unreadSilent = true; return; }
        unreadSilent = false;
        const order = layoutMoved ? pick(['layout-first', 'visual-first', 'together']) : 'visual-only';
        if (order === 'layout-first') { win.emit('resize'); vv.emit('resize'); }
        else if (order === 'visual-first') { vv.emit('resize'); win.emit('resize'); }
        else if (order === 'together') { win.emit('resize'); vv.emit('resize'); }
        else vv.emit('resize');
        const layout = width() < 720 ? 'stack' : 'docked';
        if (layout !== root.dataset.layout) { root.dataset.layout = layout; win.emit('ouro:layout-changed'); }
    };

    let checks = 0;
    const check = (label) => {
        checks++;
        const css = head.children[0]?.textContent || '';
        const vvh = Number(css.match(/--vvh:(-?[\d.]+)px/)?.[1]);
        const open = doc.body.classList.contains('keyboard-open');
        if (width() >= 720) {
            if (css !== ':root{--vvh:100dvh}' || open) {
                throw new Error(`${kind} seed ${seed} (${label}): docked mismatch ${css} open=${open}`);
            }
            return;
        }
        if (!T.focused && open) throw new Error(`${kind} seed ${seed} (${label}): keyboard-open while unfocused`);
        if (kind === 'H4f' && open) throw new Error(`${kind} seed ${seed} (${label}): keyboard-open on fine pointer`);
        if (T.scale > 1) return;
        if (vvh !== Math.ceil(vv.height)) {
            throw new Error(`${kind} seed ${seed} (${label}): --vvh=${vvh} !== ceil(${vv.height})`);
        }
        if (win.scrollY !== 0) {
            throw new Error(`${kind} seed ${seed} (${label}): scrollY=${win.scrollY} !== 0`);
        }
    };

    const settle = (label) => {
        run(350);
        if (unreadSilent) {
            doc.emit(pick(['pointerdown', 'touchstart', 'scroll']));
            unreadSilent = false;
            run(32);
        }
        check(label);
    };

    try {
        const release = bindKeyboardViewport({ shownRoom: () => null });
        run(32);
        for (let i = 0; i < length; i++) {
            const action = pick(desk
                ? ['focus', 'blur', 'resize', 'scroll', 'cached', 'wait']
                : ['focus', 'blur', 'turn', 'pinch', 'mode', 'scroll', 'collapse', 'cached', 'wait']);
            if (action === 'focus' && !T.focused) {
                T.focused = true;
                doc.activeElement = field;
                if (silentHost && chance(0.4)) win.scrollY = 180;
                doc.emit('focusin', { target: field });
                if (soft) { T.k = keyboard(); syncGeometry(false); }
            } else if (action === 'blur' && T.focused) {
                T.focused = false;
                doc.activeElement = doc.body;
                doc.emit('focusout', { target: field });
                if (chance(0.3)) doc.emit('pointerdown'); // early tap during close
                run(pick([0, 40, 300]));
                T.k = 0;
                syncGeometry(silentHost && chance(0.45));
            } else if (action === 'turn' && !desk) {
                T.orient = T.orient === 'p' ? 'l' : 'p';
                if (T.focused && T.k > 0) T.k = chance(0.6) ? keyboard() : 0;
                syncGeometry(false);
                win.emit('orientationchange');
            } else if (action === 'pinch' && !desk) {
                T.scale = 2;
                syncGeometry(false);
                if (T.focused && chance(0.5)) {
                    T.focused = false;
                    doc.activeElement = doc.body;
                    doc.emit('focusout', { target: field });
                    T.k = 0;
                }
                run(pick([0, 50]));
                T.scale = 1;
                syncGeometry(silentHost && !T.focused && chance(0.4));
            } else if (action === 'mode' && telegram) {
                T.mode = T.mode === 'active' ? 'expanded' : 'active';
                if (chance(0.5)) {
                    root.dataset.telegramFullscreen = T.mode;
                    win.emit('ouro:host-chrome');
                    syncGeometry(false);
                } else {
                    syncGeometry(false);
                    root.dataset.telegramFullscreen = T.mode;
                    win.emit('ouro:host-chrome');
                }
            } else if (action === 'collapse' && telegram && T.focused) {
                vv.height = 60;
                vv.emit('resize');
                T.focused = false;
                doc.activeElement = doc.body;
                doc.emit('focusout', { target: field });
                T.k = 0;
                run(300);
                syncGeometry(true); // silent restore after host collapse
            } else if (action === 'resize' && desk) {
                T.winW = 300 + Math.floor(r() * 1100);
                T.winH = 300 + Math.floor(r() * 700);
                syncGeometry(false);
            } else if (action === 'scroll') {
                win.scrollY = 140;
                win.emit('scroll');
            } else if (action === 'cached') {
                win.emit('pagehide', { persisted: true });
                if (!desk && chance(0.5)) T.orient = T.orient === 'p' ? 'l' : 'p';
                T.k = 0;
                T.focused = false;
                doc.activeElement = doc.body;
                syncGeometry(true);
                unreadSilent = false;
                root.dataset.layout = width() < 720 ? 'stack' : 'docked';
                win.emit('pageshow', { persisted: true });
            } else {
                run(pick([0, 16, 80, 260]));
            }
            if (chance(0.65)) settle(`step ${i}`);
        }
        if (T.focused) {
            T.focused = false;
            doc.activeElement = doc.body;
            doc.emit('focusout', { target: field });
            T.k = 0;
            syncGeometry(silentHost && chance(0.5));
        }
        settle('end');
        release.dispose();
        if (win.count() + doc.count() + vv.count() + frames.size + timers.size) {
            throw new Error(`${kind} seed ${seed}: teardown left listeners or timers behind`);
        }
        return { checks };
    } finally {
        for (const [k, d] of Object.entries(prior)) {
            if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
        }
    }
}
