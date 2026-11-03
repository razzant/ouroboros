import test from 'node:test';
import assert from 'node:assert/strict';
import { bindScrollFade } from '../modules/scroll_fade.js';

function fixture() {
    const attrs = new Set(), listeners = new Map(), frames = new Map(), observers = [];
    let next = 0;
    class Observer {
        constructor(callback) { this.callback = callback; this.disconnected = false; observers.push(this); }
        observe() { this.disconnected = false; }
        disconnect() { this.disconnected = true; }
    }
    const win = { ResizeObserver: Observer, MutationObserver: Observer,
        requestAnimationFrame(fn) { frames.set(++next, fn); return next; },
        cancelAnimationFrame(id) { frames.delete(id); } };
    const scroller = { ownerDocument: { defaultView: win }, children: [],
        scrollLeft: 0, scrollTop: 0, scrollWidth: 700, clientWidth: 300,
        scrollHeight: 500, clientHeight: 100,
        toggleAttribute(name, on) { on ? attrs.add(name) : attrs.delete(name); },
        addEventListener(name, fn) { listeners.set(name, fn); },
        removeEventListener(name, fn) { if (listeners.get(name) === fn) listeners.delete(name); } };
    const flush = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn()); };
    return { scroller, attrs, listeners, frames, observers, flush };
}

for (const axis of ['x', 'y']) {
    test(`${axis} fade follows both edges, content resize and disposal`, () => {
        const f = fixture();
        const start = axis === 'x' ? 'data-scroll-start' : 'data-scroll-above';
        const end = axis === 'x' ? 'data-scroll-end' : 'data-scroll-below';
        const position = axis === 'x' ? 'scrollLeft' : 'scrollTop';
        const dispose = axis === 'x' ? bindScrollFade(f.scroller, {axis}) : bindScrollFade(f.scroller);
        f.flush();
        assert.deepEqual([...f.attrs], [end]);
        f.scroller[position] = 200; f.listeners.get('scroll')(); f.flush();
        assert.equal(f.attrs.has(start) && f.attrs.has(end), true);
        f.scroller[position] = 400; f.listeners.get('scroll')(); f.flush();
        assert.deepEqual([...f.attrs], [start]);
        f.scroller[axis === 'x' ? 'scrollWidth' : 'scrollHeight'] += 200;
        f.observers[0].callback(); f.flush();
        assert.equal(f.attrs.has(end), true);
        f.listeners.get('scroll')(); dispose();
        assert.equal(f.frames.size, 0);
        assert.equal(f.listeners.size, 0);
        assert.equal(f.observers.every(o => o.disconnected), true);
        f.observers[0].callback();
        assert.equal(f.frames.size, 0);
    });
}

// A named body (a capped part of a task card's Details, chat.js) is a region the keyboard can focus and scroll exactly
// while it scrolls: WebKit offers no other way into a scroller with nothing focusable inside.
test('a named body is a keyboard region exactly while it scrolls, and nothing without a name', () => {
    const f = fixture();
    const named = new Map();
    Object.defineProperty(f.scroller, 'tabIndex', { set(value) { named.set('tabindex', String(value)); } });
    Object.assign(f.scroller, {
        setAttribute(name, value) { named.set(name, String(value)); },
        removeAttribute(name) { named.delete(name); },
        hasAttribute(name) { return named.has(name); },
    });
    const dispose = bindScrollFade(f.scroller, { region: 'Scrollable task notes' });
    f.flush();
    assert.deepEqual(Object.fromEntries(named), { tabindex: '0', role: 'region', 'aria-label': 'Scrollable task notes' });
    f.scroller.scrollHeight = f.scroller.clientHeight;  // short now, or hidden: both measure no extent
    f.observers[0].callback(); f.flush();
    assert.deepEqual([...named.keys()], [], 'no tab stop while it does not scroll');
    f.scroller.scrollHeight = 500;
    f.observers[0].callback(); f.flush();
    assert.equal(named.get('role'), 'region', 'a region again once it scrolls');
    dispose();
    const plain = fixture();  // its stub has no attribute API: a body without a name is only decorated
    assert.doesNotThrow(() => { bindScrollFade(plain.scroller); plain.flush(); });
});
