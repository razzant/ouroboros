import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bindScrollFade, revealPastFade } from '../modules/scroll_fade.js';

// A room's feed carries no fade (DESIGN principle 9, owner 21.10): what darkens content under its glass is the
// bar's and the composer's own drawing. A mask on the feed would also make it one stacking context and drop the
// lazy-load margin of the photos inside it. Every other scroller keeps the shared fade.
const ROOM_FEED = /(?:#chat-messages|\.chat-messages)(?![\w-])/;
function topLevel(text, separator) {  // split outside parentheses, so `:is(a, b) .c` keeps its group
    const parts = [];
    let depth = 0, start = 0;
    [...text].forEach((char, index) => {
        if (char === '(') depth += 1;
        else if (char === ')') depth -= 1;
        else if (!depth && separator.test(char)) { parts.push(text.slice(start, index)); start = index + 1; }
    });
    return [...parts, text.slice(start)].map((part) => part.trim()).filter(Boolean);
}
function roomFeedFades(css) {
    css = css.replace(/\/\*[\s\S]*?\*\//g, '');
    const found = [];
    for (const [, selectors, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
        const subjects = topLevel(selectors, /,/).map((one) => topLevel(one, /[\s>+~]/).at(-1));
        const masked = /(?:^|[;\s])(?:-webkit-)?mask(?:-image)?\s*:(?!\s*none\b)|--scroll-fade-/.test(body);
        if (/data-room-top/.test(selectors) || (masked && subjects.some((subject) => ROOM_FEED.test(subject)))) {
            found.push(selectors.trim());
        }
    }
    return found;
}

test("a room's feed carries no fade or room-top switch, while the shared fade still masks every other scroller", () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.deepEqual(roomFeedFades(css), []);
    assert.match(css, /\.scroll-fade-y \{[^}]*\bmask-image: linear-gradient\(/, 'the shared fade itself stays');
    for (const broken of [
        ':is(#chat-messages, .chat-messages)[data-scroll-above] { --scroll-fade-top: 88px; }',
        '#page-chat .chat-messages { mask-image: linear-gradient(transparent, black); }',
        '[data-room-top="crisp"] .chat-messages { mask-image: none; }',
    ]) assert.equal(roomFeedFades(broken).length, 1, broken);
    const allowed = `.scroll-fade-y { mask-image: linear-gradient(180deg, transparent 0, black 32px); }
        .chat-live-timeline.scroll-fade-y[data-scroll-above] { --scroll-fade-top: 32px; }
        .chat-messages .chat-mirror-text { mask-image: linear-gradient(black, transparent); }
        :is(#chat-messages, .chat-messages) > .chat-bubble { -webkit-mask-image: linear-gradient(black, transparent); }
        .project-panel .chat-messages:focus-visible { background-image: linear-gradient(red, red); }`;
    assert.deepEqual(roomFeedFades(allowed), [], 'a card or line inside the feed may still fade on its own');
});

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

test('a revealed row clears the sheet’s current fade, including a row already inside the viewport', () => {
    const { scroller } = fixture();
    let edge = 32;
    scroller.ownerDocument.defaultView.getComputedStyle = () => ({ getPropertyValue: name => name === '--scroll-fade-edge' ? `${edge}px` : '' });
    scroller.getBoundingClientRect = () => ({ top: 20, bottom: 120 });
    const row = { getBoundingClientRect: () => ({ top: 80, bottom: 110 }) };
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 22, 'the last ten pixels were inside the box but under the fade');
    edge = 40; scroller.scrollTop = 0;
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 30, 'a different CSS token changes the clearance without a JS constant');
    scroller.scrollTop = 400; // settled bottom: the same row has no bottom fade to clear
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 400);
});

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

test('keyboard focus clears the fade, while pointer focus and focus handed back after a pointer leave the list still', () => {
    const f = fixture();
    f.scroller.ownerDocument.defaultView.getComputedStyle = () => ({ getPropertyValue: () => '32px' });
    let keyboard = false;
    const row = { getBoundingClientRect: () => ({ top: 20, bottom: 50 }), matches: selector => selector === ':focus-visible' && keyboard };
    f.scroller.contains = (node) => node === row || node === f.scroller;
    f.scroller.getBoundingClientRect = () => ({ top: 0, bottom: 100 });
    f.scroller.scrollTop = 200;
    bindScrollFade(f.scroller);
    f.listeners.get('focusin')({ target: f.scroller });
    assert.equal(f.scroller.scrollTop, 200, 'landing focus on the scroller itself is not a row to reveal');
    f.listeners.get('focusin')({ target: row });
    assert.equal(f.scroller.scrollTop, 200, 'pointer focus, or focus a closed menu hands back, does not move the list');
    keyboard = true;
    f.listeners.get('focusin')({ target: row });
    assert.equal(f.scroller.scrollTop, 188, 'the row clears the 32px top fade before focus is visible');
});

test('keyboard focus handed back to the row it left stays where it stood; a row that moved meanwhile comes clear', () => {
    const f = fixture();
    f.scroller.ownerDocument.defaultView.getComputedStyle = () => ({ getPropertyValue: () => '32px' });
    let top = 20;  // a reader's opener under the top fade, half above the box
    const row = { getBoundingClientRect: () => ({ top, bottom: top + 30 }), matches: selector => selector === ':focus-visible' };
    const other = { getBoundingClientRect: () => ({ top: 60, bottom: 70 }), matches: () => true };
    f.scroller.contains = (node) => node === row || node === other || node === f.scroller;
    f.scroller.getBoundingClientRect = () => ({ top: 0, bottom: 100 });
    f.scroller.scrollTop = 200;
    bindScrollFade(f.scroller);
    // The reader opened from the row takes the focus; Escape closes it and hands the focus back (a keyboard's).
    f.listeners.get('focusout')({ target: row });
    f.listeners.get('focusin')({ target: row });
    assert.equal(f.scroller.scrollTop, 200, 'the opener returns at the same place: the chat does not move');
    // A row the list moved while its menu was open (a project pinned to the top) is revealed on the keyboard's return.
    f.listeners.get('focusout')({ target: row });
    top = -400;
    f.listeners.get('focusin')({ target: row });
    assert.equal(f.scroller.scrollTop, -232, 'the moved row clears the top fade');
    // Another row reached by the keyboard is revealed as before.
    f.scroller.scrollTop = 200; top = 20;
    f.listeners.get('focusout')({ target: other });
    f.listeners.get('focusin')({ target: row });
    assert.equal(f.scroller.scrollTop, 188);
});
