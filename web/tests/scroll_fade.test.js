import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { bindScrollFade, replaceScrollFade, revealPastFade } from '../modules/scroll_fade.js';
import { ROOM_EDGE_HTML } from '../modules/page_header.js';

// A room's feed carries no fade (DESIGN principle 9, owner 21.10): what darkens content under its glass is the
// bar's and the composer's own drawing. A mask on the feed would also make it one stacking context and drop the
// lazy-load margin of the photos inside it.
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

test("a room's feed carries no fade or room-top switch", () => {
    for (const sheet of ['style.css', 'ui.css']) {
        assert.deepEqual(roomFeedFades(readFileSync(new URL(`../${sheet}`, import.meta.url), 'utf8')), []);
    }
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
        constructor(callback) {
            this.callback = callback; this.disconnected = false; observers.push(this);
            this.targets = new Set(); this.observeCalls = []; this.unobserveCalls = []; this.disconnects = 0;
        }
        observe(target) { this.targets.add(target); this.observeCalls.push(target); this.disconnected = false; }
        unobserve(target) { this.targets.delete(target); this.unobserveCalls.push(target); }
        disconnect() { this.targets.clear(); this.disconnects += 1; this.disconnected = true; }
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

function readStyle(scroller, tokens = {}, computed = {}) {
    scroller.ownerDocument.defaultView.getComputedStyle = () => ({
        ...computed, getPropertyValue: name => tokens[name] || '',
    });
}

test('a revealed row clears the sheet’s current fade, including a row already inside the viewport', () => {
    const { scroller } = fixture();
    const tokens = { '--scroll-fade-edge': '32px' };
    readStyle(scroller, tokens);
    scroller.getBoundingClientRect = () => ({ top: 20, bottom: 120 });
    const row = { getBoundingClientRect: () => ({ top: 80, bottom: 110 }) };
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 22, 'the last ten pixels were inside the box but under the fade');
    tokens['--scroll-fade-edge'] = '40px'; scroller.scrollTop = 0;
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 30, 'a different CSS token changes the clearance without a JS constant');
    scroller.scrollTop = 400; // settled bottom: the same row has no bottom fade to clear
    revealPastFade(scroller, row);
    assert.equal(scroller.scrollTop, 400);
});

test('a two-axis reveal clears the physical RTL edge and keeps padding inside the compact fade', () => {
    const { scroller } = fixture();
    Object.assign(scroller, { scrollLeft: 0, scrollTop: 80, clientWidth: 60, clientHeight: 60 });
    readStyle(scroller, { '--scroll-fade-edge': '32px', '--scroll-fade-inline-edge': '16px',
        '--scroll-fade-top-clear': '12px', '--scroll-fade-limit': '.25' },
    { direction: 'rtl', overflowX: 'auto', overflowY: 'auto', scrollPaddingRight: '16px', scrollPaddingLeft: '8px' });
    scroller.getBoundingClientRect = () => ({ left: 0, top: 0 });
    const row = { getBoundingClientRect: () => ({ left: 10, right: 40, top: 20, bottom: 40 }) };
    revealPastFade(scroller, row, { axis: 'both' });
    assert.equal(scroller.scrollLeft, -13, 'at RTL start the left edge needs 15px of fade and 8px of clear padding');
    assert.equal(scroller.scrollTop, 76, '12px reserved top + a quarter of the 48px reading box');
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

test('both axes are independent; overflow hidden and an RTL rubber band do not invent edge travel', () => {
    const f = fixture();
    const computed = { direction: 'rtl', overflowX: 'auto', overflowY: 'auto' };
    readStyle(f.scroller, {}, computed);
    const state = () => [...f.attrs].sort();
    const dispose = bindScrollFade(f.scroller, { axis: 'both' });
    f.flush();
    assert.deepEqual(state(), ['data-scroll-below', 'data-scroll-end', 'data-scroll-rtl']);
    Object.assign(f.scroller, { scrollLeft: -400, scrollTop: 200 });
    f.listeners.get('scroll')(); f.flush();
    assert.deepEqual(state(), ['data-scroll-above', 'data-scroll-below', 'data-scroll-rtl', 'data-scroll-start']);
    f.scroller.scrollLeft = 6; // Safari can report a positive overscroll at RTL start.
    computed.overflowY = 'hidden';
    f.observers[0].callback(); f.flush();
    assert.deepEqual(state(), ['data-scroll-end', 'data-scroll-rtl']);
    dispose();
    assert.deepEqual(state(), [], 'no disposed mask follows a remount');
});

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
    assert.deepEqual([...named.keys()], [], 'disposal releases the optional region too');
    named.set('tabindex', '-1');
    named.set('role', 'listbox');
    const disposeAuthored = bindScrollFade(f.scroller, { region: 'Not an override' });
    f.flush();
    f.scroller.scrollHeight = f.scroller.clientHeight;
    f.observers.at(-2).callback(); f.flush();
    disposeAuthored();
    assert.deepEqual(Object.fromEntries(named), { tabindex: '-1', role: 'listbox' });
    const plain = fixture();  // its stub has no attribute API: a body without a name is only decorated
    assert.doesNotThrow(() => { bindScrollFade(plain.scroller); plain.flush(); });
});

test('keyboard focus clears the fade, while pointer focus and focus handed back after a pointer leave the list still', () => {
    const f = fixture();
    readStyle(f.scroller, { '--scroll-fade-edge': '32px' });
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
    readStyle(f.scroller, { '--scroll-fade-edge': '32px' });
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

test('appended batches observe only new direct children; removal and disposal release their registrations', () => {
    const f = fixture();
    const children = f.scroller.children;
    const dispose = bindScrollFade(f.scroller);
    const resize = f.observers[0], mutations = f.observers[1];
    assert.deepEqual(resize.observeCalls, [f.scroller], 'the scroller itself is observed once');
    const batch = (count) => {
        const added = Array.from({ length: count }, () => ({ nodeType: 1, parentNode: f.scroller }));
        children.push(...added);
        mutations.callback([{ type: 'childList', target: f.scroller, addedNodes: added, removedNodes: [] }]);
        return added;
    };
    const first = batch(20);
    assert.equal(resize.observeCalls.length, 21, 'a 20-row append registers 20 new children, not the whole list again');
    const second = batch(20);
    assert.equal(resize.observeCalls.length, 41, 'the next append leaves the first 20 registrations alone');
    assert.equal(resize.targets.size, 41);
    children.splice(0, 20);
    for (const child of first) child.parentNode = null;
    mutations.callback([{ type: 'childList', target: f.scroller, addedNodes: [], removedNodes: first }]);
    assert.deepEqual(resize.unobserveCalls, first, 'removed rows are unobserved individually');
    assert.equal(resize.targets.size, 21);
    const moved = second[0];
    children.splice(0, 1); children.push(moved);
    mutations.callback([{ type: 'childList', target: f.scroller, addedNodes: [moved], removedNodes: [moved] }]);
    assert.equal(resize.observeCalls.length, 41, 'a row moved within the same scroller keeps its registration');
    assert.equal(resize.unobserveCalls.length, 20);
    dispose();
    assert.equal(resize.disconnected, true);
    assert.equal(resize.targets.size, 0);
    assert.equal(mutations.disconnected, true);
});

test('a mixed mutation batch releases exactly the rows that ended outside the scroller and never re-registers one that only moved', () => {
    const f = fixture();
    const row = (parentNode = f.scroller) => ({ nodeType: 1, parentNode });
    const elsewhere = { nodeType: 1 };
    const [kept, moved, left] = [row(), row(), row()];
    f.scroller.children.push(kept, moved, left);
    bindScrollFade(f.scroller);
    const resize = f.observers[0], mutations = f.observers[1];
    assert.deepEqual(resize.observeCalls, [f.scroller, kept, moved, left]);
    // One delivered batch, in the order a script made the writes: `moved` leaves and comes back; `left` leaves, comes
    // back and leaves again, for good; `late` arrives and is taken away; `fresh` arrives; text and deeper nodes are not rows.
    const [late, fresh] = [row(), row()];
    left.parentNode = elsewhere; late.parentNode = null;
    const at = (addedNodes, removedNodes, target = f.scroller) => ({ type: 'childList', target, addedNodes, removedNodes });
    mutations.callback([
        at([], [moved]), at([moved], []), at([], [left]), at([left], []), at([], [left]),
        at([late], []), at([], [late]), at([fresh], []), at([{ nodeType: 3 }], []), at([row()], [], elsewhere),
    ]);
    assert.deepEqual(resize.unobserveCalls, [left], 'the row that ended elsewhere is released once, though it was also added in the batch');
    assert.deepEqual(resize.observeCalls, [f.scroller, kept, moved, left, fresh], 'a row that moved or came and went is never re-registered');
    assert.deepEqual([...resize.targets], [f.scroller, kept, moved, fresh]);
    // The same row, taken back into the scroller by a later batch, is registered again.
    left.parentNode = f.scroller;
    mutations.callback([at([left], [])]);
    assert.equal(resize.targets.has(left), true);
    assert.equal(resize.observeCalls.at(-1), left);
});

// Files swaps the preview's binding with every file shown, so a release must reach whatever is live on the element,
// not a disposer handed out earlier. The shell's pagehide over that swap — a kept page keeps the live binding, a page
// really going releases it — is asserted against the real Files page in tests/test_ui_smoke_files_project_drafts.py.
test('a release reaches the binding live on a rebound element, not the spent disposer', () => {
    const f = fixture();
    const first = bindScrollFade(f.scroller);
    first();  // Files' setPreview gave up the binding it had
    bindScrollFade(f.scroller, { region: 'File contents' });
    const live = f.observers.slice(2);
    assert.equal(live.length, 2, 'one ResizeObserver and one MutationObserver follow the swapped-in binding');
    replaceScrollFade(f.scroller);  // the shell's pagehide releases what is live on each
    assert.equal(live.every((o) => o.disconnected), true, 'the live binding is released, not the spent one');
    assert.equal(f.listeners.size, 0);
    assert.equal(f.frames.size, 0);
});

// One edge for everything that scrolls (DESIGN "One edge", owner decision 27.11.2026): one curve (--edge-curve) and
// two blur steps (--edge-blur-soft, --edge-blur-deep) in ui.css's :root, read by the shared scroller masks, a room's
// edge and every painted fade. A sheet may say where and how long an edge is, never draw one of its own.
const SHEETS = readdirSync(new URL('..', import.meta.url)).filter((name) => name.endsWith('.css'));
const SHARED_EDGE_IMAGES = new Set(['--scroll-fade-y-image', '--scroll-fade-x-image']);
function gradientsIn(value) {
    const found = [];
    for (const match of value.matchAll(/(?:repeating-)?(?:linear|radial|conic)-gradient\(/g)) {
        let depth = 1, at = match.index + match[0].length;
        while (depth && at < value.length) depth += { '(': 1, ')': -1 }[value[at++]] || 0;
        found.push(value.slice(match.index + match[0].length, at - 1).trim());
    }
    return found;
}
// A colour that is not fully opaque: `transparent`, or an alpha below 1 in any notation (0, 0.5, .5, 50%).
const alphaBelowOne = (alpha) => (alpha.endsWith('%') ? parseFloat(alpha) < 100 : parseFloat(alpha) < 1);
function fades(stops) {
    if (/\btransparent\b/.test(stops)) return true;
    for (const [, args] of stops.matchAll(/(?:rgba?|hsla?)\(([^()]*(?:\([^()]*\)[^()]*)*)\)/g)) {
        const slash = args.split('/');
        const alpha = slash.length > 1 ? slash.at(-1).trim() : (args.split(',').length === 4 || /^var\(/.test(args.trim())
            && args.split(',').length === 2 ? args.split(',').at(-1).trim() : '');
        if (alpha && alphaBelowOne(alpha)) return true;
    }
    return false;
}
// The style rules of a sheet: each selector list with its declarations, values with whitespace collapsed.
function rulesOf(css) {
    css = css.replace(/\/\*[\s\S]*?\*\//g, '');
    return [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, selectors, body]) => ({
        rule: selectors.trim().replace(/\s+/g, ' '),
        decls: body.split(';').map((d) => d.split(/:(.*)/s)).filter(([p, v]) => p && v !== undefined)
            .map(([p, v]) => [p.trim(), v.trim().replace(/\s+/g, ' ')]),
    }));
}
// The classes and ids a rule's subjects carry (the last compound of each selector), so that rules styling the same
// element are seen together however the sheet splits them.
const subjectKeys = (rule) => new Set(topLevel(rule, /,/).flatMap((one) => {
    const last = topLevel(one, /[\s>+~]/).at(-1) || '';
    return last.replace(/:not\([^)]*\)/g, '').match(/[.#][\w-]+/g) || [];
}));
function edgeRecipes(css, sheet) {
    const rules = rulesOf(css), found = [];
    const isMask = (prop) => /^(-webkit-)?mask(-image)?$/.test(prop);
    const masked = new Set(rules.filter(({ decls }) => decls.some(([p, v]) => isMask(p) && v !== 'none'))
        .flatMap(({ rule }) => [...subjectKeys(rule)]));
    for (const { rule, decls } of rules) {
        for (const [prop, value] of decls) {
            if (/^--edge-(curve|blur-)/.test(prop) && !(sheet === 'ui.css' && rule === ':root')) found.push(`${rule} defines ${prop}`);
            if (prop.startsWith('--') && !SHARED_EDGE_IMAGES.has(prop) && gradientsIn(value).some(fades)) found.push(`${rule} ${prop}`);
            if (SHARED_EDGE_IMAGES.has(prop) && sheet !== 'ui.css') found.push(`${rule} redefines ${prop}`);
            if (isMask(prop)) {
                const own = gradientsIn(value).filter((stops) => !/^(to [a-z ]+, |var\(--edge-to\), )?var\(--edge-curve\)$/.test(stops)
                    && !/^#000, ?#000$/.test(stops));
                if (own.length) found.push(`${rule} ${prop}`);
            }
            if (/^background(-image)?$/.test(prop) && gradientsIn(value).some((stops) => fades(stops)
                && new Set(stops.split(/,(?![^(]*\))/).map((part) => part.trim())).size > 1)) found.push(`${rule} ${prop}`);
            // A blur on anything a sheet masks is an edge: it reads the edge's tokens, in whichever rule it stands.
            if (/^(-webkit-)?backdrop-filter$/.test(prop) && value !== 'none' && !/^var\(--edge-blur-(soft|deep)\)$/.test(value)
                && [...subjectKeys(rule)].some((key) => masked.has(key))) found.push(`${rule} ${prop}`);
        }
    }
    return found;
}

test('every edge in every sheet is the one shared edge', () => {
    for (const sheet of SHEETS) {
        assert.deepEqual(edgeRecipes(readFileSync(new URL(`../${sheet}`, import.meta.url), 'utf8'), sheet), [], sheet);
    }
    // The recipes this replaced, each its own edge: the room's bar, the folded answer, a painted list fade, a private
    // blur radius or curve.
    for (const broken of [
        `.chat-page-header::before { background: linear-gradient(180deg, rgba(var(--shell-rgb), 0.72) 0, rgba(var(--shell-rgb), 0.00) 100%);
            backdrop-filter: blur(12px); mask-image: linear-gradient(180deg, var(--edge-mask)); }`,
        '.project-answer.is-folded > .message { mask-image: linear-gradient(180deg, black 60%, transparent 100%); }',
        '.sidebar-head::after { background: linear-gradient(var(--surface-side), transparent); }',
        '.toast-edge { -webkit-backdrop-filter: blur(14px); -webkit-mask-image: linear-gradient(var(--edge-curve)); }',
        '.page { --edge-curve: #000, transparent; }',
        '.list { --list-fade: linear-gradient(black 80%, transparent); }',
        // The room edge's own split: the mask in one rule, a private radius in another rule for the same element.
        '.fake-edge { mask-image: linear-gradient(var(--edge-curve)); } .fake-edge[data-b] { backdrop-filter: blur(9px); }',
        // A painted fade whose clear end is written with an integer alpha.
        '.fake-fade { background: linear-gradient(rgba(var(--shell-rgb), 1), rgba(var(--shell-rgb), 0)); }',
        '.fake-fade { background: linear-gradient(rgb(0 0 0 / 1), rgb(0 0 0 / 0%)); }',
    ]) assert.ok(edgeRecipes(broken, 'style.css').length, broken);
    const allowed = `.room-edge { mask-image: linear-gradient(var(--edge-to), var(--edge-curve)), linear-gradient(#000, #000); }
        .room-edge[data-edge="deep"] { backdrop-filter: var(--edge-blur-deep); mask-image: linear-gradient(to top, var(--edge-curve)); }
        .icon { mask-image: url("data:image/svg+xml,x"); }
        .chip { background: linear-gradient(var(--surface-capsule), var(--surface-capsule)), rgba(var(--shell-rgb), 0.98); }
        .glass { backdrop-filter: var(--glass); }
        .bar-meter { background: linear-gradient(to right, var(--accent) 40%, var(--neutral) 40%); }
        .list.scroll-fade-y[data-scroll-below] { --scroll-fade-edge: 24px; }
        .opaque { background: linear-gradient(rgba(var(--shell-rgb), 1), rgba(var(--shell-rgb), 1)); }
        .room-edge[data-edge="soft"] { backdrop-filter: var(--edge-blur-soft); }
        .toast { backdrop-filter: blur(14px); } .toast-words { mask-image: linear-gradient(var(--edge-curve)); }`;
    assert.deepEqual(edgeRecipes(allowed, 'style.css'), [], 'what places an edge, or paints no fade, is not a recipe');
});

test('the shared scroller masks and the curve are one curve, and a room edge is two blur steps under its tint', () => {
    const ui = readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    const alphas = (text) => [...text.matchAll(/rgb\(0 0 0 \/ (\.\d+)\)/g)].map((m) => Number(m[1]));
    const curve = alphas(ui.match(/--edge-curve:([^;]+);/)[1]);
    assert.equal(curve.length, 9);
    curve.forEach((alpha, i) => {
        const t = 1 - (i + 1) / 10;
        assert.ok(Math.abs(alpha - t * t * (3 - 2 * t)) < 0.001, 'smoothstep: flat at both ends, so neither draws a line');
    });
    for (const image of ['--scroll-fade-y-image', '--scroll-fade-x-image']) {
        const stops = alphas(ui.match(new RegExp(`${image}:([^;]+);`))[1]);
        assert.deepEqual(stops, [...[...curve].reverse(), ...curve], `${image} lays the curve over both ends`);
    }
    // A room's edge, read from the parsed sheet (whitespace and line breaks do not matter).
    const rules = rulesOf(readFileSync(new URL('../style.css', import.meta.url), 'utf8'));
    const decl = (rule, prop) => rules.filter((r) => r.rule === rule).flatMap((r) => r.decls).find(([p]) => p === prop)?.[1];
    assert.equal(decl('.room-edge', 'mask-image'), 'linear-gradient(var(--edge-to), var(--edge-curve)), linear-gradient(#000, #000)');
    assert.equal(decl('.room-edge', '-webkit-mask-image'), decl('.room-edge', 'mask-image'));
    assert.equal(decl('.room-edge[data-edge="soft"]', 'backdrop-filter'), 'var(--edge-blur-soft)');
    assert.equal(decl('.room-edge[data-edge="deep"]', 'backdrop-filter'), 'var(--edge-blur-deep)');
    const blurring = rules.filter(({ rule, decls }) => subjectKeys(rule).has('.room-edge')
        && decls.some(([p, v]) => p === 'backdrop-filter' && v !== 'none'));
    assert.equal(blurring.length, 2, 'two blur steps, the soft and the deep');
    assert.equal(ROOM_EDGE_HTML.match(/data-edge="(\w+)"/g).join(' '), 'data-edge="soft" data-edge="deep" data-edge="tint"',
        'the tint is drawn last, over the blurred feed');
    const barEdge = rules.filter(({ rule, decls }) => /(chat-page-header|project-panel-bar|chat-input-area)[^,]*::before/.test(rule)
        && decls.some(([p, v]) => (/backdrop-filter$/.test(p) || /mask-image$/.test(p)) && v !== 'none'));
    assert.deepEqual(barEdge.map(({ rule }) => rule), [], 'the bars draw no edge of their own beside it');
});
