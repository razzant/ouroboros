import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { bindScrollFade } from '../modules/scroll_fade.js';

// The shell's one layout switch: theme.js writes html[data-layout] before CSS from one number pair and keeps
// it current; layout.js is the door every module reads it through (docs/DESIGN.md "Shell", P1).

const themeSource = fs.readFileSync(new URL('../theme.js', import.meta.url), 'utf8');

function bootTheme({ docked = true, media = true } = {}) {
    const handlers = new Map();
    const events = [];
    const root = { dataset: {} };
    const noop = () => {};
    const colorQuery = { matches: false, addEventListener: noop, removeEventListener: noop };
    const layoutQuery = {
        matches: docked,
        addEventListener: (key, fn) => handlers.set(`layout:${key}`, fn),
        removeEventListener: (key) => handlers.delete(`layout:${key}`),
    };
    const window = {
        addEventListener: noop, removeEventListener: noop,
        dispatchEvent: (event) => events.push(event),
        ...(media ? { matchMedia: (q) => (q.includes('min-width') ? layoutQuery : colorQuery) } : {}),
    };
    const document = {
        documentElement: root, createElement: () => ({ dataset: {}, classList: { toggle: noop }, setAttribute: noop }),
        querySelectorAll: () => [], addEventListener: noop, removeEventListener: noop,
    };
    vm.runInNewContext(themeSource, {
        document, window, localStorage: { getItem: () => null, setItem: noop },
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
    });
    return { root, window, layoutQuery, handlers, events };
}

test('theme.js paints the layout before CSS and publishes its number once', () => {
    const wide = bootTheme({ docked: true });
    assert.equal(wide.root.dataset.layout, 'docked');
    assert.equal(wide.window.ouroLayout.dockedMinWidth, 720);
    // The Project panel keeps its own width beside Main (owner 18.10, 4A): no fill threshold.
    assert.equal('projectBesideMainMinWidth' in wide.window.ouroLayout, false);
    assert.equal(wide.window.ouroLayout.layout, 'docked');
    assert.ok(Object.isFrozen(wide.window.ouroLayout), 'the facts are read-only');
    assert.equal(bootTheme({ docked: false }).root.dataset.layout, 'stack');
    // No media queries at all (an old shell): a window, so docked.
    assert.equal(bootTheme({ media: false }).root.dataset.layout, 'docked');
});

test('a resize or rotation across the switch repaints and announces once', () => {
    const s = bootTheme({ docked: true });
    s.layoutQuery.matches = false;
    s.handlers.get('layout:change')();
    assert.equal(s.root.dataset.layout, 'stack');
    assert.equal(s.window.ouroLayout.layout, 'stack');
    assert.deepEqual(s.events.map((e) => [e.type, e.detail.layout]), [['ouro:layout-changed', 'stack']]);
    // The same answer again is not a change.
    s.handlers.get('layout:change')();
    assert.equal(s.events.length, 1);
});

test('layout.js reads the painted attribute, falls back to the facts, and hears changes', async () => {
    const mod = await import('../modules/layout.js');
    const w = { ouroLayout: { dockedMinWidth: 720, layout: 'stack' }, listeners: new Map(),
        addEventListener(type, fn) { this.listeners.set(type, fn); }, removeEventListener(type) { this.listeners.delete(type); } };
    assert.equal(mod.currentLayout({ root: { dataset: { layout: 'docked' } }, w }), 'docked');
    assert.equal(mod.currentLayout({ root: { dataset: {} }, w }), 'stack', 'no attribute yet: the facts answer');
    assert.equal(mod.currentLayout({ root: { dataset: {} }, w: {} }), 'docked', 'no facts either: a window');
    assert.equal(mod.isStackLayout({ root: { dataset: { layout: 'stack' } }, w }), true);
    assert.throws(() => mod.layoutFacts({}), /theme\.js/);
    const seen = [];
    const dispose = mod.onLayoutChange((layout) => seen.push(layout), { w });
    w.listeners.get('ouro:layout-changed')({ detail: { layout: 'docked' } });
    assert.deepEqual(seen, ['docked']);
    dispose();
    assert.equal(w.listeners.size, 0, 'the subscription has a disposer');
});

test('the Project panel has no fill rule: it keeps its own width beside Main', async () => {
    const mod = await import('../modules/layout.js');
    assert.equal('projectPanelFillsContent' in mod, false);
});

test('a pinch is read against the layout width, so a smaller Interface size is no pinch', async () => {
    const { pinchMagnified } = await import('../modules/layout.js');
    const at = ({ inner, client, visual }) => ({ innerWidth: inner, visualViewport: { width: visual },
        document: { documentElement: { clientWidth: client } } });
    // Measured in mobile Chromium, 390 px wide: 80 % lays the page out at 488 px, innerWidth reads 489.
    assert.equal(pinchMagnified(at({ inner: 489, client: 488, visual: 487.5 })), false);
    assert.equal(pinchMagnified(at({ inner: 390, client: 390, visual: 195 })), true, 'a pinch to 2x');
    // iOS WebKit's innerWidth is the zoomed visible width (measured: 201 of a 402 px layout at a pinch to 2x).
    assert.equal(pinchMagnified(at({ inner: 201, client: 402, visual: 201 })), true, 'a pinch in iOS WebKit');
    assert.equal(pinchMagnified(at({ inner: 700, client: 685, visual: 685 })), false, 'a classic scrollbar is no pinch');
    assert.equal(pinchMagnified({ innerWidth: 390 }), false, 'no visual viewport, no pinch');
});

test('measurePinnedBars writes the pinned head and foot heights, and says when they leave no room for a row', async () => {
    const { measurePinnedBars } = await import('../modules/layout.js');
    const props = new Map();
    const box = (height) => ({ getBoundingClientRect: () => ({ height }) });
    const head = box(104), foot = box(60.5);
    const row = box(48);
    const attrs = new Set();
    // The foot is the whole bar a short window pins: the six sections, the version and the update.
    const sidebar = { style: { setProperty: (name, value) => props.set(name, value) }, clientHeight: 500,
        ownerDocument: { defaultView: { getComputedStyle: () => ({ getPropertyValue: name => name === '--space-4' ? '16px' : '' }) } },
        toggleAttribute: (name, on) => { if (on) attrs.add(name); else attrs.delete(name); },
        querySelector: (sel) => ({ '.sidebar-head': head, '.sidebar-foot': foot, '.nav-project-row': row })[sel] ?? null };
    const observed = [];
    let fire = null;
    let disconnected = false;
    class Observer {
        constructor(callback) { fire = callback; }
        observe(el, options) { observed.push([el, options?.box]); }
        disconnect() { disconnected = true; }
    }
    const dispose = measurePinnedBars(sidebar, { Observer });
    // The border box: a band or a safe area that arrives later changes only the bars' padding.
    // The column too: a window that grows or shrinks changes whether a row still fits between the bars.
    assert.deepEqual(observed, [[head, 'border-box'], [foot, 'border-box'], [sidebar, 'border-box']]);
    assert.equal(props.get('--sidebar-head-h'), '104px');
    assert.equal(props.get('--sidebar-foot-h'), '60.5px');
    assert.equal(attrs.has('data-bars-crowd'), false, 'head, a row and foot fit in 500 px');
    foot.getBoundingClientRect = () => ({ height: 188 });  // the sections two to a line, the version under them
    fire();
    assert.equal(props.get('--sidebar-foot-h'), '188px');
    assert.equal(attrs.has('data-bars-crowd'), false, '104 + 48 + 188 = 340 still fit');
    sidebar.clientHeight = 330;  // a shorter window: no Project row between the two bars
    fire();
    assert.equal(attrs.has('data-bars-crowd'), true, 'the head scrolls away with the list there');
    sidebar.clientHeight = 500;
    fire();
    assert.equal(attrs.has('data-bars-crowd'), false, 'and is pinned again when the room returns');
    head.getBoundingClientRect = () => ({ height: 106 });
    foot.getBoundingClientRect = () => ({ height: 152 });
    for (const [height, crowded] of [[340, false], [320, true], [306, true], [300, true], [340, false]]) {
        sidebar.clientHeight = height;
        fire();
        assert.equal(attrs.has('data-bars-crowd'), crowded, `${height}px includes the 16px scroll clearance at both bars`);
    }
    dispose();
    assert.equal(disconnected, true);
    // A page without the bars, or without ResizeObserver, measures nothing and still returns a disposer.
    assert.equal(typeof measurePinnedBars({ querySelector: () => null }, { Observer }), 'function');
    assert.equal(typeof measurePinnedBars(sidebar, { Observer: undefined }), 'function');
});

test('app startup binds the column fade to hidden rows, and clears it at settled edges', () => {
    const source = fs.readFileSync(new URL('../app.js', import.meta.url), 'utf8');
    const start = source.indexOf('const primarySidebar =');
    const end = source.indexOf('const navProjectsHeader =', start);
    assert.ok(start >= 0 && end > start, 'execute the sidebar startup including its real fade binding');
    const attrs = new Set(), listeners = new Map(), frames = [];
    class Observer { observe() {} disconnect() {} }
    const sidebar = { scrollTop: 0, clientHeight: 320, scrollHeight: 1000, children: [],
        ownerDocument: { defaultView: { ResizeObserver: Observer, MutationObserver: Observer,
            requestAnimationFrame: fn => frames.push(fn) } },
        querySelector: () => null,
        toggleAttribute: (name, on) => on ? attrs.add(name) : attrs.delete(name),
        addEventListener: (name, fn) => listeners.set(name, fn) };
    vm.runInNewContext(source.slice(start, end), {
        document: { getElementById: id => id === 'primary-sidebar' ? sidebar : null },
        createShellMotion: () => ({}), isStackLayout: () => false,
        renderNavRows: () => '', measurePinnedBars: () => {}, bindScrollFade,
    });
    const check = (top, height, expected) => {
        sidebar.scrollTop = top; sidebar.clientHeight = height;
        listeners.get('scroll')?.();
        while (frames.length) frames.shift()();
        assert.deepEqual([attrs.has('data-scroll-above'), attrs.has('data-scroll-below')], expected);
    };
    check(0, 320, [false, true]);
    check(300, 320, [true, true]);
    check(680, 320, [true, false]);
    check(0, 1000, [false, false]);
});

// The primary input, the switch's one other fact (owner 06.11; docs/DESIGN.md "Shell"): theme.js paints
// html[data-pointer] beside html[data-layout] before CSS from one `(pointer: coarse)` query; layout.js is the door.
function bootPointer({ docked = true, coarse = false, media = true } = {}) {
    const handlers = new Map();
    const events = [];
    const root = { dataset: {} };
    const noop = () => {};
    const query = (matches, key) => ({ matches, addEventListener: (type, fn) => handlers.set(`${key}:${type}`, fn),
        removeEventListener: (type) => handlers.delete(`${key}:${type}`) });
    const layoutQuery = query(docked, 'layout');
    const pointerQuery = query(coarse, 'pointer');
    const colorQuery = query(false, 'color');
    const window = {
        addEventListener: noop, removeEventListener: noop,
        dispatchEvent: (event) => events.push(event),
        ...(media ? { matchMedia: (q) => (q.includes('min-width') ? layoutQuery : q.includes('pointer') ? pointerQuery : colorQuery) } : {}),
    };
    const document = {
        documentElement: root, createElement: () => ({ dataset: {}, classList: { toggle: noop }, setAttribute: noop }),
        querySelectorAll: () => [], addEventListener: noop, removeEventListener: noop,
    };
    vm.runInNewContext(themeSource, {
        document, window, localStorage: { getItem: () => null, setItem: noop },
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
    });
    return { root, window, handlers, events, layoutQuery, pointerQuery };
}

test('theme.js paints the primary input beside the layout and publishes it', () => {
    const finger = bootPointer({ docked: true, coarse: true });
    assert.equal(finger.root.dataset.layout, 'docked');
    assert.equal(finger.root.dataset.pointer, 'coarse', 'a tablet: docked columns, a finger');
    assert.equal(finger.window.ouroLayout.pointer, 'coarse');
    assert.equal(bootPointer({ docked: true, coarse: false }).root.dataset.pointer, 'fine', 'a desk with a mouse');
    assert.equal(bootPointer({ docked: false, coarse: true }).root.dataset.layout, 'stack', 'the input never moves the columns');
    assert.equal(bootPointer({ media: false }).root.dataset.pointer, 'fine', 'no media queries: a desktop window');
});

test('a new primary input repaints and announces once, on the one layout event', () => {
    const s = bootPointer({ docked: true, coarse: false });
    s.pointerQuery.matches = true;  // a Surface's cover came off
    s.handlers.get('pointer:change')();
    assert.equal(s.root.dataset.pointer, 'coarse');
    assert.equal(s.root.dataset.layout, 'docked');
    s.handlers.get('pointer:change')();  // the same answer again is no change
    assert.deepEqual(s.events.map((e) => [e.type, e.detail.layout, e.detail.pointer]), [['ouro:layout-changed', 'docked', 'coarse']]);
});

test('layout.js reads the primary input through one door, and a covering room from it', async () => {
    const mod = await import('../modules/layout.js');
    const w = { ouroLayout: { dockedMinWidth: 720, layout: 'docked', pointer: 'coarse' } };
    assert.deepEqual([...mod.POINTERS], ['fine', 'coarse']);
    assert.equal(mod.currentPointer({ root: { dataset: { pointer: 'fine' } }, w }), 'fine');
    assert.equal(mod.currentPointer({ root: { dataset: {} }, w }), 'coarse', 'no attribute yet: the facts answer');
    assert.equal(mod.currentPointer({ root: { dataset: {} }, w: {} }), 'fine', 'neither: a desktop window');
    assert.equal(mod.isCoarsePointer({ root: { dataset: { pointer: 'coarse' } }, w }), true);
    const room = (layout, pointer) => mod.roomCoversMain({ root: { dataset: { layout, pointer } }, w: {} });
    assert.equal(room('docked', 'fine'), false, 'a desk: the panel lies beside Main');
    assert.equal(room('docked', 'coarse'), true, 'a finger: the Project in Main\'s place');
    assert.equal(room('stack', 'fine'), true, 'a phone-width window: the room fills the screen');
});

test('a section holds its list beside the page from a 40rem box, and only where the engine draws that', async () => {
    const { sectionHoldsTwoPanes } = await import('../modules/layout.js');
    const page = (width) => ({ getBoundingClientRect: () => ({ width }) });
    // An engine: whether it has container queries, and its root font size (the Interface size).
    const engine = (queries, fontSize = '16px') => ({
        CSS: { supports: (property, value) => queries && property === 'container-type' && value === 'inline-size' },
        getComputedStyle: () => ({ fontSize }),
    });
    const root = {};
    assert.equal(sectionHoldsTwoPanes(page(640), { w: engine(true), root }), true);
    assert.equal(sectionHoldsTwoPanes(page(639.9), { w: engine(true), root }), false, 'under 40rem: one column');
    assert.equal(sectionHoldsTwoPanes(page(700), { w: engine(true, '20px'), root }), false, '40rem is 800 px at a 20 px root');
    // The sheets draw two panes inside `@container sectionplace`; without container queries that rule never applies,
    // so a module that assumed two panes would hide Back beside a list nobody sees.
    assert.equal(sectionHoldsTwoPanes(page(1200), { w: engine(false), root }), false, 'no container queries: one column at every width');
    assert.equal(sectionHoldsTwoPanes(page(1200), { w: {}, root }), true, 'a document that cannot say is a current engine');
    assert.equal(sectionHoldsTwoPanes({}, { w: engine(true), root }), false, 'a box nobody can measure is one column');
});
