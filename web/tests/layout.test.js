import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

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

test('measurePinnedBars writes the pinned head and foot heights for the rows\' scroll margin', async () => {
    const { measurePinnedBars } = await import('../modules/layout.js');
    const props = new Map();
    const box = (height) => ({ getBoundingClientRect: () => ({ height }) });
    const head = box(104), foot = box(60.5);
    const sidebar = { style: { setProperty: (name, value) => props.set(name, value) },
        querySelector: (sel) => ({ '.sidebar-head': head, '.sidebar-foot-meta': foot })[sel] ?? null };
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
    assert.deepEqual(observed, [[head, 'border-box'], [foot, 'border-box']]);
    assert.equal(props.get('--sidebar-head-h'), '104px');
    assert.equal(props.get('--sidebar-foot-h'), '60.5px');
    foot.getBoundingClientRect = () => ({ height: 88 });  // an update capsule wrapped onto a second line
    fire();
    assert.equal(props.get('--sidebar-foot-h'), '88px');
    dispose();
    assert.equal(disconnected, true);
    // A page without the bars, or without ResizeObserver, measures nothing and still returns a disposer.
    assert.equal(typeof measurePinnedBars({ querySelector: () => null }, { Observer }), 'function');
    assert.equal(typeof measurePinnedBars(sidebar, { Observer: undefined }), 'function');
});
