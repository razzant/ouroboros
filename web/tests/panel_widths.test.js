import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { SIDEBAR_MIN_FALLBACK, setupResizablePanels, sidebarMinWidth } from '../modules/panel_widths.js';

// The sidebar's resize handle and its stored width (panel_widths.js): below --sidebar-min-width (ui.css) Ouroboros's
// card, the search row and the sections at the foot no longer hold whole, so a drag stops there and a narrower width
// stored before is raised when the page loads; a width at or above it is kept as stored.

const MIN = 270;

function fakeShell({ min = `${MIN}px`, sidebarWidth = 300 } = {}) {
    const vars = new Map();
    const style = (map) => ({ setProperty: (k, v) => map.set(k, v), removeProperty: (k) => map.delete(k), getPropertyValue: (k) => map.get(k) || '' });
    const root = { dataset: { layout: 'docked' }, style: style(vars) };
    const element = (width) => {
        const own = new Map();
        const handles = [];
        return {
            own, handles, width,
            style: style(own),
            getBoundingClientRect() { return { width: Number.parseFloat(own.get('--drag-width')) || this.width }; },
            appendChild(child) { handles.push(child); },
        };
    };
    const sidebar = element(sidebarWidth);
    const panel = element(400);
    const posts = [];
    const previous = { document: globalThis.document, getComputedStyle: globalThis.getComputedStyle,
        requestAnimationFrame: globalThis.requestAnimationFrame, cancelAnimationFrame: globalThis.cancelAnimationFrame, fetch: globalThis.fetch };
    globalThis.document = {
        documentElement: root,
        body: { classList: { add() {}, remove() {} } },
        getElementById: (id) => (id === 'primary-sidebar' ? sidebar : id === 'project-panel' ? panel : null),
        createElement: () => {
            const listeners = {};
            return { listeners, addEventListener: (k, fn) => { listeners[k] = fn; }, setPointerCapture() {}, releasePointerCapture() {} };
        },
    };
    globalThis.getComputedStyle = (node) => ({ getPropertyValue: (k) => (node === root && k === '--sidebar-min-width' ? min : '') });
    globalThis.requestAnimationFrame = (fn) => { fn(); return 0; };
    globalThis.cancelAnimationFrame = () => {};
    globalThis.fetch = async (url, init) => { posts.push(JSON.parse(init.body)); return { ok: true }; };
    const restore = () => Object.assign(globalThis, previous);
    return { root, vars, sidebar, posts, restore };
}

const settle = () => new Promise((ok) => setTimeout(ok, 450));  // the stored width is posted after 400 ms

test('the minimum is the stylesheet one, and the stylesheet draws the sidebar no narrower', () => {
    const shell = fakeShell();
    try {
        assert.equal(sidebarMinWidth(shell.root), MIN);
    } finally { shell.restore(); }
    const none = fakeShell({ min: '' });
    try { assert.equal(sidebarMinWidth(none.root), SIDEBAR_MIN_FALLBACK); } finally { none.restore(); }
    const ui = fs.readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    assert.match(ui, new RegExp(`--sidebar-min-width: ${SIDEBAR_MIN_FALLBACK}px;`), 'the fallback is the stylesheet value');
    const style = fs.readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(style, /width: max\(var\(--sidebar-min-width\), var\(--drag-width, var\(--sidebar-width\)\)\);/);
});

test('a narrower stored width is raised to the minimum and stored again; an allowed one is kept', async () => {
    const narrow = fakeShell();
    try {
        setupResizablePanels({ sidebar_width: 180 });
        assert.equal(narrow.vars.get('--sidebar-width'), `${MIN}px`);
        await settle();
        assert.deepEqual(narrow.posts, [{ sidebar_width: MIN }]);
    } finally { narrow.restore(); }
    const allowed = fakeShell();
    try {
        setupResizablePanels({ sidebar_width: 320 });
        assert.equal(allowed.vars.get('--sidebar-width'), '320px');
        await settle();
        assert.deepEqual(allowed.posts, [], 'an allowed width is not written back');
    } finally { allowed.restore(); }
});

test('a drag stops at the minimum, and a drag inside the range keeps its width', async () => {
    const drag = (shell, delta) => {
        const handle = shell.sidebar.handles[0];
        handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 500 + delta });
        const during = shell.sidebar.own.get('--drag-width');
        handle.listeners.pointerup({ pointerId: 1 });
        return during;
    };
    const shell = fakeShell({ sidebarWidth: 300 });
    try {
        setupResizablePanels({ sidebar_width: 300 });
        assert.equal(drag(shell, -200), `${MIN}px`, 'dragged far left, the sidebar stops at the minimum');
        assert.equal(shell.vars.get('--sidebar-width'), `${MIN}px`);
        shell.sidebar.width = MIN;
        assert.equal(drag(shell, 40), `${MIN + 40}px`, 'a width above the minimum is the one dragged to');
        assert.equal(shell.vars.get('--sidebar-width'), `${MIN + 40}px`);
        await settle();
        assert.deepEqual(shell.posts, [{ sidebar_width: MIN + 40 }]);
    } finally { shell.restore(); }
});
