import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { SIDEBAR_MAX, SIDEBAR_MIN_FALLBACK, setupResizablePanels, sidebarMaxWidth, sidebarMinWidth, sidebarRailExtra } from '../modules/panel_widths.js';

// The sidebar's resize handle and its stored width (panel_widths.js): below --sidebar-min-width (ui.css) Ouroboros's
// card, the search row and the Projects list no longer hold whole (the sections and foot live in the external rail),
// so a drag stops there and a narrower width stored before is raised when the page loads; valid desktop widths
// (including one temporarily capped by a narrow viewport's or inset #app's CSS max-width) are kept as stored and
// recover when the window widens or side insets clear.

const MIN = 270;

function fakeShell({
    min = `${MIN}px`,
    max = '560px',
    sidebarMaxToken = `${SIDEBAR_MAX}px`,
    sidebarWidth = 300,
    panelWidth = 400,
    extra = 0,
    extraRaw = null,
    railWidth = 0,
    gridColumns = '',
    paddingStart = '0px',
    paddingEnd = '12px',
    paddingLeft = null,
    paddingRight = null,
    direction = 'ltr',
    columnGap = 'normal',
    roomMin = '320px',
    viewportWidth = 1280,
    appWidth = null,
    boxCap = Infinity,
    useInlinePadding = true,
    deferFrames = false,
} = {}) {
    const bounds = { maxWidth: max, boxCap, viewportWidth, appWidth, sidebarMaxToken };
    const vars = new Map();
    const style = (map) => ({ setProperty: (k, v) => map.set(k, v), removeProperty: (k) => map.delete(k), getPropertyValue: (k) => map.get(k) || '' });
    const root = {
        dataset: { layout: 'docked' },
        style: style(vars),
        get clientWidth() { return bounds.viewportWidth; },
    };
    const appNode = { get clientWidth() { return bounds.appWidth ?? 0; }, getBoundingClientRect: () => ({ width: bounds.appWidth ?? 0 }) };
    const railNode = railWidth > 0 ? { getBoundingClientRect: () => ({ width: railWidth }) } : null;
    const element = (width, { isSidebar = false } = {}) => {
        const own = new Map();
        const handles = [];
        return {
            own, handles, width, hidden: false,
            style: style(own),
            // The box the stylesheet draws: the dragged or the column's width, plus what the rail adds, capped by CSS max-width.
            getBoundingClientRect() {
                if (this.hidden) return { width: 0 };
                const base = Number.parseFloat(own.get('--drag-width')) || this.width;
                const raw = base > 0 ? base + this.extra : 0;
                return { width: isSidebar ? Math.min(raw, bounds.boxCap) : raw };
            },
            querySelector: (sel) => (isSidebar && sel === '.nav-rail' ? railNode : null),
            parentElement: isSidebar ? appNode : null,
            extra: 0,
            appendChild(child) { handles.push(child); },
        };
    };
    const sidebar = element(sidebarWidth, { isSidebar: true });
    sidebar.extra = extra;
    const panel = element(panelWidth);
    const posts = [];
    const pendingFrames = new Map();
    let nextFrameId = 1;
    const previous = { document: globalThis.document, getComputedStyle: globalThis.getComputedStyle,
        requestAnimationFrame: globalThis.requestAnimationFrame, cancelAnimationFrame: globalThis.cancelAnimationFrame,
        fetch: globalThis.fetch, innerWidth: globalThis.innerWidth };
    globalThis.innerWidth = viewportWidth;
    globalThis.document = {
        documentElement: root,
        body: { classList: { add() {}, remove() {} } },
        getElementById: (id) => (id === 'primary-sidebar' ? sidebar : id === 'project-panel' ? panel : id === 'app' ? appNode : null),
        createElement: () => {
            const listeners = {};
            return { listeners, addEventListener: (k, fn) => { listeners[k] = fn; }, setPointerCapture() {}, releasePointerCapture() {} };
        },
    };
    const leftPad = paddingLeft ?? (direction === 'rtl' ? paddingEnd : paddingStart);
    const rightPad = paddingRight ?? (direction === 'rtl' ? paddingStart : paddingEnd);
    globalThis.getComputedStyle = (node) => {
        if (node === sidebar) {
            return {
                maxWidth: bounds.maxWidth,
                gridTemplateColumns: gridColumns,
                direction,
                columnGap,
                paddingLeft: leftPad,
                paddingRight: rightPad,
                ...(useInlinePadding ? { paddingInlineStart: paddingStart, paddingInlineEnd: paddingEnd } : {}),
                getPropertyValue: (k) => (k === '--nav-rail-extra' ? (extraRaw ?? `${extra}px`) : ''),
            };
        }
        return {
            maxWidth: 'none',
            getPropertyValue: (k) => (node !== root ? '' : k === '--sidebar-min-width' ? min
                : k === '--sidebar-max-width' ? bounds.sidebarMaxToken : k === '--room-min-width' ? roomMin : ''),
        };
    };
    globalThis.requestAnimationFrame = deferFrames
        ? (fn) => { const id = nextFrameId++; pendingFrames.set(id, fn); return id; }
        : (fn) => { fn(); return 0; };
    globalThis.cancelAnimationFrame = (id) => { pendingFrames.delete(id); };
    const flushFrames = () => {
        const queued = [...pendingFrames.values()];
        pendingFrames.clear();
        for (const fn of queued) fn();
    };
    globalThis.fetch = async (url, init) => { posts.push(JSON.parse(init.body)); return { ok: true }; };
    const restore = () => Object.assign(globalThis, previous);
    return { root, vars, sidebar, panel, bounds, posts, flushFrames, pendingFrames, restore };
}

const settle = () => new Promise((ok) => setTimeout(ok, 450));  // the stored width is posted after 400 ms
const drag = (shell, delta, target = shell.sidebar) => {
    const handle = target.handles[0];
    handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
    handle.listeners.pointermove({ clientX: 500 + delta });
    const during = target.own.get('--drag-width');
    handle.listeners.pointerup({ pointerId: 1 });
    return during;
};
const dragPath = (shell, deltas, target = shell.sidebar, { cancel = false } = {}) => {
    const handle = target.handles[0];
    const seen = [];
    handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
    for (const delta of deltas) {
        handle.listeners.pointermove({ clientX: 500 + delta });
        seen.push(target.own.get('--drag-width'));
    }
    handle.listeners[cancel ? 'pointercancel' : 'pointerup']({ pointerId: 1 });
    return seen;
};
const pressRelease = (shell, target = shell.sidebar) => {
    const handle = target.handles[0];
    handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
    const during = target.own.get('--drag-width');
    handle.listeners.pointerup({ pointerId: 1 });
    return during;
};
const dragCancel = (shell, delta, target = shell.sidebar) => {
    const handle = target.handles[0];
    handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
    handle.listeners.pointermove({ clientX: 500 + delta });
    const during = target.own.get('--drag-width');
    handle.listeners.pointercancel({ pointerId: 1 });
    return during;
};

test('the minimum and maximum are the stylesheet tokens, and the stylesheet draws the sidebar inside them', () => {
    const shell = fakeShell();
    try {
        assert.equal(sidebarMinWidth(shell.root), MIN);
        assert.equal(sidebarMaxWidth(shell.root), SIDEBAR_MAX);
    } finally { shell.restore(); }
    const custom = fakeShell({ min: '280px', sidebarMaxToken: '520px' });
    try {
        assert.equal(sidebarMinWidth(custom.root), 280);
        assert.equal(sidebarMaxWidth(custom.root), 520);
    } finally { custom.restore(); }
    const none = fakeShell({ min: '', sidebarMaxToken: '' });
    try {
        assert.equal(sidebarMinWidth(none.root), SIDEBAR_MIN_FALLBACK);
        assert.equal(sidebarMaxWidth(none.root), SIDEBAR_MAX);
    } finally { none.restore(); }
    const ui = fs.readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    assert.match(ui, new RegExp(`--sidebar-min-width: ${SIDEBAR_MIN_FALLBACK}px;`), 'the min fallback is the stylesheet value');
    assert.match(ui, new RegExp(`--sidebar-max-width: ${SIDEBAR_MAX}px;`), 'the max fallback is the stylesheet value');
    const style = fs.readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    // The box is the column's width and, beside it, what the rail adds (owner 24.11: the room pays for the rail).
    assert.match(style, /width: calc\(max\(var\(--sidebar-min-width\), var\(--drag-width, var\(--sidebar-width\)\)\) \+ var\(--nav-rail-extra\)\);/);
});

test('a narrower or above-max stored width is normalized once; an allowed desktop width is kept', async () => {
    const narrow = fakeShell();
    try {
        setupResizablePanels({ sidebar_width: 180 });
        assert.equal(narrow.vars.get('--sidebar-width'), `${MIN}px`);
        await settle();
        assert.deepEqual(narrow.posts, [{ sidebar_width: MIN }]);
    } finally { narrow.restore(); }
    const wide = fakeShell();
    try {
        setupResizablePanels({ sidebar_width: 720 });
        assert.equal(wide.vars.get('--sidebar-width'), `${SIDEBAR_MAX}px`);
        await settle();
        assert.deepEqual(wide.posts, [{ sidebar_width: SIDEBAR_MAX }]);
    } finally { wide.restore(); }
    const allowed = fakeShell();
    try {
        setupResizablePanels({ sidebar_width: 320 });
        assert.equal(allowed.vars.get('--sidebar-width'), '320px');
        await settle();
        assert.deepEqual(allowed.posts, [], 'an allowed width is not written back');
    } finally { allowed.restore(); }
});

test('a drag stops at the minimum, and a drag inside the range keeps its width', async () => {
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

test('the stylesheet caps every sidebar width by #app width and owns shared column bounds for overlays', () => {
    const ui = fs.readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    assert.match(ui, /--room-min-width: 320px;/, 'the folded composer and question count fit this column');
    const css = fs.readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    const decl = (prop) => css.match(new RegExp(`^\\s*${prop}\\s*:([^;]+);`, 'm'))?.[1] || '';
    const appBlock = css.match(/(?:^|\n)#app\s*\{([^}]*)\}/)?.[1] || '';
    assert.match(appBlock, /margin-left\s*:\s*var\(--app-inset-left\)/);
    assert.match(appBlock, /margin-right\s*:\s*var\(--app-inset-right\)/);
    assert.doesNotMatch(css, /html\[data-telegram-fullscreen\] #app\s*\{/, 'no duplicate #app horizontal inset rule');
    const dockedSidebar = css.match(/html\[data-layout="docked"\] #primary-sidebar\s*\{([^}]*)\}/)?.[1] || '';
    assert.match(dockedSidebar, /max-width\s*:\s*var\(--sidebar-max-box-width\)/);
    const maxBox = decl('--sidebar-max-box-width');
    assert.match(maxBox, /var\(--app-width\)/);
    assert.match(maxBox, /var\(--room-min-width\)/);
    assert.match(decl('--sidebar-column-start'), /var\(--nav-rail-w\)/);
    const colWidth = decl('--sidebar-column-width');
    assert.match(colWidth, /var\(--sidebar-effective-width\)/);
    assert.match(colWidth, /var\(--nav-rail-w\)/);
    const callCss = fs.readFileSync(new URL('../../skills/live_conversation/ui/call.css', import.meta.url), 'utf8');
    assert.match(callCss, /inset-inline-start\s*:\s*var\(--sidebar-column-start,/);
    assert.match(callCss, /width\s*:\s*var\(--sidebar-column-width,/);
    assert.doesNotMatch(callCss, /--sidebar-width/, 'docked call mark never reads raw unclamped preference');
});

test('a valid desktop width opened in a narrow docked window never posts narrower, survives no-motion, and recovers when widened', async () => {
    // 720 px docked window leaves 720 - 320 = 400 px for the whole box (360 px column + 40 px rail).
    // CSS max-width caps the rendered box without mutating or persisting the owner's 400 px desktop preference.
    const shell = fakeShell({
        max: '400px',
        sidebarWidth: 400,
        extra: 40,
        railWidth: 52,
        gridColumns: '52px 335px',
        viewportWidth: 720,
        boxCap: 400,
    });
    try {
        setupResizablePanels({ sidebar_width: 400 });
        assert.equal(shell.vars.get('--sidebar-width'), '400px', 'stored preference stays intact on narrow boot');
        assert.equal(shell.sidebar.getBoundingClientRect().width, 400, 'CSS caps the displayed box to leave the room 320 px');
        await settle();
        assert.deepEqual(shell.posts, [], 'never POST a narrower preference on boot');

        // Press/release without motion (both zero pointermove and delta=0), dragging right into the cap, and
        // moving inward then returning to the exact starting width before release all leave the unshown 400 px intact.
        assert.equal(pressRelease(shell), undefined, 'no-motion press/release writes no drag width');
        assert.equal(drag(shell, 0), undefined, 'zero-delta move writes no drag width');
        assert.equal(drag(shell, 60), '360px', 'dragging right into the viewport cap clamps the live drag to 360 px');
        assert.deepEqual(dragPath(shell, [-40, 0]), ['320px', '360px'], 'out-and-back at the cap returns live drag to 360 px');
        assert.deepEqual(dragPath(shell, [-40, 25]), ['320px', '360px'], 'inward then back past the cap returns to 360 px');
        assert.equal(shell.vars.get('--sidebar-width'), '400px', 'no-net-change interactions keep 400 px preference');
        await settle();
        assert.deepEqual(shell.posts, [], 'still zero POSTs');

        // Widening the window lifts the CSS display cap and recovers the original 400 px column (440 px box).
        shell.bounds.maxWidth = '600px';
        shell.bounds.boxCap = 600;
        shell.bounds.viewportWidth = 1280;
        assert.equal(shell.sidebar.getBoundingClientRect().width, 440, 'widening recovers 400 px column + 40 px rail');
        assert.equal(shell.vars.get('--sidebar-width'), '400px');
        await settle();
        assert.deepEqual(shell.posts, [], 'mere resize never mutates or posts');

        // Narrow again and explicitly drag narrower: now a real drag commits the new preference.
        shell.bounds.maxWidth = '400px';
        shell.bounds.boxCap = 400;
        shell.bounds.viewportWidth = 720;
        assert.equal(drag(shell, -30), '330px');
        assert.equal(shell.vars.get('--sidebar-width'), '330px');
        await settle();
        assert.deepEqual(shell.posts, [{ sidebar_width: 330 }]);
    } finally { shell.restore(); }

    // Drag bounds still follow dynamic viewport maxWidth changes while preserving valid stored widths like 560 or 400.25.
    for (const stored of [560, 400.25]) {
        const dynamic = fakeShell({ max: '480px', sidebarWidth: 300 });
        try {
            setupResizablePanels({ sidebar_width: stored });
            assert.equal(dynamic.vars.get('--sidebar-width'), `${stored}px`);
            await settle();
            assert.deepEqual(dynamic.posts, [], 'valid desktop widths never post on narrow boot');
            assert.equal(drag(dynamic, 260), '480px');
            assert.equal(dynamic.vars.get('--sidebar-width'), '480px');
            dynamic.bounds.maxWidth = '440px';
            assert.equal(drag(dynamic, 200), '440px');
            dynamic.bounds.maxWidth = '560px';
            assert.equal(drag(dynamic, 200), '500px');
            await settle();
            assert.deepEqual(dynamic.posts.at(-1), { sidebar_width: 500 });
        } finally { dynamic.restore(); }
    }
});

test('unregistered calc() and expression maxWidth derive rail extra from rendered geometry without counting border/padding/scrollbars', async () => {
    const css = fs.readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /@property --nav-rail-extra \{\s*syntax: '<length>';/, 'a registered length where supported');
    assert.match(css, /--nav-rail-extra: calc\(var\(--nav-rail-w\) - var\(--space-3\)\);/, 'the rail less the inset it replaces');

    // Simulate WebKit without @property: --nav-rail-extra stays a raw calc() string and maxWidth stays a min()/calc() expression.
    // The grid's second track (249 px) reflects a 1 px border and 6 px scrollbar on a 340 px box; neither is counted as extra rail.
    const raw = fakeShell({
        sidebarWidth: 300,
        extra: 40,
        extraRaw: 'calc(var(--nav-rail-w) - var(--space-3))',
        max: 'min(calc(560px + var(--nav-rail-extra)), calc(100vw - var(--room-min-width)))',
        railWidth: 52,
        gridColumns: '52px 249px',
        paddingStart: '0px',
        paddingEnd: '12px',
        viewportWidth: 1280,
    });
    const follow = () => { raw.sidebar.width = Number.parseFloat(raw.vars.get('--sidebar-width')); };
    try {
        assert.equal(sidebarRailExtra(raw.sidebar), 40, 'geometry derives 52 - 12 = 40 without border/scrollbar drift');
        setupResizablePanels({ sidebar_width: 300 });
        assert.equal(pressRelease(raw), undefined, 'no-motion press/release never mutates under raw calc()');
        assert.equal(drag(raw, 0), undefined, 'zero-delta move never mutates under raw calc()');
        assert.equal(raw.vars.get('--sidebar-width'), '300px');
        await settle();
        assert.deepEqual(raw.posts, []);

        for (const step of [10, 10, 10, -30]) {
            const before = raw.sidebar.width;
            assert.equal(drag(raw, step), `${before + step}px`, 'each real drag starts from the column shown');
            assert.equal(raw.vars.get('--sidebar-width'), `${before + step}px`, 'stores the column, never adding the 40 px rail');
            follow();
        }
        assert.equal(raw.sidebar.getBoundingClientRect().width, 300 + 40);
        await settle();
        assert.deepEqual(raw.posts.at(-1), { sidebar_width: 300 });

        // Expression maxWidth falls back to min(560, viewportWidth - --room-min-width - extra).
        raw.bounds.viewportWidth = 720;
        raw.bounds.boxCap = 400;
        assert.equal(drag(raw, 200), '360px', 'at 720 px expression maxWidth caps drag at 720 - 320 - 40 = 360 px');
        raw.bounds.viewportWidth = 1280;
        raw.bounds.boxCap = Infinity;
        raw.sidebar.width = 360;
        assert.equal(drag(raw, 400), '560px', 'at 1280 px expression maxWidth caps drag at 560 px');
        raw.sidebar.width = 560;
        assert.equal(dragCancel(raw, -40), '520px', 'pointercancel cleans up without committing');
        assert.equal(raw.vars.get('--sidebar-width'), '560px');
        assert.equal(raw.sidebar.own.get('--drag-width'), undefined);
        await settle();
        assert.deepEqual(raw.posts.at(-1), { sidebar_width: 560 }, 'pointercancel did not overwrite the previous drag');
    } finally { raw.restore(); }

    // RTL without paddingInlineStart/End still resolves physical start/end padding accurately, and a 1-column grid never counts as a side rail.
    const rtl = fakeShell({
        extra: 40,
        extraRaw: 'calc(52px - 12px)',
        railWidth: 52,
        gridColumns: '52px 255px',
        direction: 'rtl',
        paddingStart: '0px',
        paddingEnd: '12px',
        useInlinePadding: false,
    });
    try { assert.equal(sidebarRailExtra(rtl.sidebar), 40); } finally { rtl.restore(); }

    const stacked = fakeShell({ extra: 0, extraRaw: 'calc(52px - 12px)', railWidth: 320, gridColumns: '320px' });
    try { assert.equal(sidebarRailExtra(stacked.sidebar), 0, 'a single-column layout is never treated as an extra side rail'); } finally { stacked.restore(); }

    // Deferred-rAF sidebar gestures at the CSS cap: returning to original width with a pending final frame leaves the
    // unshown preference intact, and pointercancel discards a pending frame without saving.
    const deferred = fakeShell({
        max: '400px',
        sidebarWidth: 400,
        extra: 40,
        railWidth: 52,
        gridColumns: '52px 335px',
        viewportWidth: 720,
        boxCap: 400,
        deferFrames: true,
    });
    try {
        setupResizablePanels({ sidebar_width: 400 });
        const handle = deferred.sidebar.handles[0];
        handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 460 });
        deferred.flushFrames();
        assert.equal(deferred.sidebar.own.get('--drag-width'), '320px');
        handle.listeners.pointermove({ clientX: 500 });
        assert.equal(deferred.pendingFrames.size, 1, 'return frame is still pending before release');
        handle.listeners.pointerup({ pointerId: 1 });
        assert.equal(deferred.pendingFrames.size, 0);
        assert.equal(deferred.vars.get('--sidebar-width'), '400px', 'flushed return-to-start keeps unshown 400 px');
        assert.equal(deferred.sidebar.own.get('--drag-width'), undefined);

        handle.listeners.pointerdown({ clientX: 500, pointerId: 2, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 455 });
        assert.equal(deferred.pendingFrames.size, 1, 'cancel frame is still pending');
        handle.listeners.pointercancel({ pointerId: 2 });
        assert.equal(deferred.pendingFrames.size, 0);
        assert.equal(deferred.vars.get('--sidebar-width'), '400px', 'sidebar pointercancel discards pending drag');
        assert.equal(deferred.sidebar.own.get('--drag-width'), undefined);
        await settle();
        assert.deepEqual(deferred.posts, []);
    } finally { deferred.restore(); }

    // Project panel keeps its exact original handle semantics (no rail subtraction, 320..1100 bounds, pointerup
    // without movement persists displayed width, pointercancel flushes a pending final frame and persists, and
    // hiding mid-drag leaves the stored width intact).
    const panelShell = fakeShell({
        sidebarWidth: 300,
        panelWidth: 540,
        extra: 40,
        railWidth: 52,
        gridColumns: '52px 255px',
        deferFrames: true,
    });
    try {
        setupResizablePanels({ sidebar_width: 300, project_panel_width: 540 });
        assert.equal(panelShell.vars.get('--project-panel-width'), '540px');
        const handle = panelShell.panel.handles[0];

        // Pointerup without movement reads and persists displayed width.
        assert.equal(pressRelease(panelShell, panelShell.panel), undefined);
        assert.equal(panelShell.vars.get('--project-panel-width'), '540px');
        await settle();
        assert.deepEqual(panelShell.posts, [{ project_panel_width: 540 }]);

        // Pointercancel after +45 px widen (leftward drag, dir === 'left') with a pending final frame flushes and stores 585 px.
        handle.listeners.pointerdown({ clientX: 500, pointerId: 1, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 455 });
        assert.equal(panelShell.pendingFrames.size, 1, 'final Project frame is pending before pointercancel');
        assert.equal(panelShell.panel.own.get('--drag-width'), undefined);
        handle.listeners.pointercancel({ pointerId: 1 });
        assert.equal(panelShell.pendingFrames.size, 0);
        assert.equal(panelShell.vars.get('--project-panel-width'), '585px', 'Project pointercancel keeps 585 px');
        assert.equal(panelShell.panel.own.get('--drag-width'), undefined);
        panelShell.panel.width = 585;
        await settle();
        assert.deepEqual(panelShell.posts.at(-1), { project_panel_width: 585 });

        // Pointercancel after an already-flushed frame also keeps and persists the dragged width.
        handle.listeners.pointerdown({ clientX: 500, pointerId: 2, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 485 });
        panelShell.flushFrames();
        assert.equal(panelShell.panel.own.get('--drag-width'), '600px');
        handle.listeners.pointercancel({ pointerId: 2 });
        assert.equal(panelShell.vars.get('--project-panel-width'), '600px');
        panelShell.panel.width = 600;
        await settle();
        assert.deepEqual(panelShell.posts.at(-1), { project_panel_width: 600 });

        // Hiding the Project panel mid-drag (e.g. Escape closed it) leaves the stored width intact on pointerup and pointercancel.
        const postsBeforeHide = panelShell.posts.length;
        for (const endEvent of ['pointerup', 'pointercancel']) {
            handle.listeners.pointerdown({ clientX: 500, pointerId: 3, preventDefault() {} });
            handle.listeners.pointermove({ clientX: 420 });
            panelShell.panel.hidden = true;
            handle.listeners[endEvent]({ pointerId: 3 });
            panelShell.panel.hidden = false;
            assert.equal(panelShell.vars.get('--project-panel-width'), '600px', `hidden mid-drag (${endEvent}) keeps stored width`);
            assert.equal(panelShell.panel.own.get('--drag-width'), undefined);
        }
        await settle();
        assert.equal(panelShell.posts.length, postsBeforeHide, 'hidden mid-drag never posts');

        // Normal drag bounds (320..1100) without rail subtraction.
        handle.listeners.pointerdown({ clientX: 500, pointerId: 4, preventDefault() {} });
        handle.listeners.pointermove({ clientX: 1000 });
        handle.listeners.pointerup({ pointerId: 4 });
        assert.equal(panelShell.vars.get('--project-panel-width'), '320px', 'Project panel stops at its 320 px minimum');
        handle.listeners.pointerdown({ clientX: 500, pointerId: 5, preventDefault() {} });
        handle.listeners.pointermove({ clientX: -1500 });
        handle.listeners.pointerup({ pointerId: 5 });
        assert.equal(panelShell.vars.get('--project-panel-width'), '1100px', 'Project panel stops at its 1100 px maximum');
        await settle();
        assert.deepEqual(panelShell.posts.at(-1), { project_panel_width: 1100 });
    } finally { panelShell.restore(); }
});

test('paired Telegram side insets cap docked drags against #app width (including WebKit subpixel quantization) and honor --sidebar-max-width', async () => {
    // 844 px viewport with paired 47 px insets -> 750 px #app; WebKit lays out 429.984375 px box / 320.015625 px room.
    const exprMax = 'min(calc(var(--sidebar-max-width) + var(--nav-rail-extra)), calc(var(--app-width) - var(--room-min-width)))';
    for (const resolved of [true, false]) {
        const inset = { maxWidth: resolved ? '429.984375px' : exprMax, appWidth: resolved ? 750 : 749.984375, boxCap: 429.984375 };
        const shell = fakeShell({
            max: inset.maxWidth, sidebarWidth: 420, extra: 40, extraRaw: resolved ? '40px' : 'calc(var(--nav-rail-w) - var(--space-3))',
            railWidth: 52, gridColumns: '52px 365px', viewportWidth: 844, appWidth: inset.appWidth, boxCap: inset.boxCap,
        });
        try {
            setupResizablePanels({ sidebar_width: 420 });
            assert.equal(shell.vars.get('--sidebar-width'), '420px');
            assert.equal(shell.sidebar.getBoundingClientRect().width, 429.984375);
            assert.equal(pressRelease(shell), undefined);
            assert.equal(drag(shell, 80), '390px', 'subpixel cap rounds to 390 px, never pulling inward to 389 px');
            assert.equal(shell.sidebar.getBoundingClientRect().width, 429.984375, 'CSS stays the final fractional cap');
            assert.equal(shell.vars.get('--sidebar-width'), '420px');
            await settle();
            assert.deepEqual(shell.posts, []);
            Object.assign(shell.bounds, { maxWidth: resolved ? '524px' : exprMax, appWidth: 844, boxCap: 524 });
            assert.equal(shell.sidebar.getBoundingClientRect().width, 460);
            Object.assign(shell.bounds, inset);
            assert.equal(drag(shell, -30), '360px');
            assert.equal(shell.vars.get('--sidebar-width'), '360px');
            await settle();
            assert.deepEqual(shell.posts, [{ sidebar_width: 360 }]);
        } finally { shell.restore(); }
    }
    const customMax = fakeShell({
        sidebarMaxToken: '500px', max: exprMax, sidebarWidth: 460, extra: 40,
        extraRaw: 'calc(var(--nav-rail-w) - var(--space-3))', railWidth: 52, gridColumns: '52px 435px', viewportWidth: 1400, appWidth: 1400,
    });
    try {
        setupResizablePanels({ sidebar_width: 540 });
        assert.equal(customMax.vars.get('--sidebar-width'), '500px');
        assert.equal(drag(customMax, 200), '500px');
        await settle();
        assert.deepEqual(customMax.posts, [{ sidebar_width: 500 }]);
    } finally { customMax.restore(); }
});
