import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bindHeaderControlPlacement, bindPageBack, bindPageHeader, renderMobileNavToggle, renderPageHeader } from '../modules/page_header.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';
import { createRoomHeader } from '../modules/chat_header.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

test('Skills keeps its return and tabs at the leading edge, with one extra line only in a narrow page column', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /#page-skills \{ container: skillspage \/ inline-size; \}/);
    assert.match(css, /#page-skills \.app-page-return \{ flex: none; max-width: none; \}/,
        'changing the trailing actions cannot shrink or move the return before the tabs');
    assert.match(css, /#page-skills \.skills-scroll \{ padding-top: 0; \}/);
    assert.match(css, /#page-skills > \.app-page-tabs \{ padding-bottom: 0; \}/);
    assert.match(css, /#page-skills > \.app-page-tabs > \.app-tab-strip \{[\s\S]*?margin-bottom: calc\(-1 \* var\(--space-4\)\);/,
        'the strip lends its shadows space without a blank row before the search');
    const narrow = css.slice(css.indexOf('@container skillspage'), css.indexOf('.skills-search-chrome {'));
    assert.match(narrow, /max-width: 44rem/);
    assert.match(narrow, /grid-template-areas: "back toolbar" "tabs tabs"/);
    assert.match(narrow, /\.app-page-toolbar \{ min-height: max\(50px, var\(--window-chrome-top, 0px\)\);/,
        'the first row keeps its band with native chrome and without the host token');
    assert.match(narrow, /\.app-page-lead \{ display: contents;/);
    assert.match(narrow, /\.app-tab-strip \{[\s\S]*?grid-area: tabs;/);
    assert.match(css, /html\[data-layout="docked"\] \.app-page-header\[data-lead\] \{[\s\S]*?grid-template-areas: "title toolbar"/,
        'a wide column keeps the single existing row');
    assert.doesNotMatch(narrow, /@media|ResizeObserver/);
});

test('page headers retain their toolbar contract with an optional chrome recipe', () => {
    for (const field of ['toolbarHtml', 'actionsHtml']) {
        const basic = renderPageHeader({ title: 'Chat', [field]: '<button>Act</button>' });
        assert.match(basic, /class="app-page-toolbar app-page-actions"/);
        const grouped = renderPageHeader({ title: 'Chat', [field]: '<button>Act</button>', toolbarClass: 'ui-chrome-group' });
        assert.match(grouped, /class="app-page-toolbar app-page-actions ui-chrome-group"/);
        assert.equal((grouped.match(/<button>Act<\/button>/g) || []).length, 1);
    }
    assert.doesNotMatch(renderPageHeader({ title: 'Chat', toolbarClass: 'ui-chrome-group' }), /app-page-toolbar/);
});

test('one header: no icon, no caption line, the tabs are written under the row, a Project keeps its name', () => {
    const page = renderPageHeader({ title: 'Widgets', icon: '<svg/>', description: 'Reviewed surfaces live here.' });
    assert.doesNotMatch(page, /app-page-icon|<svg\/>/, 'no page icon anywhere (owner 5A); the ☰ glyph is a control');
    assert.doesNotMatch(page, /Reviewed surfaces live here|app-page-lede|app-page-description/, 'a page has no caption line');
    assert.match(page, /<div class="page-header app-page-header" data-sidebar-names data-empty>/, 'named by the sidebar, nothing else to say');
    const tabbed = renderPageHeader({ title: 'Dashboard', tabsHtml: '<div role="tablist"></div>' });
    assert.match(tabbed, /<\/div>\s*<div class="app-page-tabs"><div role="tablist"><\/div><\/div>/, 'the stack layout keeps them under the row');
    assert.equal(tabbed.indexOf('app-page-tabs') > tabbed.indexOf('</div>'), true);
    // On a desk the tabs take the hidden title's place (bindPageHeader): the bar is not empty (owner 19.10, A).
    assert.match(tabbed, /data-sidebar-names>/);
    assert.doesNotMatch(tabbed, /data-empty/);
    // The shell return slot stands before the lead, both after the hidden title.
    assert.match(tabbed, /<\/h2>\s*<\/div><span class="app-page-return" data-page-return><\/span><div class="app-page-lead"><\/div>/, 'the Back slot, then the lead slot, follow the hidden title');
    const acting = renderPageHeader({ title: 'Files', actionsHtml: '<button>Refresh</button>' });
    assert.match(acting, /data-sidebar-names>/);
    assert.doesNotMatch(acting, /data-empty/);
    const project = renderPageHeader({ title: '', titleId: 'project-panel-title', named: false, trailingHtml: '<button>×</button>' });
    assert.match(project, /<h2 class="app-page-title" id="project-panel-title"><\/h2>/);
    assert.doesNotMatch(project, /data-sidebar-names|data-empty|"app-page-lead"/, 'a Project is named by its owner: its name stays');
    assert.match(project, /app-page-trailing/);
    assert.match(project, /data-mobile-nav-toggle/, 'a header shows ☰ unless its room carries it in the composer');
    assert.match(renderPageHeader({ title: 'x', titleId: '"><i>' }), /id="&quot;&gt;&lt;i&gt;"/);
    assert.doesNotMatch(renderPageHeader({ title: 'x', showMobileNav: false }), /data-mobile-nav-toggle/);
    // A room's ☰ (owner 19.10, 1A): the composer row's first control, capsule-sized with the finger's hit area.
    assert.match(renderMobileNavToggle({ inComposer: true }), /class="shell-icon-control mobile-nav-toggle composer-menu ui-hit-area"/);
    assert.match(renderMobileNavToggle(), /class="shell-icon-control mobile-nav-toggle"/);
    // The control is Ouroboros's own picture (press him, and his places open). Both placements carry one status mark
    // for what waits behind it, hidden until app.js paints it.
    for (const toggle of [renderMobileNavToggle(), renderMobileNavToggle({ inComposer: true })]) {
        assert.match(toggle, /aria-label="Open navigation"[^>]*><span class="places-face" aria-hidden="true"><\/span><span class="status-mark" aria-hidden="true" data-shape="none" data-hue="ink" hidden><i><\/i><i><\/i><i><\/i><\/span><\/button>/);
        assert.doesNotMatch(toggle, /<svg|places-menu|places-back/, 'no prototype variant of the control remains');
    }
});

test('Restart and the budget are the same nodes, placed by layout: the row on a desk, the status Shade on a phone', () => {
    const node = (name, parent) => {
        const el = { name, children: [], parentElement: null,
            get nextSibling() { const i = this.parentElement.children.indexOf(this); return this.parentElement.children[i + 1] || null; },
            detach() { if (this.parentElement) this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1); },
            prepend(x) { x.detach(); x.parentElement = this; this.children.unshift(x); },
            append(x) { x.detach(); x.parentElement = this; this.children.push(x); },
            insertBefore(x, ref) { x.detach(); x.parentElement = this; const i = ref ? this.children.indexOf(ref) : -1; if (i < 0) this.children.push(x); else this.children.splice(i, 0, x); },
            querySelector(sel) { const all = []; (function walk(n) { for (const c of n.children) { all.push(c); walk(c); } }(this)); return all.find((c) => c.sel === sel) || null; } };
        if (parent) parent.append(el);
        return el;
    };
    const toolbar = node('toolbar');
    const actions = node('actions', toolbar);
    const restart = node('restart', actions); restart.sel = '[data-chat-command="restart"]';
    const panic = node('panic', actions); panic.sel = '[data-chat-command="panic"]';
    const budget = node('budget', toolbar);
    const status = node('status', toolbar);
    // The room status control (room_work.js): its Shade's end slot, repainted after each move.
    let repaints = 0;
    const shade = { extras: node('extras'), update: () => { repaints += 1; } };
    const listeners = new Map();
    globalThis.window = { ouroLayout: { layout: 'stack' }, addEventListener: (t, fn) => listeners.set(t, fn), removeEventListener: (t) => listeners.delete(t) };
    globalThis.document = { documentElement: { dataset: {} } };
    const dispose = bindHeaderControlPlacement({ headerActions: actions, budgetPill: budget, shade });
    assert.deepEqual(shade.extras.children.map((c) => c.name), ['restart', 'budget'], 'stack: Restart, then the budget, in the status Shade');
    assert.deepEqual(actions.children.map((c) => c.name), ['panic'], 'Panic stays in the bar; no ⋯');
    assert.equal(repaints, 1, 'the Shade learns its slot changed');
    globalThis.window.ouroLayout.layout = 'docked';
    listeners.get('ouro:layout-changed')({ detail: { layout: 'docked' } });
    assert.deepEqual(actions.children.map((c) => c.name), ['restart', 'panic'], 'docked: Restart back before Panic');
    assert.deepEqual(toolbar.children.map((c) => c.name), ['actions', 'budget', 'status'], 'the budget back after the actions');
    assert.deepEqual(shade.extras.children, [], 'the Shade keeps nothing of the row');
    // A docked bar too narrow for the full row (a narrow content column) moves them too; a wide one keeps the row.
    const bar = { clientWidth: 500 };
    const narrow = bindHeaderControlPlacement({ headerActions: actions, budgetPill: budget, shade, bar });
    assert.deepEqual(shade.extras.children.map((c) => c.name), ['restart', 'budget'], 'a 500px bar: in the Shade');
    for (const width of [0, 500]) {
        bar.clientWidth = width;
        listeners.get('ouro:layout-changed')({ detail: { layout: 'docked' } });
        assert.deepEqual(shade.extras.children.map((c) => c.name), ['restart', 'budget'], 'hidden and returned: no move that could strand restored focus');
    }
    bar.clientWidth = 900;
    listeners.get('ouro:layout-changed')({ detail: { layout: 'docked' } });
    assert.deepEqual(shade.extras.children, [], 'a visible wider bar still moves its controls');
    bar.clientWidth = 0;
    listeners.get('ouro:layout-changed')({ detail: { layout: 'docked' } });
    assert.deepEqual(toolbar.children.map((c) => c.name), ['actions', 'budget', 'status'], 'a hidden wide bar keeps its row too');
    narrow();
    const wideToolbar = node('toolbar');
    const wideActions = node('actions', wideToolbar);
    node('restart', wideActions).sel = '[data-chat-command="restart"]';
    node('panic', wideActions).sel = '[data-chat-command="panic"]';
    const wideShade = { extras: node('extras') };
    const wide = bindHeaderControlPlacement({ headerActions: wideActions, budgetPill: node('budget', wideToolbar),
        shade: wideShade, bar: { clientWidth: 900 } });
    assert.deepEqual(wideActions.children.map((c) => c.name), ['restart', 'panic'], 'a 900px bar: the row');
    assert.deepEqual(wideShade.extras.children, []);
    wide();
    dispose();
    assert.equal(listeners.size, 0);
    assert.equal(typeof bindHeaderControlPlacement({ headerActions: null }), 'function', 'no actions: a no-op disposer');
    assert.equal(typeof bindHeaderControlPlacement({ headerActions: actions }), 'function', 'no Shade (a Project bar): a no-op disposer');
    delete globalThis.window; delete globalThis.document;
});

test('a room\'s bar ends in one group of capsules, the consciousness chip first, one gap apart (owner seed S3)', () => {
    const env = installDom();
    try {
        // Main's commands (.app-page-actions) and a Project's Panic and × (.app-page-trailing): the chip leads either.
        for (const slot of ['app-page-toolbar app-page-actions ui-chrome-group', 'app-page-trailing']) {
            const bar = document.createElement('div');
            const title = document.createElement('div');
            title.className = 'app-page-title-block';
            const group = document.createElement('div');
            group.className = slot;
            const panic = document.createElement('button');
            panic.className = 'chat-header-btn danger ui-chrome-control';
            group.append(panic);
            bar.append(title, group);
            const header = createRoomHeader({ bar, ws: { isConnected: () => true, send: () => ({ status: 'sent' }) } });
            const chip = group.children[0];
            assert.ok(chip.classList.contains('cx-chip'), `${slot}: the chip leads the end group`);
            assert.equal(group.children[1], panic);
            assert.equal(bar.children.includes(chip), false, 'never a capsule of its own between the title and the group');
            header.destroy();
        }
    } finally { restoreDom(env); }
    // One gap rule holds the whole row: the group's, with Main's commands laid in it; no second gap inside.
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    assert.match(css, /:is\(\.chat-page-header, \.project-panel-bar\) > :is\(\.app-page-actions, \.app-page-trailing\) \{ gap: var\(--space-2\); \}/);
    assert.match(css, /\.chat-header-actions \{ display: contents; \}/);
    assert.equal([...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].filter(([, head, body]) => head.includes('chat-header-actions') && /gap/.test(body)).length, 0);
    // The budget's amount keeps the capsule's centre line; its track is out of the line (U004).
    const track = css.match(/\.chat-budget-bar \{([^}]*)\}/)[1];
    assert.match(track, /position: absolute/);
    assert.doesNotMatch(css.match(/\.chat-budget-pill \{([^}]*)\}/)[1], /flex-direction: column/);
});

test('a page\'s own navigation starts its bar on a desk and goes back under the row in the stack layout', () => {
    // A small DOM: comment marks, and an engine that drops focus from a node it moves.
    const doc = { activeElement: null, createComment: () => el('#comment') };
    function el(name, ...kids) {
        const node = { name, children: [], parentNode: null, attrs: new Set(), ownerDocument: doc, isConnected: true,
            classList: { contains: (value) => node.name === value },
            get firstElementChild() { return node.children.find((child) => child.name !== '#comment') || null; },
            get nextElementSibling() {
                const list = node.parentNode?.children || [];
                return list.slice(list.indexOf(node) + 1).find((child) => child.name !== '#comment') || null;
            },
            contains: (other) => other === node || node.children.some((child) => child.contains(other)),
            detach() {
                if (!node.parentNode) return;
                const list = node.parentNode.children;
                list.splice(list.indexOf(node), 1);
                node.parentNode = null;
                if (doc.activeElement && node.contains(doc.activeElement)) doc.activeElement = null;
            },
            insert(index, child) { child.detach(); child.parentNode = node; node.children.splice(index, 0, child); },
            append(child) { node.insert(node.children.length, child); },
            prepend(child) { node.insert(0, child); },
            before(child) { node.parentNode.insert(node.parentNode.children.indexOf(node), child); },
            after(child) { node.parentNode.insert(node.parentNode.children.indexOf(node) + 1, child); },
            querySelector(selector) {
                const want = selector.replace(/^\./, '');
                const walk = (at) => at.children.reduce((found, child) => found || (child.name === want ? child : walk(child)), null);
                return walk(node);
            },
            toggleAttribute(attr, on) { if (on) node.attrs.add(attr); else node.attrs.delete(attr); },
            focus() { doc.activeElement = node; } };
        kids.forEach((kid) => node.append(kid));
        return node;
    }
    const names = (node) => node.children.filter((child) => child.name !== '#comment').map((child) => child.name);
    const listeners = new Map();
    globalThis.window = { ouroLayout: { layout: 'docked' }, addEventListener: (type, fn) => listeners.set(type, fn),
        removeEventListener: (type) => listeners.delete(type) };
    globalThis.document = { documentElement: { dataset: {} } };
    const layout = (name) => listeners.get('ouro:layout-changed')({ detail: { layout: name } });
    try {
        // Dashboard, Settings, Skills: the tab strip renderPageHeader wrote under the row.
        const tab = el('tab');
        const strip = el('app-tab-strip', tab);
        const under = el('app-page-tabs', strip);
        const lead = el('app-page-lead');
        const header = el('app-page-header', el('app-page-title-block', el('app-page-title-row'), lead), el('app-page-toolbar'));
        el('page', header, under);
        tab.focus();
        const dispose = bindPageHeader(header);
        assert.equal(strip.parentNode, lead, 'a desk: the tabs start the bar');
        assert.ok(header.attrs.has('data-lead'));
        assert.ok(header.attrs.has('data-scroll-x'), 'the one-row desk band owns its horizontal scroll');
        assert.equal(doc.activeElement, tab, 'the owner\'s focus stays on the moved tab');
        layout('stack');
        assert.deepEqual([strip.parentNode, names(under), lead.children.length, header.attrs.has('data-lead')],
            [under, ['app-tab-strip'], 0, false], 'the stack layout: the same node back under the row');
        assert.ok(!header.attrs.has('data-scroll-x'), 'the phone header gives the horizontal strip back its ownership');
        layout('docked');
        assert.equal(strip.parentNode, lead);
        dispose();
        assert.equal(listeners.size, 0);
        // Files: its location leads; its own actions join the bar's end before Refresh, and go home with it.
        const location = el('files-breadcrumb');
        const actions = el('files-browser-actions');
        const home = el('files-browser-header', location, actions);
        const filesLead = el('app-page-lead');
        const toolbar = el('app-page-toolbar', el('refresh'));
        const filesHeader = el('app-page-header', el('app-page-title-block', filesLead), toolbar);
        el('page', filesHeader, el('files-layout', home));
        filesHeader.clientWidth = 440;
        const off = bindPageHeader(filesHeader, { lead: location, actions });
        assert.deepEqual([names(filesLead), names(toolbar), names(home)], [['files-breadcrumb'], ['files-browser-actions', 'refresh'], []]);
        actions.focus();
        filesHeader.clientWidth = 175;
        layout('docked');
        assert.deepEqual([names(home), names(toolbar)], [[], ['files-browser-actions', 'refresh']],
            'a narrow column keeps the same one band; the navigation rail yields to actions');
        assert.equal(doc.activeElement, actions);
        layout('stack');
        assert.deepEqual([names(home), names(toolbar)], [['files-breadcrumb', 'files-browser-actions'], ['refresh']]);
        off();
        // A header with nothing of its own to place (Widgets) binds nothing.
        assert.equal(typeof bindPageHeader(el('app-page-header', el('app-page-title-block', el('app-page-lead')))), 'function');
        assert.equal(listeners.size, 0);
        // A bar whose one action is already in it (Circles): nothing moves, yet on a desk it lays out as the band of
        // every named page, and goes back to the base header in the stack layout.
        const barOnly = el('app-page-header', el('app-page-title-block', el('app-page-lead')), el('app-page-toolbar', el('new')));
        barOnly.clientWidth = 900;
        const offBar = bindPageHeader(barOnly);
        assert.ok(barOnly.attrs.has('data-lead'), 'a desk: the desk band');
        layout('stack');
        assert.ok(!barOnly.attrs.has('data-lead'), 'the stack layout: the base header');
        layout('docked');
        assert.ok(barOnly.attrs.has('data-lead'));
        offBar();
        assert.equal(listeners.size, 0);
        doc.documentElement = { dataset: { telegramFullscreen: 'active' } };
        const update = el('update'), overflow = el('list'), telegramActions = el('app-page-toolbar', update, el('refresh'));
        const telegram = el('app-page-header', el('app-page-title-block', el('app-page-lead')), telegramActions);
        el('page', telegram, overflow);
        update.focus();
        const offTelegram = bindPageHeader(telegram, { secondary: update, overflow });
        assert.deepEqual(names(telegramActions), ['refresh'], 'fullscreen: one page action in the host row');
        assert.deepEqual(names(overflow), ['update'], 'the other action stays reachable in the list');
        assert.equal(doc.activeElement, update);
        doc.documentElement.dataset.telegramFullscreen = 'expanded';
        layout('stack');
        assert.deepEqual(names(telegramActions), ['update', 'refresh'], 'leaving fullscreen restores the same controls');
        offTelegram();
    } finally { delete globalThis.window; delete globalThis.document; }
});

test('the optional toolbar class cannot create an HTML attribute or node', () => {
    const html = renderPageHeader({ title: 'Chat', actionsHtml: '<button>Act</button>', toolbarClass: '\" onclick=\"alert(1)\"><img src=x>' });
    assert.doesNotMatch(html, /class="app-page-toolbar app-page-actions " onclick=/);
    assert.doesNotMatch(html, /<img/);
    assert.match(html, /&quot;/);
    assert.match(html, /&lt;img/);
});

test('a named return occupies one shell slot in both layouts, and clearing releases it', (t) => {
    const { mount, window } = setup(t);
    const page = new NodeStub(); page.className = 'page';
    page.innerHTML = renderPageHeader({ page: 'skills', tabsHtml: '<div role="tablist"></div>' });
    mount.append(page);
    const header = page.querySelector('.app-page-header');
    header.toggleAttribute = (name, on) => on ? header.setAttribute(name, '') : header.removeAttribute(name);
    let notify, top = { back: { label: 'Settings', title: 'Settings › Tools' } }, presses = 0;
    const off = bindPageBack({ layers: { page: () => top, onChange: fn => { notify = fn; return () => {}; }, back: () => presses++ },
        currentHeader: () => header });
    t.after(off);
    const button = header.querySelector('[data-shell-back]');
    for (const layout of ['docked', 'stack', 'docked']) {
        window.ouroLayout = { layout }; notify();
        assert.equal(header.querySelector('[data-shell-back]'), button, 'one retained control, never in the tab rail');
        assert.equal(button.getAttribute('aria-label'), 'Back to Settings › Tools');
        assert.equal(button.querySelector('.shell-back-name').textContent, 'Settings');
    }
    button.dispatchEvent({ type: 'click' }); assert.equal(presses, 1);
    top = null; notify();
    assert.equal(header.querySelector('[data-shell-back]'), null);
    assert.equal(header.hasAttribute('data-return'), false);
});

test('a level shows only the owed jump, then its own parent, in either layout', (t) => {
    const { mount, window } = setup(t), page = new NodeStub();
    page.innerHTML = renderPageHeader({ page: 'settings', leadHtml: '<span class="section-lead-page"><span data-page-return></span></span>' });
    mount.append(page);
    const header = page.querySelector('.app-page-header');
    const level = { id: 'level', page: true, back: { label: 'Models' } };
    let jump = { id: 'jump', page: true, returnTo: {}, back: { label: 'Consciousness' } }, notify;
    const calls = [], layers = { page: () => jump || level, returnPage: () => jump,
        ids: () => ['level', ...(jump ? ['jump'] : [])], get: id => id === 'level' ? level : jump,
        onChange: fn => { notify = fn; return () => {}; }, back: (...args) => calls.push(args) };
    t.after(bindPageBack({ layers, currentHeader: () => header }));
    const first = header.querySelector('[data-shell-back]');
    for (const layout of ['docked', 'stack']) {
        window.ouroLayout = { layout }; notify();
        assert.equal(first.getAttribute('aria-label'), 'Back to Consciousness');
        assert.equal(header.querySelector('[data-shell-level]'), null, 'the jump is the only back control');
    }
    first.dispatchEvent({ type: 'click' });
    assert.deepEqual(calls, [['control', 'jump']]);
    jump = null; notify();
    assert.equal(header.querySelector('[data-shell-back]'), first);
    assert.equal(first.getAttribute('aria-label'), 'Back to Models');
    assert.equal(header.querySelector('[data-shell-level]'), null, 'one control when the jump is spent');
    first.dispatchEvent({ type: 'click' });
    assert.deepEqual(calls.at(-1), ['control', 'level']);
});

test('outside a section column a file or room level above a jump keeps its own Back', (t) => {
    const { mount } = setup(t), page = new NodeStub();
    page.innerHTML = renderPageHeader({ page: 'files' }); mount.append(page);
    const header = page.querySelector('.app-page-header'), jump = { id: 'jump', returnTo: {}, back: { label: 'Settings' } };
    let level = { id: 'file', page: true, back: { label: 'Folder' } }, notify;
    const layers = { page: () => level || jump, returnPage: () => jump,
        ids: () => ['jump', ...(level ? ['file'] : [])], get: id => id === 'jump' ? jump : level,
        onChange: fn => { notify = fn; return () => {}; } };
    t.after(bindPageBack({ layers, currentHeader: () => header }));
    assert.equal(header.querySelector('[data-shell-back]').getAttribute('aria-label'), 'Back to Folder');
    assert.equal(header.querySelector('[data-shell-level]'), null);
    level = null; notify();
    assert.equal(header.querySelector('[data-shell-back]').getAttribute('aria-label'), 'Back to Settings');
});
