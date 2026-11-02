import test from 'node:test';
import assert from 'node:assert/strict';
import { bindHeaderControlPlacement, bindPageHeader, renderMobileNavToggle, renderPageHeader } from '../modules/page_header.js';

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
    assert.match(tabbed, /<\/h2>\s*<\/div><div class="app-page-lead"><\/div>/, 'the lead slot follows the hidden title');
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
    // Both placements carry one status mark for what waits behind the ☰, hidden until app.js paints it.
    for (const toggle of [renderMobileNavToggle(), renderMobileNavToggle({ inComposer: true })]) {
        assert.match(toggle, /aria-label="Open navigation"[^>]*><svg[^]*<\/svg><span class="status-mark" aria-hidden="true" data-shape="none" data-hue="ink" hidden><i><\/i><i><\/i><i><\/i><\/span><\/button>/);
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
        assert.equal(doc.activeElement, tab, 'the owner\'s focus stays on the moved tab');
        layout('stack');
        assert.deepEqual([strip.parentNode, names(under), lead.children.length, header.attrs.has('data-lead')],
            [under, ['app-tab-strip'], 0, false], 'the stack layout: the same node back under the row');
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
        let resized, disconnected = false;
        doc.defaultView = { ResizeObserver: class {
            constructor(callback) { resized = callback; }
            observe(target) { assert.equal(target, filesHeader); }
            disconnect() { disconnected = true; }
        }, requestAnimationFrame: (callback) => { callback(); return 1; }, cancelAnimationFrame() {} };
        filesHeader.clientWidth = 440;
        const off = bindPageHeader(filesHeader, { lead: location, actions });
        assert.deepEqual([names(filesLead), names(toolbar), names(home)], [['files-breadcrumb'], ['files-browser-actions', 'refresh'], []]);
        actions.focus();
        filesHeader.clientWidth = 175;
        resized([]);
        assert.deepEqual([names(home), names(toolbar)], [['files-breadcrumb', 'files-browser-actions'], ['refresh']],
            'a sidebar resize returns the same controls to their home, even while the window stays docked');
        assert.equal(doc.activeElement, actions);
        filesHeader.clientWidth = 440;
        resized([]);
        assert.equal(location.parentNode, filesLead);
        assert.equal(doc.activeElement, actions);
        layout('stack');
        assert.deepEqual([names(home), names(toolbar)], [['files-breadcrumb', 'files-browser-actions'], ['refresh']]);
        off();
        assert.ok(disconnected);
        // A header with nothing of its own to place (Widgets) binds nothing.
        assert.equal(typeof bindPageHeader(el('app-page-header', el('app-page-title-block', el('app-page-lead')))), 'function');
        assert.equal(listeners.size, 0);
    } finally { delete globalThis.window; delete globalThis.document; }
});

test('the optional toolbar class cannot create an HTML attribute or node', () => {
    const html = renderPageHeader({ title: 'Chat', actionsHtml: '<button>Act</button>', toolbarClass: '\" onclick=\"alert(1)\"><img src=x>' });
    assert.doesNotMatch(html, /class="app-page-toolbar app-page-actions " onclick=/);
    assert.doesNotMatch(html, /<img/);
    assert.match(html, /&quot;/);
    assert.match(html, /&lt;img/);
});
