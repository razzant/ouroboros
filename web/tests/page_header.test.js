import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bindHeaderControlPlacement, bindPageBack, bindPageHeader, renderBackControl, renderMobileNavToggle, renderPageHeader } from '../modules/page_header.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';
import { applyPayload } from '../modules/i18n.js';
import { createRoomHeader } from '../modules/chat_header.js';
import { ClassList, installDom, restoreDom } from './chat_dom_fixture.js';

test('Extensions keeps the shared section band; its own box decides, never the window', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    const ui = readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    // The place is a section (ui.css "The section list"): the shared container measures it. Its own
    // container left with the old Skills page, and the views are pages of the section, so no tabs band
    // of its own survives.
    assert.match(ui, /\.section-place \{\n    container: sectionplace \/ inline-size;/);
    assert.doesNotMatch(css, /#page-skills \{ container:/, 'no container of its own: the retired skillspage is gone');
    assert.doesNotMatch(css, /@container skillspage/);
    assert.doesNotMatch(css, /#page-skills[^{]*\.app-page-tabs/, 'no tabs band survives the section');
    assert.match(css, /html\[data-layout="docked"\] \.app-page-header\[data-lead\] \{[\s\S]*?grid-template-areas: "title toolbar"/,
        'a wide column keeps the single existing row');
    // The views own only their search and scroller; the band's end holds the open view's actions, and a
    // registry page hides that end (extensions.js paintLead).
    assert.match(css, /\.extensions-toolbar \{ margin-inline-start: auto; \}/);
    assert.match(css, /\.extensions-toolbar\[hidden\] \{ display: none; \}/);
    assert.match(css, /#page-skills,\n#page-widgets \{\n    flex-direction: column;\n    min-height: 0;\n    overflow: hidden;\n\}/);
    assert.match(css, /#page-skills \.skills-scroll \{ padding-top: 0; \}/);
    assert.match(css, /#page-skills :is\(\.skills-scroll, \.skills-search-chrome\) \{ padding-inline: var\(--page-header-padding-x\); \}/);
    assert.match(css, /\.extensions-views > \.skills-search-chrome \{ margin-top: var\(--space-2\); \}/);
    // A return and a page alone rearrange the band by the place's own box, never the window's width.
    const narrow = css.slice(css.indexOf('.extensions-mark[hidden]'), css.indexOf('.extensions-views[data-settings-arrive'));
    assert.ok(narrow.includes('@container sectionplace (max-width: calc(40rem - 0.02px)) {'), 'the place\'s box decides');
    assert.doesNotMatch(narrow, /@media/, 'the column decides, never the window\'s width');
    assert.match(narrow, /html\[data-layout="stack"\] #page-skills\[data-section-view="page"\] > \.app-page-header \{ grid-template-columns: minmax\(0, 1fr\); grid-template-areas: "lead"; \}/,
        'a page alone is one lead row: its round Back and its own name');
    assert.match(narrow, /#page-skills\[data-section-view="home"\] > \.app-page-header\[data-return\] \.section-lead-page \{ display: flex; \}/);
    assert.match(narrow, /#page-skills\[data-section-view="home"\] > \.app-page-header\[data-return\] \.section-lead-rail \{ display: none; \}/,
        'a return takes the list\'s row');
});

test('a section band has the same inset without native chrome and one leading edge with or without a return', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    const ui = readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    // Without the desktop shell's attribute the band is the browser inset; ui.css never reads the shell's geometry
    // (tests/test_launcher_window_chrome.py keeps that geometry under the attribute).
    assert.match(ui, /--page-header-row-h: calc\(var\(--control-h-float\) \+ 36px\);/);
    assert.doesNotMatch(ui.replace(/\/\*[\s\S]*?\*\//g, ''), /window-chrome/);
    assert.match(css, /:root\[data-window-chrome="integrated"\] \{[^}]*--page-header-row-h: max\(var\(--control-h-float\), var\(--window-chrome-top\)\);/,
        'a measured native band still follows Interface size; only an absent band needs the browser inset');
    assert.match(css, /\.app-page-header\[data-lead\] \{[^}]*min-height: var\(--page-header-row-h\)/);
    assert.match(css, /\.app-page-header\[data-return="column"\] > \.app-page-title-block \{ display: block; \}/);
    assert.match(css, /\.app-page-header\[data-return\] > \.app-page-title-block \{ display: flex;/, 'an ordinary leading return keeps its actual sibling');
    assert.match(css, /\.app-page-lead \.app-tab-strip \{[^}]*padding: 10px 0 26px;/);
    // The Dashboard's band always holds its name on a desk, so no rule lays an empty band over its columns: the
    // header stays in flow on every page (the rendered positions: test_dashboard_heading_stays_put_on_every_page).
    assert.doesNotMatch(css, /html\[data-layout="docked"\] #page-dashboard > \.app-page-header:not\(\[data-controls\]\)/);
    assert.doesNotMatch(css, /#page-dashboard > \.app-page-header\[data-empty\]/, 'no Dashboard rule keys on a measured emptiness');
    // A section place's band keeps no end column (Logs' Clear ended 12px short of the cards), even under a return.
    const rule = 'html[data-layout="docked"] .section-place > .app-page-header[data-lead] { grid-template-columns: minmax(0, 1fr); grid-template-areas: "title"; }';
    assert.ok(css.includes(rule) && css.indexOf(rule) > css.indexOf('html[data-layout="docked"] .app-page-header[data-lead][data-return] {'),
        'after the rule a return sets, so it holds there too');
    // A phone's page row stands at the same inset (18px below a browser's top), under a system band when there is
    // one; a room keeps its own top, and Telegram's row and a macOS window's lights outrank this rule.
    // The drawer's × caps its rail on the page row's one inset, and the head beside it starts on the same line (owner
    // 24.11; the × stood at 6px, ☰ at 18px, before they shared it).
    assert.match(css, /:where\(html\[data-layout="stack"\]:not\(\[data-window-chrome="integrated"\]\)\) :is\(#content > \.page:not\(#page-chat\) > \.app-page-header, \.nav-rail, \.sidebar-head\) \{\s*padding-top: max\(calc\(\(var\(--page-header-row-h\) - var\(--control-h-float\)\) \/ 2\), calc\(var\(--chrome-inset\) \+ var\(--inset-top\)\)\);/);
    assert.match(css, /html\[data-layout="stack"\] \.app-page-header \{[^}]*padding-top: calc\(var\(--chrome-inset\) \+ var\(--inset-top\)\);/,
        'rooms keep the stack row\'s own top');
});

test('header actions use the floating control tokens while Back keeps the round glass recipe', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /\.app-page-header \.btn, \.app-page-tabs \.btn \{[^}]*--control-h: var\(--control-h-float\);[^}]*--button-font-size: var\(--type-meta\)/);
    assert.doesNotMatch(css, /\.app-page-header \.shell-back[^}]*--control-h:/);
});

test('Back is the shell\'s round control; its words are drawn only inside the header-row container', () => {
    const html = renderBackControl({ to: 'Tools', title: 'Settings › Tools', attrs: 'data-shell-back' });
    assert.match(html, /aria-label="Back to Settings › Tools"/);
    assert.doesNotMatch(html, /shell-back-label/, 'its words are written by bindPageBack only for a return to another place');
    // Which at-rules wrap each declaration that lends the capsule its words (a small brace walk over the sheet).
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const lenders = [];
    const stack = [];
    let head = '';
    for (const ch of css) {
        if (ch === '{') { stack.push(head.trim()); head = ''; } else if (ch === '}') { stack.pop(); head = ''; } else {
            head += ch;
            if (/--return-words:\s*block/.test(head)) { lenders.push(stack.join(' / ')); head = ''; }
        }
    }
    assert.equal(lenders.length, 2, 'one for a page row, one for a room bar');
    for (const where of lenders) {
        assert.match(where, /^@container (pagereturn|roomreturn) \(min-width: [\d.]+rem\) \/ html\[data-layout="docked"\]/, where);
    }
    assert.match(css, /\.shell-back\[data-named\] > \.shell-back-label \{[^}]*display: var\(--return-words, none\)/,
        'outside those rows the words are not drawn');
});

test('Telegram sections keep a visible drawer control and one title baseline inside both host buttons', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    const telegram = css.slice(css.indexOf('/* A section shares Telegram'));
    assert.match(telegram, /--telegram-header-start: 108px/);
    assert.match(telegram, /--telegram-header-end: 100px/);
    assert.match(telegram, /> \.app-page-leading \{[^}]*display: flex;/);
    assert.match(telegram, /:is\(\.app-page-title, \.app-page-lead-title\) \{/);
    assert.doesNotMatch(telegram, /> \.app-page-leading \{ display: none;/);
    // Below the width that holds the name (4rem) beside the row's last action, the action leaves the row and the name
    // keeps its place to the host's buttons (at 320 and 350 px it was 0px wide); the row's own width decides, in CSS.
    assert.match(telegram, /> \.app-page-header \{\n    position: relative;\n    container: tgrow \/ inline-size;/);
    const narrow = telegram.slice(telegram.indexOf('@container tgrow (max-width: calc(328px + 4rem - 0.02px)) {'));
    assert.ok(telegram.includes('@container tgrow (max-width: calc(328px + 4rem - 0.02px)) {'), 'the row\'s own query');
    // Out of the row the action stays positioned: its finger's reach (ui.css .ui-hit-area) is measured from its own
    // box, never the header's (a static Refresh took the presses of every Skills tab at 320 and 350 px).
    assert.match(narrow.slice(0, narrow.indexOf('\n}')), /:is\(\.app-page-toolbar > button:last-child, \.logs-clear-list\) \{ position: relative; inset: auto; \}[\s\S]*:is\(\.app-page-title, \.app-page-lead-title\) \{ right: var\(--telegram-header-end\); \}/);
    assert.doesNotMatch(telegram, /button:last-child, \.logs-clear-list\) \{ position: static/);
    // The name ends in an ellipsis: a block whose line is the row's height (a flex box cut "Consciousne" mid-letter).
    assert.match(telegram, /:is\(\.app-page-title, \.app-page-lead-title\) \{[^}]*display: block;\n    line-height: var\(--touch-target-min\);/);
    assert.doesNotMatch(telegram.slice(telegram.indexOf(':is(.app-page-title, .app-page-lead-title) {'), telegram.indexOf('grid-area: auto;\n}', telegram.indexOf(':is(.app-page-title, .app-page-lead-title) {'))), /display: flex/);
    assert.match(telegram, /:is\(\.app-page-toolbar > button:last-child, \.logs-clear-list\) \{\n    position: absolute;/, 'a wide row keeps the action in it');
    // A page's name takes the host row's one title slot: a page alone, and two panes of an automatic view; a list
    // alone (the automatic view in one column) keeps the place's name.
    const hide = (view) => `.section-place[data-section-view="${view}"] > .app-page-header .app-page-title { display: none; }`;
    const two = telegram.slice(telegram.indexOf('@container sectionplace (min-width: 40rem) {'));
    assert.ok(telegram.indexOf('@container sectionplace (min-width: 40rem) {') > 0 && two.slice(0, two.indexOf('\n}')).includes(hide('auto')),
        'two panes of the automatic view hide the place\'s title under the page\'s name');
    assert.ok(telegram.includes(hide('page')));
    assert.equal(telegram.split(hide('auto')).length, 2, 'only inside the two-pane query');
    // Two panes: the header is no size container, so the lead over the rail reads the place's box for 100cqi
    // (ui.css --section-rail-w); as one, Logs' chips stood 24 px left of the rail's end at 1200 px.
    const unsized = '.section-place > .app-page-header { container-type: normal; }';
    assert.ok(two.slice(0, two.indexOf('\n}')).includes(unsized), 'two panes keep the place\'s box');
    assert.equal(telegram.split(unsized).length, 2, 'only inside the two-pane query');
    // One column of the automatic view is the list alone: neither the page's name nor its controls (Logs' chips and
    // Clear) stand in the host row over it; a page alone keeps its lead.
    const lead = (view) => `.section-place[data-section-view="${view}"] > .app-page-header .app-page-lead { display: none; }`;
    const one = telegram.slice(telegram.indexOf('@container sectionplace (max-width: calc(40rem - 0.02px)) {'));
    assert.ok(telegram.includes('@container sectionplace (max-width: calc(40rem - 0.02px)) {') && one.slice(0, one.indexOf('\n}')).includes(lead('auto')),
        'the list alone shows no lead in the host row');
    assert.equal(telegram.split(lead('auto')).length, 2, 'only below the two-pane query');
    assert.ok(!telegram.includes(lead('page')), 'a page alone keeps its name and controls');
    // The controls it does not show keep no room from the place's name (Clear's 96px clipped "Dashboard").
    const room = (view) => `.section-place[data-section-view="${view}"] > .app-page-header { --telegram-title-end: var(--telegram-header-end); }`;
    assert.ok(one.slice(0, one.indexOf('\n}')).includes(room('auto')), 'the list alone gives its name the whole slot');
    assert.ok(!telegram.includes(room('page')), 'a page alone keeps the room its controls take');
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

test('a section keeps one header row with two panes and with a page alone', () => {
    const ui = readFileSync(new URL('../ui.css', import.meta.url), 'utf8');
    const settings = readFileSync(new URL('../settings.css', import.meta.url), 'utf8');
    const header = 'html[data-layout="stack"] .section-place > .app-page-header';
    const two = ui.slice(ui.indexOf('@container sectionplace (min-width: 40rem) {'));
    assert.ok(two.includes(`${header} { grid-template-areas: "leading lead lead"; }`), 'Places and both column leads share a row');
    assert.ok(two.includes(`${header} .app-page-title-row { position: absolute;`), 'the section name stays available to readers without another row');
    assert.match(two, /grid-template-columns: calc\(var\(--section-rail-w\) - var\(--section-rail-end\) - var\(--space-3\) - 44px - var\(--space-2\)\) minmax\(0, 1fr\)/,
        'the page lead stays over its column after the places control takes room from the rail lead');
    const alone = settings.slice(settings.indexOf('@container sectionplace (max-width: calc(40rem - 0.02px)) {'));
    assert.match(alone, /\[data-section-view="home"\].*grid-template-areas: "leading title toolbar"/,
        'a list alone keeps Places and its section name');
    // The named-return band: a page alone is the one "lead" area (its round Back and its name), as is any header owing
    // a return; Settings keeps Search in its list on a touch window, so its two panes keep Places in one row too.
    assert.match(alone, /\[data-section-view="page"\].*grid-template-areas: "lead";/,
        'a page alone puts Back and its own name in that row');
    assert.ok(settings.includes('#page-settings > .app-page-header[data-return] { grid-template-columns: minmax(0, 1fr); grid-template-areas: "lead"; }'),
        'a header that owes a return is that one row too');
    assert.match(settings, /html\[data-layout="stack"\] #page-settings > \.app-page-header \{\s*grid-template-areas: "leading title toolbar";\s*\}/,
        'Settings\' touch header is one row with or without its list beside the page');
    for (const css of [ui, settings]) assert.doesNotMatch(css, /grid-template-areas:\s*"leading title toolbar"\s*"lead lead lead"/,
        'neither shared nor Settings rules can restore the retired second row');
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
        node.classList = new ClassList(node);
        node.classList.add(name);
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
        assert.ok(header.classList.contains('scroll-fade-x'), 'the desk band uses the shared fade');
        assert.equal(doc.activeElement, tab, 'the owner\'s focus stays on the moved tab');
        layout('stack');
        assert.deepEqual([strip.parentNode, names(under), lead.children.length, header.attrs.has('data-lead')],
            [under, ['app-tab-strip'], 0, false], 'the stack layout: the same node back under the row');
        assert.ok(!header.attrs.has('data-scroll-x'), 'the phone header gives the horizontal strip back its ownership');
        assert.equal(header.classList.contains('scroll-fade-x'), false, 'the stack header does not mask the returned strip');
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
        // A header that holds the place's name alone (Widgets, owner 27.11.2026) is the desk band too; nothing moves.
        const nameOnly = el('app-page-header', el('app-page-title-block'));
        const offName = bindPageHeader(nameOnly);
        assert.ok(nameOnly.attrs.has('data-lead'), 'a desk: its name on the band, as Circles\' is');
        layout('stack');
        assert.ok(!nameOnly.attrs.has('data-lead'), 'the stack layout: the base header, ☰ and the name');
        layout('docked');
        offName();
        assert.equal(listeners.size, 0);
        assert.equal(typeof bindPageHeader(null), 'function', 'no header, nothing bound');
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
        assert.deepEqual(names(telegramActions), ['update', 'refresh'], 'fullscreen keeps the same actions together');
        assert.deepEqual(names(overflow), [], 'no isolated action above the list');
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
        assert.equal(button.title, 'Back to Settings › Tools', 'its tooltip says where it leads');
    }
    button.dispatchEvent({ type: 'click' }); assert.equal(presses, 1);
    top = null; notify();
    assert.equal(header.querySelector('[data-shell-back]'), null);
    assert.equal(header.hasAttribute('data-return'), false);
});

test('a place the sidebar names by its glyph alone says its name; Files and Settings keep their own words', () => {
    // Each module's own header call, whatever the order of its fields (the rendered titles are pinned in the browser:
    // tests/test_header_question_a_browser.py::test_a_named_return_yields_to_the_pages_own_name_and_controls).
    const call = (name, page) => {
        const source = readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8');
        const found = [...source.matchAll(/renderPageHeader\(\{([^}]*)\}\)/g)].map(([, body]) => body)
            .find((body) => new RegExp(`page:\\s*'${page}'`).test(body));
        assert.ok(found !== undefined, `${name} draws the shared header of ${page}`);
        return /\bnamed:\s*false\b/.test(found);
    };
    assert.deepEqual(['circles', 'widgets', 'dashboard'].map((page) => call(`${page}.js`, page)), [true, true, true],
        'Circles and Widgets at the band\'s start, the Dashboard over its list');
    assert.deepEqual([['files.js', 'files'], ['settings.js', 'settings'], ['extensions.js', 'skills']].map(([n, p]) => call(n, p)),
        [false, false, false], 'Files\' location, Settings\' search; Extensions keeps Settings chosen in the sidebar');
    const shown = renderPageHeader({ page: 'widgets', named: false });
    assert.doesNotMatch(shown, /data-sidebar-names|data-empty/, 'a visible name: never the hidden title, never an empty band');
    assert.equal((shown.match(/<h2/g) || []).length, 1, 'one heading for readers, the one shown');
    assert.match(renderPageHeader({ page: 'files', actionsHtml: '<button>R</button>' }), /data-sidebar-names/);
});

test('a return to another place names it beside the chevron; a level or a move inside one place keeps the circle', (t) => {
    const { mount } = setup(t), page = new NodeStub();
    page.innerHTML = renderPageHeader({ page: 'dashboard', named: false,
        leadHtml: '<span class="section-lead-page"><span data-page-return></span></span>' });
    mount.append(page);
    const header = page.querySelector('.app-page-header');
    let top = { id: 'jump', page: true, destination: 'dashboard:costs', returnTo: { key: 'settings:' },
        back: { label: 'Settings', title: 'Settings' } }, notify;
    t.after(bindPageBack({ layers: { page: () => top, onChange: fn => { notify = fn; return () => {}; }, back: () => {} },
        currentHeader: () => header }));
    const button = header.querySelector('[data-shell-back]');
    const words = () => button.querySelector('.shell-back-label')?.textContent ?? null;
    assert.equal(words(), 'Settings', 'Settings → Costs: `‹ Settings`');
    assert.ok(button.hasAttribute('data-named') && header.hasAttribute('data-return-named'), 'the page\'s name shows after it');
    assert.equal(button.getAttribute('aria-label'), 'Back to Settings', 'one accessible name, as before');
    top = { ...top, destination: 'dashboard:costs', returnTo: { key: 'settings:consciousness' },
        back: { label: 'Consciousness', title: 'Settings › Consciousness' } }; notify();
    assert.deepEqual(button.querySelector('.shell-back-label').children.map((step) => step.textContent), ['Settings', 'Consciousness'],
        'each step its own words, so the interface language reaches them; style.css draws the ›');
    top = { ...top, returnTo: { key: 'project:p1' }, back: { label: 'Nova', title: 'Nova' } }; notify();
    assert.equal(button.querySelector('.shell-back-label').getAttribute('dir'), 'auto', 'a Project\'s name keeps its own direction');
    // The other direction: a jump between two pages of one place, and a level, keep the round control.
    top = { ...top, destination: 'settings:models', returnTo: { key: 'settings:consciousness' } }; notify();
    assert.equal(words(), null, 'Settings › Consciousness → Settings › Models: one place, the circle');
    assert.ok(!button.hasAttribute('data-named') && !header.hasAttribute('data-return-named'));
    top = { id: 'level', page: true, back: { label: 'Dashboard' } }; notify();
    assert.equal(words(), null, 'a level\'s Back is the circle');
    assert.equal(button.getAttribute('aria-label'), 'Back to Dashboard');
    top = null; notify();
    assert.ok(!header.hasAttribute('data-return-named'), 'a spent return leaves no mark');
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
    assert.deepEqual(calls, [['keyboard', 'jump']]);
    jump = null; notify();
    assert.equal(header.querySelector('[data-shell-back]'), first);
    assert.equal(first.getAttribute('aria-label'), 'Back to Models');
    assert.equal(header.querySelector('[data-shell-level]'), null, 'one control when the jump is spent');
    first.dispatchEvent({ type: 'click' });
    assert.deepEqual(calls.at(-1), ['keyboard', 'level']);
    // A level opened after the jump stands above it: the circle is the top page layer's, so it peels first.
    const later = { id: 'later', page: true, back: { label: 'Light model' } };
    jump = { id: 'jump', page: true, returnTo: {}, back: { label: 'Main' } };
    layers.page = () => later; notify();
    assert.equal(first.getAttribute('aria-label'), 'Back to Light model');
    first.dispatchEvent({ type: 'click' });
    assert.deepEqual(calls.at(-1), ['keyboard', 'later']);
    layers.page = () => jump; notify();
    assert.equal(first.getAttribute('aria-label'), 'Back to Main', 'then the jump');
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

test('the capsule\'s words, its accessible name and its tooltip come from one localized destination', (t) => {
    const { mount } = setup(t), page = new NodeStub();
    page.innerHTML = renderPageHeader({ page: 'circles', named: false });
    mount.append(page);
    const header = page.querySelector('.app-page-header');
    let top = { id: 'jump', page: true, destination: 'circles', returnTo: { key: 'settings:' }, back: { label: 'Settings', title: 'Settings' } }, notify;
    t.after(bindPageBack({ layers: { page: () => top, onChange: fn => { notify = fn; return () => {}; }, back: () => {} }, currentHeader: () => header }));
    // The dictionary alone (the stub DOM has no overlay walker): the shell's overlay is not what this control relies on.
    const load = (payload) => {
        const doc = globalThis.document;
        if (!doc) return applyPayload(payload);
        const { body } = doc;
        doc.body = null;
        try { return applyPayload(payload); } finally { doc.body = body; }
    };
    t.after(() => load({ language: '', english: true, entries: {} }));
    const button = header.querySelector('[data-shell-back]');
    // The language changes while the jump is still owed: the next paint (the shell's ouro:language-changed) says it all.
    load({ language: 'fr', english: false, revision: 1, entries: { Settings: { text: 'Réglages' }, 'Back to {place}': { text: 'Retour à {place}' } } });
    notify();
    assert.equal(button.querySelector('.shell-back-label').textContent, 'Réglages');
    assert.equal(button.getAttribute('aria-label'), 'Retour à Réglages', 'the spoken name agrees with the drawn word');
    assert.equal(button.title, 'Retour à Réglages');
    assert.ok(button.dataset.i18nFmt, 'the page overlay leaves the control alone: it is already translated');
    // A Project's name is the owner's, never looked up.
    top = { ...top, returnTo: { key: 'project:p' }, back: { label: 'Settings', title: 'Settings' } }; notify();
    assert.equal(button.querySelector('.shell-back-label').textContent, 'Settings');
    assert.equal(button.getAttribute('aria-label'), 'Retour à Settings');
});
