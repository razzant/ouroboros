import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PAGES, ZONES, leadHeadWithFind, mountNavRail, navRowFor, pageById, pageLabel, pagesInZone, renderNavRail, renderNavRow, renderNavRows, renderProjectsHeader } from '../modules/pages.js';
import { PAGE_ICONS, SHELL_ICONS } from '../modules/page_icons.js';
import { appendHtml, domDouble } from './dom_double.js';
import { statusMarkHtml } from '../modules/status_mark.js';

// The one list of places (P9/P11): it renders the sidebar's rows and the Projects header, and names the
// stack layout's header titles. The parity with app.js's init map and the page sections is pinned statically
// (tests/test_pages_registry_static.py).

test('the registry: unique ids, today\'s order, Main in the head, the tools in the rail, Extensions behind its door', () => {
    // Circles leads the tools: the one home of the places Ouroboros lives in (DESIGN "Circles").
    assert.deepEqual(PAGES.map((p) => p.id), ['chat', 'circles', 'files', 'skills', 'widgets', 'dashboard', 'settings']);
    assert.deepEqual(PAGES.map((p) => p.label), ['Main Chat', 'Circles', 'Files', 'Extensions', 'Widgets', 'Dashboard', 'Settings']);
    assert.deepEqual(pagesInZone('head').map((p) => p.id), ['chat']);
    assert.deepEqual(pagesInZone('rail').map((p) => p.id), ['circles', 'files', 'widgets', 'dashboard', 'settings']);
    assert.deepEqual(ZONES, ['head', 'rail', 'door']);
    // The rail (owner 24.11, «Рельс снаружи»): the sections at its top, then the system rows at its foot.
    assert.deepEqual(pagesInZone('rail').map((p) => p.group), ['sections', 'sections', 'sections', 'system', 'system']);
    // Extensions has no row of its own (owner 24.11, «как задумано»): it opens through Settings' door, and Settings'
    // row stands for it while it shows; every other place stands for itself.
    assert.deepEqual(pagesInZone('door').map((p) => [p.id, p.door]), [['skills', 'settings']]);
    assert.deepEqual(PAGES.map((p) => navRowFor(p.id)), ['chat', 'circles', 'files', 'settings', 'widgets', 'dashboard', 'settings']);
    assert.equal(navRowFor('nowhere'), 'nowhere');
    for (const page of PAGES) {
        assert.ok(PAGE_ICONS[page.icon], `${page.id} names an icon the registry holds`);
        assert.ok(ZONES.includes(page.zone));
        assert.equal(Boolean(page.door), page.zone === 'door', `${page.id}: only a place behind a door names it`);
        if (page.door) assert.equal(pageById(page.door)?.zone, 'rail', `${page.id}: its door's place has a row`);
        assert.match(page.init, /^init[A-Z]/);
        assert.ok(Object.isFrozen(page));
    }
    assert.equal(pageLabel('chat'), 'Main Chat');
    assert.equal(pageLabel('skills'), 'Extensions', 'a place behind a door keeps its title');
    assert.equal(pageLabel('nowhere'), '');
    assert.equal(pageById('files').label, 'Files');
});

test('rows render from the registry, in order, with the label verbatim and the icon once', () => {
    const rows = renderNavRows('rail');
    const ids = [...rows.matchAll(/data-nav-page="([a-z]+)"/g)].map((m) => m[1]);
    assert.deepEqual(ids, ['circles', 'files', 'widgets', 'dashboard', 'settings']);
    const main = renderNavRow(pageById('chat'));
    // Main's place is this Ouroboros's own card (owner 12.11, fifth round): its picture instead of the chat icon, with
    // the picture's own dot for the owner's move in Main or a Project (owner 15.11, 6A; a mark of its own class, there
    // from the first paint and hidden at rest), its name with the place's label, Main's count and its room's one
    // status mark (app.js paints it, at rest hidden), the mark's words for readers, which the interface language's
    // overlay leaves to their own translation, and under them the line of what Main is doing now.
    assert.match(main, /^<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="chat" title="Main Ouroboros chat">/);
    assert.equal(main, '<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="chat" title="Main Ouroboros chat">'
        + `<span class="nav-ouro-ava" aria-hidden="true">${statusMarkHtml('nav-ava-mark')}</span>`
        + '<span class="nav-ouro-text"><span class="nav-ouro-line1">'
        + '<span class="nav-ouro-name" data-i18n-skip>Ouroboros</span>'
        + `<span class="nav-row-label">Main Chat</span><span class="nav-row-words" data-i18n-skip></span><span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`
        + '<span class="nav-ouro-work" data-i18n-skip hidden></span></span></button>');
    assert.match(statusMarkHtml('nav-mark'), /aria-hidden="true" data-shape="none" data-hue="ink" hidden>/);
    assert.equal((main.match(/class="nav-ouro-ava"/g) || []).length, 1, 'the card shows its picture once');
    assert.deepEqual(main.match(/class="status-mark [\w-]+"/g), ['class="status-mark nav-ava-mark"', 'class="status-mark nav-mark"'],
        'two marks, each of its own class: the picture\'s dot and Main\'s room mark');
    assert.equal((main.match(/<svg/g) || []).length, 0, 'the picture stands where the chat icon was');
    const files = renderNavRow(pageById('files'));
    assert.doesNotMatch(files, /nav-row-main|status-mark|nav-row-words/, 'a tool has no status of its own');
    assert.match(files, /<span class="nav-row-label">Files<\/span><\/button>$/);
    // Each tool's row carries its registry glyph once, and no other page's.
    const glyphs = new Set(pagesInZone('rail').map((page) => PAGE_ICONS[page.icon]));
    assert.equal(glyphs.size, pagesInZone('rail').length, 'every tool has a glyph of its own');
    for (const page of pagesInZone('rail')) {
        const row = renderNavRow(page);
        assert.equal((row.match(/<svg/g) || []).length, 1, `${page.id}: one glyph`);
        assert.ok(row.includes(PAGE_ICONS[page.icon]), `${page.id}: the registry's glyph for it`);
        assert.equal(rows.split(PAGE_ICONS[page.icon]).length - 1, 1, `${page.id}: its glyph once among the rows`);
    }
    assert.match(renderNavRow({ id: 'x', label: '<b>', icon: 'none', zone: 'rail' }), /title="&lt;b&gt;".*>&lt;b&gt;</);
});

test('the Projects header is the New project button alone, with the shell\'s plus glyph', () => {
    // Owner 13.11: no "Projects" word, no collapse, no unread total of the Projects; the row is Find a project
    // (projects_nav.js puts the field first) and "+" at its end. The id is the hook app.js reads.
    const html = renderProjectsHeader();
    assert.equal(html, `<button id="nav-projects-add" class="nav-projects-add" type="button" title="New project" aria-label="New project">${SHELL_ICONS.plus}</button>`);
    assert.doesNotMatch(html, /nav-projects-toggle|nav-projects-count|nav-projects-facts|nav-chevron|status-mark|aria-expanded/);
    assert.doesNotMatch(html, /nav-activity-marker|chat-live-typing|nav-projects-activity/);
    // With its heading gone the list's region keeps a name of its own: "Projects" (index.html).
    const shell = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    const section = shell.match(/<section id="nav-projects"[^>]*>/)?.[0] || '';
    assert.match(section, / aria-label="Projects"/);
});

test('the rail is the registry\'s: the sections at its top, the system rows at its foot; the version row is the column\'s foot', () => {
    const rail = renderNavRail();
    const ids = (html) => [...html.matchAll(/data-nav-page="([a-z]+)"/g)].map((m) => m[1]);
    const [sections, system] = rail.split('<div class="nav-rail-system">');
    assert.equal(sections, `<div class="nav-rail-sections">${['circles', 'files', 'widgets'].map((id) => renderNavRow(pageById(id))).join('')}</div>`);
    assert.deepEqual(ids(system), ['dashboard', 'settings']);
    assert.doesNotMatch(rail, /data-nav-page="skills"/, 'Extensions opens through Settings, not from the rail');
    // The rail's foot holds the system rows alone (owner 25.11): the version never wraps its glyphs.
    assert.match(system, /<\/button><\/div>$/);
    assert.doesNotMatch(rail, /nav-version/, 'the version row is not the rail\'s');
    // The column's foot is one quiet line after the Projects (index.html): utils.js writes #nav-version, update_status.js
    // its check and capsule.
    const shell = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    const foot = shell.indexOf('<div class="sidebar-foot-meta"><span id="nav-version"></span></div>');
    assert.ok(foot > shell.indexOf('id="nav-projects-list"') && foot < shell.indexOf('</nav>'), 'the foot follows the Projects');
    // Names fit by layout alone: the rail draws glyphs, its words stay the rows' names (style.css).
    assert.doesNotMatch(readFileSync(new URL('../modules/pages.js', import.meta.url), 'utf8'), /measureNavTools|ResizeObserver|MutationObserver|createRange/);
});

// mountNavRail runs on the shared DOM double, whose html builder stands in for the engine's insertAdjacentHTML.
function railDocument(markup) {
    const { doc } = domDouble({ parseHtml: true });
    const proto = Object.getPrototypeOf(doc.createElement('div'));
    proto.insertAdjacentHTML ??= function insertAdjacentHTML(where, html) {
        assert.equal(where, 'beforeend');
        appendHtml(this, html);
    };
    const sidebar = appendHtml(doc.createElement('nav'), markup);
    doc.body.append(sidebar);
    return sidebar;
}
// The double reads ids as attributes: `[id=…]`, never `#…`.
const versions = (sidebar) => sidebar.querySelectorAll('[id="nav-version"]');
const kids = (node) => node.children.map((child) => child.className || child.id);

test('Find a project and + are the head\'s first row on every surface, an older document\'s row moved there', () => {
    const shell = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    const head = shell.slice(shell.indexOf('<div class="sidebar-head">'), shell.indexOf('<section id="nav-projects"'));
    assert.match(head.replace(/<!--[\s\S]*?-->/g, ''), /^<div class="sidebar-head">\s*<div class="nav-projects-header"><\/div>/,
        'index.html: the find row leads the head, before the card the registry renders after it');
    const older = railDocument('<div class="sidebar-head"><a class="nav-row nav-row-main"></a></div>'
        + '<section class="nav-section nav-projects"><div class="nav-projects-header"></div><div class="nav-projects-list"></div></section>');
    const row = leadHeadWithFind(older);
    assert.equal(row?.className, 'nav-projects-header');
    assert.deepEqual(kids(older.querySelector('.sidebar-head')), ['nav-projects-header', 'nav-row nav-row-main']);
    assert.deepEqual(kids(older.querySelector('.nav-projects')), ['nav-projects-list']);
    leadHeadWithFind(older);  // a second call keeps the one order
    assert.deepEqual(kids(older.querySelector('.sidebar-head')), ['nav-projects-header', 'nav-row nav-row-main']);
    assert.equal(leadHeadWithFind(railDocument('<div class="sidebar-head"></div>')), null);
});

test('mountNavRail: rail and × lead in DOM order, retained HTML reuses its ×, v7.7.0 synthesizes one', () => {
    const shell = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    assert.ok(shell.indexOf('<div class="nav-rail">') < shell.indexOf('<div class="sidebar-head">'),
        'index.html places .nav-rail (and #nav-drawer-close) before .sidebar-head in DOM order');

    const fresh = railDocument('<div class="nav-rail"><button id="nav-drawer-close" class="shell-icon-control nav-drawer-close"></button></div>'
        + '<div class="sidebar-head"></div><section class="nav-section nav-projects"></section>');
    const freshClose = fresh.querySelector('[id="nav-drawer-close"]');
    mountNavRail(fresh);
    const rail = fresh.querySelector('.nav-rail');
    assert.deepEqual(kids(fresh), ['nav-rail', 'sidebar-head', 'nav-section nav-projects', 'sidebar-foot-meta'],
        'the rail leads in DOM order; the version row is the column\'s foot after the Projects (owner 25.11)');
    assert.deepEqual(kids(rail), ['shell-icon-control nav-drawer-close', 'nav-rail-sections', 'nav-rail-system'], 'the × caps the rail');
    assert.equal(rail.firstElementChild, freshClose, 'the fresh × node is kept in place');
    assert.deepEqual(rail.querySelectorAll('[data-nav-page]').map((row) => row.dataset.navPage), ['circles', 'files', 'widgets', 'dashboard', 'settings']);
    assert.equal(versions(fresh).length, 1);
    assert.ok(fresh.querySelector(':scope > .sidebar-foot-meta > [id="nav-version"]'),
        'the version row is the sidebar\'s own last child, outside the rail');
    mountNavRail(fresh);
    assert.equal(rail.children.length, 3, 'composed once');

    // A document whose .nav-rail sits after .sidebar-head is normalized so the rail and × lead in DOM order.
    const trailing = railDocument('<div class="sidebar-head"></div><section class="nav-section nav-projects"></section>'
        + '<div class="nav-rail"><button id="nav-drawer-close" class="shell-icon-control nav-drawer-close"></button></div>');
    mountNavRail(trailing);
    assert.deepEqual(kids(trailing), ['nav-rail', 'sidebar-head', 'nav-section nav-projects', 'sidebar-foot-meta']);

    // v7.7.1 to v7.12: the drawer's head holds the ×, the foot follows the Projects; until v7.11 it holds the version row.
    const previous = railDocument('<div class="sidebar-head"><div class="drawer-head"><button id="nav-drawer-close" class="nav-drawer-close"></button></div></div>'
        + '<section class="nav-section nav-projects"></section>'
        + '<div class="sidebar-foot"><div class="sidebar-foot-meta"><span id="nav-version"></span></div></div>');
    const kept = previous.querySelector('.sidebar-foot-meta');
    const keptClose = previous.querySelector('[id="nav-drawer-close"]');
    mountNavRail(previous);
    const adopted = previous.querySelector('.nav-rail');
    assert.deepEqual(kids(previous), ['nav-rail', 'sidebar-head', 'nav-section nav-projects', 'sidebar-foot-meta'],
        'the rail leads in DOM order; the retained version row is adopted as the column\'s foot');
    assert.equal(previous.querySelector('.sidebar-foot'), null);
    assert.equal(versions(previous).length, 1, 'one #nav-version');
    assert.equal(previous.querySelector(':scope > .sidebar-foot-meta'), kept, 'the retained version row, adopted');
    assert.equal(previous.querySelector('.nav-rail .sidebar-foot-meta'), null, 'the rail holds no version row (owner 25.11)');
    assert.equal(adopted.firstElementChild, keptClose, 'the retained head\'s × node is reused, not recreated');
    assert.equal(previous.querySelectorAll('[id="nav-drawer-close"]').length, 1, 'one #nav-drawer-close');
    assert.equal(previous.querySelector('.drawer-head'), null, 'its emptied row leaves');

    // v7.7.0: a lone #nav-version beside the old rows, no #nav-drawer-close, and the Utilities group the rail replaces.
    const oldest = railDocument('<div class="sidebar-scroll"><button class="nav-row nav-row-main" data-nav-page="chat"></button>'
        + '<section class="nav-section nav-projects"></section><div class="nav-section nav-utilities">'
        + '<button class="nav-row" data-nav-page="files"></button></div></div><span id="nav-version"></span>');
    const loose = oldest.querySelector('[id="nav-version"]');
    mountNavRail(oldest);
    assert.deepEqual(kids(oldest), ['nav-rail', 'sidebar-scroll', 'sidebar-foot-meta'],
        'the rail leads ahead of the preserved wrapper; the fresh version row is the column\'s foot');
    assert.equal(oldest.querySelector('.nav-utilities'), null);
    assert.equal(versions(oldest).length, 1);
    assert.equal(oldest.querySelector(':scope > .sidebar-foot-meta > [id="nav-version"]'), loose, 'the lone version, adopted into the fresh foot row');
    const synthClose = oldest.querySelector('.nav-rail').firstElementChild;
    assert.equal(synthClose.tagName, 'BUTTON');
    assert.equal(synthClose.id, 'nav-drawer-close', 'v7.7.0 gets a synthesized explicit × first in the rail');
    assert.equal(synthClose.className, 'shell-icon-control nav-drawer-close');
    assert.equal(synthClose.getAttribute('type'), 'button');
    assert.equal(synthClose.getAttribute('aria-label'), 'Close navigation');
    assert.equal(synthClose.getAttribute('aria-controls'), 'primary-sidebar');
    assert.equal(oldest.querySelectorAll('[id="nav-drawer-close"]').length, 1);
    assert.deepEqual(oldest.querySelectorAll('[data-nav-page]').map((row) => row.dataset.navPage),
        ['circles', 'files', 'widgets', 'dashboard', 'settings', 'chat'], 'one set of destinations, rail first');
    mountNavRail(oldest);
    assert.equal(oldest.querySelector('.nav-rail').children.length, 3, 'v7.7.0 mount stays idempotent');

    // A document with neither a rail, a foot nor a version gets nothing.
    const bare = railDocument('<div class="sidebar-head"></div>');
    mountNavRail(bare);
    assert.equal(bare.querySelector('.nav-rail'), null);
});
