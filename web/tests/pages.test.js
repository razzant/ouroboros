import test from 'node:test';
import assert from 'node:assert/strict';
import { PAGES, ZONES, pageById, pageLabel, pagesInZone, renderNavRow, renderNavRows, renderProjectsHeader } from '../modules/pages.js';
import { PAGE_ICONS, SHELL_ICONS } from '../modules/page_icons.js';
import { statusMarkHtml } from '../modules/status_mark.js';

// The one list of places (P9/P11): it renders the sidebar's rows and the Projects header, and names the
// stack layout's header titles. The parity with app.js's init map and the page sections is pinned statically
// (tests/test_pages_registry_static.py).

test('the registry: unique ids, today\'s order, one place in the head and the tools in the foot', () => {
    // Circles leads the tools: the one home of the places Ouroboros lives in (DESIGN "Circles").
    assert.deepEqual(PAGES.map((p) => p.id), ['chat', 'circles', 'files', 'skills', 'widgets', 'dashboard', 'settings']);
    assert.deepEqual(PAGES.map((p) => p.label), ['Main Chat', 'Circles', 'Files', 'Skills', 'Widgets', 'Dashboard', 'Settings']);
    assert.deepEqual(pagesInZone('head').map((p) => p.id), ['chat']);
    assert.deepEqual(pagesInZone('foot').map((p) => p.id), ['circles', 'files', 'skills', 'widgets', 'dashboard', 'settings']);
    assert.deepEqual(ZONES, ['head', 'foot']);
    for (const page of PAGES) {
        assert.ok(PAGE_ICONS[page.icon], `${page.id} names an icon the registry holds`);
        assert.ok(ZONES.includes(page.zone));
        assert.match(page.init, /^init[A-Z]/);
        assert.ok(Object.isFrozen(page));
    }
    assert.equal(pageLabel('chat'), 'Main Chat');
    assert.equal(pageLabel('nowhere'), '');
    assert.equal(pageById('files').label, 'Files');
});

test('rows render from the registry, in order, with the label verbatim and the icon once', () => {
    const rows = renderNavRows('foot');
    const ids = [...rows.matchAll(/data-nav-page="([a-z]+)"/g)].map((m) => m[1]);
    assert.deepEqual(ids, ['circles', 'files', 'skills', 'widgets', 'dashboard', 'settings']);
    const main = renderNavRow(pageById('chat'));
    // Main's place is this Ouroboros's own card (owner 12.11, fifth round): its picture instead of the chat icon, its
    // name with the place's label, Main's count and its room's one status mark (app.js paints it, at rest hidden), the
    // mark's words for readers, which the interface language's overlay leaves to their own translation, and under them
    // the line of what Main is doing now.
    assert.match(main, /^<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="chat" title="Main Ouroboros chat">/);
    assert.equal(main, '<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="chat" title="Main Ouroboros chat">'
        + '<span class="nav-ouro-ava" aria-hidden="true"></span>'
        + '<span class="nav-ouro-text"><span class="nav-ouro-line1">'
        + '<span class="nav-ouro-name" data-i18n-skip>Ouroboros</span>'
        + `<span class="nav-row-label">Main Chat</span><span class="nav-row-words" data-i18n-skip></span><span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`
        + '<span class="nav-ouro-work" data-i18n-skip hidden></span></span></button>');
    assert.match(statusMarkHtml('nav-mark'), /aria-hidden="true" data-shape="none" data-hue="ink" hidden>/);
    assert.equal((main.match(/class="nav-ouro-ava"/g) || []).length, 1, 'the card shows its picture once');
    assert.equal((main.match(/<svg/g) || []).length, 0, 'the picture stands where the chat icon was');
    const files = renderNavRow(pageById('files'));
    assert.doesNotMatch(files, /nav-row-main|status-mark|nav-row-words/, 'a tool has no status of its own');
    assert.match(files, /<span class="nav-row-label">Files<\/span><\/button>$/);
    // Each tool's row carries its registry glyph once, and no other page's.
    const glyphs = new Set(pagesInZone('foot').map((page) => PAGE_ICONS[page.icon]));
    assert.equal(glyphs.size, pagesInZone('foot').length, 'every tool has a glyph of its own');
    for (const page of pagesInZone('foot')) {
        const row = renderNavRow(page);
        assert.equal((row.match(/<svg/g) || []).length, 1, `${page.id}: one glyph`);
        assert.ok(row.includes(PAGE_ICONS[page.icon]), `${page.id}: the registry's glyph for it`);
        assert.equal(rows.split(PAGE_ICONS[page.icon]).length - 1, 1, `${page.id}: its glyph once among the rows`);
    }
    assert.match(renderNavRow({ id: 'x', label: '<b>', icon: 'none', zone: 'foot' }), /title="&lt;b&gt;".*>&lt;b&gt;</);
});

test('the Projects header is the New project button alone, with the shell\'s plus glyph', () => {
    // Owner 13.11: no "Projects" word, no collapse, no unread total of the Projects; the row is Find a project
    // (projects_nav.js puts the field first) and "+" at its end. The id is the hook app.js reads.
    const html = renderProjectsHeader();
    assert.equal(html, `<button id="nav-projects-add" class="nav-projects-add" type="button" title="New project" aria-label="New project">${SHELL_ICONS.plus}</button>`);
    assert.doesNotMatch(html, /nav-projects-toggle|nav-projects-count|nav-projects-facts|nav-chevron|status-mark|aria-expanded/);
    assert.doesNotMatch(html, /nav-activity-marker|chat-live-typing|nav-projects-activity/);
});
