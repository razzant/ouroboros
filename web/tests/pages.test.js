import test from 'node:test';
import assert from 'node:assert/strict';
import { PAGES, ZONES, pageById, pageLabel, pagesInZone, renderNavRow, renderNavRows, renderProjectsHeader } from '../modules/pages.js';
import { PAGE_ICONS, SHELL_ICONS } from '../modules/page_icons.js';
import { statusMarkHtml } from '../modules/status_mark.js';

// The one list of places (P9/P11): it renders the sidebar's rows and the Projects header, and names the
// stack layout's header titles. The parity with app.js's init map and the page sections is pinned statically
// (tests/test_pages_registry_static.py).

test('the registry: unique ids, today\'s order, one place in the head and the tools in the foot', () => {
    assert.deepEqual(PAGES.map((p) => p.id), ['chat', 'files', 'skills', 'widgets', 'dashboard', 'settings']);
    assert.deepEqual(PAGES.map((p) => p.label), ['Main Chat', 'Files', 'Skills', 'Widgets', 'Dashboard', 'Settings']);
    assert.deepEqual(pagesInZone('head').map((p) => p.id), ['chat']);
    assert.deepEqual(pagesInZone('foot').map((p) => p.id), ['files', 'skills', 'widgets', 'dashboard', 'settings']);
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
    assert.deepEqual(ids, ['files', 'skills', 'widgets', 'dashboard', 'settings']);
    const main = renderNavRow(pageById('chat'));
    assert.match(main, /^<button class="nav-row nav-row-main" type="button" data-nav-page="chat" title="Main Ouroboros chat">/);
    // Main's row carries its room's one status mark after the label (app.js paints it, at rest it is hidden) and the
    // mark's words for readers, which the interface language's overlay leaves to their own translation.
    assert.ok(main.endsWith(`<span class="nav-row-label">Main Chat</span><span class="nav-row-words" data-i18n-skip></span><span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></button>`));
    assert.match(statusMarkHtml('nav-mark'), /aria-hidden="true" data-shape="none" data-hue="ink" hidden>/);
    assert.equal((main.match(/<svg/g) || []).length, 1);
    assert.ok(main.includes(PAGE_ICONS.chat));
    const files = renderNavRow(pageById('files'));
    assert.doesNotMatch(files, /nav-row-main|status-mark|nav-row-words/, 'a tool has no status of its own');
    assert.match(files, /<span class="nav-row-label">Files<\/span><\/button>$/);
    assert.match(renderNavRow({ id: 'x', label: '<b>', icon: 'none', zone: 'foot' }), /title="&lt;b&gt;".*>&lt;b&gt;</);
});

test('the Projects header keeps the hooks the shell reads and uses the shared glyphs', () => {
    const html = renderProjectsHeader();
    for (const id of ['nav-projects-toggle', 'nav-projects-count', 'nav-projects-add']) {
        assert.ok(html.includes(`id="${id}"`), id);
    }
    // The toggle is the word and its chevron; its facts (the unread count, then the Projects' one status mark, which
    // app.js paints) follow "+" in Main's number and mark columns, outside the button and hidden from readers, whose
    // name of the toggle carries their words (owner Q1 = B).
    assert.ok(html.includes(`<span class="nav-row-label">Projects</span><span class="nav-chevron" aria-hidden="true">`));
    assert.ok(html.includes(`</button><span class="nav-projects-facts" aria-hidden="true"><span id="nav-projects-count" class="nav-row-meta"></span>`
        + `<span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`));
    assert.doesNotMatch(html, /nav-activity-marker|chat-live-typing|nav-projects-activity/);
    assert.match(html, /aria-expanded="true" aria-controls="nav-projects-list"/);
    assert.ok(html.includes(PAGE_ICONS.projects));
    assert.ok(html.includes(`<span class="nav-chevron" aria-hidden="true">${SHELL_ICONS.chevron}</span>`));
    assert.ok(html.includes(`aria-label="New project">${SHELL_ICONS.plus}</button>`));
    assert.match(html, /<span class="nav-row-label">Projects<\/span>/);
});
