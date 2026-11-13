/** The one list of places (docs/DESIGN.md "Shell", plan P9/P11).
 *
 * Every place the sidebar offers is declared here once: its id (the `page-<id>` section app.js shows and the
 * `data-nav-page` of its row), the label the row and the stack layout's header title say verbatim, its icon key in
 * page_icons.js, and its zone in the sidebar (the pinned head holds Main; the foot after the Projects the tools, in
 * this order).
 * The rows are rendered from this list (app.js), the headers read their titles from it (page_header.js), and the
 * gate lane walks it; index.html keeps no nav markup of its own (tests/test_pages_registry_static.py). */

import { PAGE_ICONS, SHELL_ICONS } from './page_icons.js';
import { statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const PAGES = Object.freeze([
    Object.freeze({ id: 'chat', label: 'Main Chat', icon: 'chat', zone: 'head', hint: 'Main Ouroboros chat', init: 'initChat' }),
    Object.freeze({ id: 'circles', label: 'Circles', icon: 'circles', zone: 'foot', init: 'initCircles' }),
    Object.freeze({ id: 'files', label: 'Files', icon: 'files', zone: 'foot', init: 'initFiles' }),
    Object.freeze({ id: 'skills', label: 'Skills', icon: 'skills', zone: 'foot', init: 'initSkills' }),
    Object.freeze({ id: 'widgets', label: 'Widgets', icon: 'widgets', zone: 'foot', init: 'initWidgets' }),
    Object.freeze({ id: 'dashboard', label: 'Dashboard', icon: 'dashboard', zone: 'foot', init: 'initDashboard' }),
    Object.freeze({ id: 'settings', label: 'Settings', icon: 'settings', zone: 'foot', init: 'initSettings' }),
]);

export const ZONES = Object.freeze(['head', 'foot']);

export function pageById(id) {
    return PAGES.find((page) => page.id === String(id || '')) || null;
}

/** The label a header says in the stack layout: the registry's words, verbatim. */
export function pageLabel(id) {
    return pageById(id)?.label || '';
}

export function pagesInZone(zone) {
    return PAGES.filter((page) => page.zone === zone);
}

/** One navigation row: the place's icon and its label. Main's place is this Ouroboros's own card (owner 12.11, the
 * fifth round, "Карточка и тихие разделы"): its picture, its name with the quiet words of the place it opens, Main's
 * count and its room's status mark in the Project rows' mark column, and under them what Main is doing now (app.js
 * paints the count, the mark, its words for readers and that line). */
export function renderNavRow(page) {
    const main = page.zone === 'head';
    if (main) {
        return `<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="${escapeHtml(page.id)}" title="${escapeHtml(page.hint || page.label)}">`
            + `<span class="nav-ouro-ava" aria-hidden="true"></span>`
            + `<span class="nav-ouro-text"><span class="nav-ouro-line1">`
            + `<span class="nav-ouro-name" data-i18n-skip>Ouroboros</span>`
            + `<span class="nav-row-label">${escapeHtml(page.label)}</span>`
            + `<span class="nav-row-words" data-i18n-skip></span><span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`
            + `<span class="nav-ouro-work" data-i18n-skip hidden></span></span></button>`;
    }
    const icon = PAGE_ICONS[page.icon] || '';
    return `<button class="nav-row" type="button" data-nav-page="${escapeHtml(page.id)}" title="${escapeHtml(page.hint || page.label)}">`
        + `${icon}<span class="nav-row-label">${escapeHtml(page.label)}</span></button>`;
}

export function renderNavRows(zone) {
    return pagesInZone(zone).map(renderNavRow).join('\n');
}

/** The head of the Projects list. PROTOTYPE (owner 13.11: no "Projects" word, no collapse, the list starts with its
 * search): the row is Find a project (projects_nav.js puts the field first in it) and "+" (New project) at its end.
 * The header's toggle, its unread total and its one status mark are gone with the collapse they spoke for: every row
 * carries its own mark, and nothing is hidden for a total to stand for. The id is the hook app.js reads. */
export function renderProjectsHeader() {
    // PROTOTYPE switch html[data-proto-side-count]: the unread total and the one status mark of the Projects (owner
    // 19.10, B) stand at the end of the search row; without the switch they are not shown.
    return `<span class="nav-projects-facts" aria-hidden="true"><span id="nav-projects-count" class="nav-row-meta"></span>`
        + `<span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`
        + `<button id="nav-projects-add" class="nav-projects-add" type="button" title="New project" aria-label="New project">${SHELL_ICONS.plus}</button>`;
}
