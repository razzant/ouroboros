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

/** One navigation row: the place's icon and its label. Main's row carries the class the shell pins and its room's
 * status mark, in the Project rows' mark column, with the mark's words for readers (app.js paints both). */
export function renderNavRow(page) {
    const main = page.zone === 'head';
    const icon = PAGE_ICONS[page.icon] || '';
    const status = main ? `<span class="nav-row-words" data-i18n-skip></span><span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span>` : '';
    return `<button class="nav-row${main ? ' nav-row-main' : ''}" type="button" data-nav-page="${escapeHtml(page.id)}" title="${escapeHtml(page.hint || page.label)}">`
        + `${icon}<span class="nav-row-label">${escapeHtml(page.label)}</span>${status}</button>`;
}

export function renderNavRows(zone) {
    return pagesInZone(zone).map(renderNavRow).join('\n');
}

/** The Projects header of the middle zone: the toggle (icon, label, chevron), "+", and the toggle's facts (the unread
 * count and the Projects' one status mark) in Main's number and mark columns; the toggle's name carries their words.
 * The ids are the hooks app.js and its tests read. */
export function renderProjectsHeader() {
    return `<button id="nav-projects-toggle" class="nav-section-toggle" type="button" aria-expanded="true" aria-controls="nav-projects-list">`
        + `<span class="nav-row-icon" aria-hidden="true">${PAGE_ICONS.projects}</span>`
        + `<span class="nav-row-label">Projects</span>`
        + `<span class="nav-chevron" aria-hidden="true">${SHELL_ICONS.chevron}</span>`
        + `</button>`
        + `<button id="nav-projects-add" class="nav-projects-add" type="button" title="New project" aria-label="New project">${SHELL_ICONS.plus}</button>`
        + `<span class="nav-projects-facts" aria-hidden="true"><span id="nav-projects-count" class="nav-row-meta"></span>`
        + `<span class="nav-mark-slot">${statusMarkHtml('nav-mark')}</span></span>`;
}
