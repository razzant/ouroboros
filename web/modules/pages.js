/** The one list of places (docs/DESIGN.md "Shell", plan P9/P11).
 *
 * Every place the sidebar offers is declared here once: its id (the `page-<id>` section app.js shows and the
 * `data-nav-page` of its row), the label the row and the stack layout's header title say verbatim, its icon key in
 * page_icons.js, and its zone: the pinned head holds Main; the rail beside the Projects (owner 24.11, «Рельс
 * снаружи») holds the sections at its top and the system rows at its foot, in this order, each with its group; a place
 * whose zone is `door` has no row of its own and opens through a door of the place it names (Extensions is a
 * Settings row under Connections, owner 24.11, «как задумано»), whose row stands chosen while it shows.
 * The rows are rendered from this list (app.js), the headers read their titles from it (page_header.js), and the
 * gate lane walks it; index.html keeps no nav markup of its own (tests/test_pages_registry_static.py). */

import { PAGE_ICONS, SHELL_ICONS } from './page_icons.js';
import { statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const PAGES = Object.freeze([
    Object.freeze({ id: 'chat', label: 'Main Chat', icon: 'chat', zone: 'head', hint: 'Main Ouroboros chat', init: 'initChat' }),
    Object.freeze({ id: 'circles', label: 'Circles', icon: 'circles', zone: 'rail', group: 'sections', init: 'initCircles' }),
    Object.freeze({ id: 'files', label: 'Files', icon: 'files', zone: 'rail', group: 'sections', init: 'initFiles' }),
    Object.freeze({ id: 'skills', label: 'Extensions', icon: 'skills', zone: 'door', door: 'settings', init: 'initSkills' }),
    Object.freeze({ id: 'widgets', label: 'Widgets', icon: 'widgets', zone: 'rail', group: 'sections', init: 'initWidgets' }),
    Object.freeze({ id: 'dashboard', label: 'Dashboard', icon: 'dashboard', zone: 'rail', group: 'system', init: 'initDashboard' }),
    Object.freeze({ id: 'settings', label: 'Settings', icon: 'settings', zone: 'rail', group: 'system', init: 'initSettings' }),
]);

export const ZONES = Object.freeze(['head', 'rail', 'door']);

export function pageById(id) {
    return PAGES.find((page) => page.id === String(id || '')) || null;
}

/** The sidebar row that stands for a place while it shows: its own, or for a place behind a door, the row of the place
 * that holds the door (Extensions shows with Settings chosen). */
export function navRowFor(id) {
    const page = pageById(id);
    return page?.zone === 'door' ? page.door : String(id || '');
}

/** The label a header says in the stack layout: the registry's words, verbatim. */
export function pageLabel(id) {
    return pageById(id)?.label || '';
}

export function pagesInZone(zone) {
    return PAGES.filter((page) => page.zone === zone);
}

/** One navigation row: the place's icon and its label. Main's place is this Ouroboros's own card (owner 12.11, the
 * fifth round, "Карточка и тихие разделы"): its picture with the dot that says the owner's move in Main or a Project
 * (owner 15.11, 6A), its name with the quiet words of the place it opens, Main's count and its room's status mark in
 * the Project rows' mark column, and under them what Main is doing now (app.js paints the dot, the count, the mark,
 * its words for readers and that line). */
export function renderNavRow(page) {
    const main = page.zone === 'head';
    if (main) {
        return `<button class="nav-row nav-row-main nav-ouro" type="button" data-nav-page="${escapeHtml(page.id)}" title="${escapeHtml(page.hint || page.label)}">`
            + `<span class="nav-ouro-ava" aria-hidden="true">${statusMarkHtml('nav-ava-mark')}</span>`
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

export function renderNavRows(zone, group = '') {
    return pagesInZone(zone).filter((page) => !group || page.group === group).map(renderNavRow).join('\n');
}

/** The rail's places (owner 24.11; docs/DESIGN.md "Navigation"): the sections at its top, then at its foot the system
 * rows over the version row, where `#nav-version` (utils.js writes it) and the update's controls (update_status.js)
 * stand. A row is its glyph; its word stays its name (style.css). */
export function renderNavRail() {
    const rows = (group) => pagesInZone('rail').filter((page) => page.group === group).map(renderNavRow).join('');
    return `<div class="nav-rail-sections">${rows('sections')}</div>`
        + `<div class="nav-rail-system">${rows('system')}<div class="sidebar-foot-meta"><span id="nav-version"></span></div></div>`;
}

/** Puts the rail's places into a document, once: into its `.nav-rail`, or into a rail that leads `#primary-sidebar` in
 * DOM order (explicit column 1 in style.css) in place of a retained document's foot (an index.html kept across an
 * update, v7.7.1 to v7.12) or ahead of the oldest one's wrapper (v7.7.0, whose rows and lone `#nav-version` stand with
 * `.sidebar-scroll` and no foot of their own). A retained document already holds the version anchor: in its foot's meta
 * row (v7.7.1 to v7.11) or beside the old rows (v7.7.0). The rail adopts it in place of its own, so the document keeps
 * one `.sidebar-foot-meta` with one `#nav-version`, and the status controller (update_status.js) finds its row through
 * the anchor as usual. The drawer's one close control caps the rail first in DOM order so keyboard and assistive-tech
 * navigation hit it immediately: a newer retained head's `#nav-drawer-close` node is reused and its emptied row leaves,
 * while v7.7.0 (which had none) gets the matching control synthesized before app.js binds it. v7.7.0's own Utilities
 * group (the rows the rail replaces) leaves the document too, so each destination and its focus stop stay one; Main,
 * the Projects, the version and the old wrapper keep their place. */
export function mountNavRail(sidebar) {
    const doc = sidebar?.ownerDocument;
    let rail = sidebar?.querySelector('.nav-rail') || null;
    const foot = rail ? null : sidebar?.querySelector('.sidebar-foot') || null;
    const version = doc?.getElementById('nav-version') || null;
    if (!doc || (!rail && !foot && !version) || rail?.querySelector('[data-nav-page]')) return;
    const kept = version?.closest('.sidebar-foot-meta') || null;
    if (!rail) {
        rail = doc.createElement('div');
        rail.className = 'nav-rail';
        foot?.remove();
        sidebar.prepend(rail);
    } else if (rail.parentElement && rail.parentElement.firstElementChild !== rail) {
        rail.parentElement.prepend(rail);
    }
    rail.insertAdjacentHTML('beforeend', renderNavRail());
    const fresh = rail.querySelector('.nav-rail-system > .sidebar-foot-meta');
    // The anchor keeps its id and its row; utils.js writes it, update_status.js mounts the check in the row.
    if (kept) fresh.replaceWith(kept);
    else if (version) fresh.children[0].replaceWith(version);
    let close = doc.getElementById('nav-drawer-close');
    if (!close) {
        close = doc.createElement('button');
        close.id = 'nav-drawer-close';
        close.className = 'shell-icon-control nav-drawer-close';
        close.setAttribute('type', 'button');
        close.setAttribute('aria-label', 'Close navigation');
        close.setAttribute('aria-controls', 'primary-sidebar');
    }
    if (close.parentElement !== rail || rail.firstElementChild !== close) {
        const row = close.closest?.('.drawer-head');
        rail.prepend(close);
        row?.remove();
    }
    sidebar.querySelector('.nav-utilities')?.remove();
}

/** The head of the Projects list (owner 13.11: no "Projects" word, no collapse, the list starts with its search): the
 * row is Find a project (projects_nav.js puts the field first in it) and "+" (New project) at its end. The header's
 * toggle, its unread total and its one status mark are gone with the collapse they spoke for: every row carries its
 * own mark, and nothing is hidden for a total to stand for. The id is the hook app.js reads. */
export function renderProjectsHeader() {
    return `<button id="nav-projects-add" class="nav-projects-add" type="button" title="New project" aria-label="New project">${SHELL_ICONS.plus}</button>`;
}
