/** The one list of places (docs/DESIGN.md "Shell", plan P9/P11).
 *
 * Every place the sidebar offers is declared here once: its id (the `page-<id>` section app.js shows and the
 * `data-nav-page` of its row), the label the row and the stack layout's header title say verbatim, its icon key in
 * page_icons.js, and its zone in the sidebar (the pinned head holds Main; the foot after the Projects the tools, in
 * this order), a tool with its group: the foot's first card holds the sections, its second the system rows.
 * The rows are rendered from this list (app.js), the headers read their titles from it (page_header.js), and the
 * gate lane walks it; index.html keeps no nav markup of its own (tests/test_pages_registry_static.py). */

import { PAGE_ICONS, SHELL_ICONS } from './page_icons.js';
import { statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const PAGES = Object.freeze([
    Object.freeze({ id: 'chat', label: 'Main Chat', icon: 'chat', zone: 'head', hint: 'Main Ouroboros chat', init: 'initChat' }),
    Object.freeze({ id: 'circles', label: 'Circles', icon: 'circles', zone: 'foot', group: 'sections', init: 'initCircles' }),
    Object.freeze({ id: 'files', label: 'Files', icon: 'files', zone: 'foot', group: 'sections', init: 'initFiles' }),
    Object.freeze({ id: 'skills', label: 'Extensions', icon: 'skills', zone: 'foot', group: 'sections', init: 'initSkills' }),
    Object.freeze({ id: 'widgets', label: 'Widgets', icon: 'widgets', zone: 'foot', group: 'sections', init: 'initWidgets' }),
    Object.freeze({ id: 'dashboard', label: 'Dashboard', icon: 'dashboard', zone: 'foot', group: 'system', init: 'initDashboard' }),
    Object.freeze({ id: 'settings', label: 'Settings', icon: 'settings', zone: 'foot', group: 'system', init: 'initSettings' }),
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

/** The foot's two cards (owner 21.11; docs/DESIGN.md "Navigation"): the sections, each in a cell of its own (the cells
 * keep the two-to-a-line columns and let a longer name take the room it needs, style.css), then the system rows over
 * a hairline and the version row, where `#nav-version` (utils.js writes it) and the update's controls
 * (update_status.js) stand. */
export function renderNavFoot() {
    const sections = pagesInZone('foot').filter((page) => page.group === 'sections')
        .map((page) => `<div class="nav-cell">${renderNavRow(page)}</div>`).join('');
    return `<div class="nav-card nav-sections">${sections}</div>`
        + `<div class="nav-card nav-system"><div class="nav-system-rows">${renderNavRows('foot', 'system')}</div>`
        + '<div class="sidebar-foot-meta"><span id="nav-version"></span></div></div>';
}

/** Puts the two cards into a document's foot, once: the document's own `.sidebar-foot`, or one this gives the oldest
 * retained document (v7.7.0, whose rows and lone `#nav-version` stand in the old .sidebar-scroll wrapper with no foot
 * of their own). A retained document already holds the version anchor: in the foot's own meta row (an index.html kept
 * across an update, v7.7.1 to v7.11) or beside the old rows (v7.7.0). The system card adopts that row in place of its
 * own, so the document keeps one `.sidebar-foot-meta` with one `#nav-version`, and the status controller
 * (update_status.js) finds its row through the anchor as usual. Once the cards stand, v7.7.0's own Utilities group —
 * the rows the cards replace — leaves the document, so each destination and its focus stop stay one; Main, the
 * Projects, the version and the old wrapper keep their place. */
export function mountNavFoot(sidebar) {
    let foot = sidebar?.querySelector('.sidebar-foot') || null;
    if (!foot && !sidebar?.querySelector('#nav-version')) return;
    if (!foot) {
        foot = document.createElement('div');
        foot.className = 'sidebar-foot';
        sidebar.appendChild(foot);
    }
    if (foot.querySelector('.nav-card')) return;
    const kept = [...foot.children].find((el) => el.matches('.sidebar-foot-meta') && el.querySelector('#nav-version')) || null;
    const loose = kept ? null : foot.parentElement?.querySelector('#nav-version') || null;
    foot.insertAdjacentHTML('afterbegin', renderNavFoot());
    const fresh = foot.querySelector('.nav-system > .sidebar-foot-meta');
    if (kept) fresh.replaceWith(kept);
    else if (loose) {
        const meta = document.createElement('div');
        meta.className = 'sidebar-foot-meta';
        meta.appendChild(loose);  // the anchor keeps its id; utils.js writes it, update_status.js mounts the check in the row
        fresh.replaceWith(meta);
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
