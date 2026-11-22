/** Extensions (docs/DESIGN.md §10 "A section with pages").
 * The sidebar place whose id stays `skills` (`#skills`, `#page-skills`, `ouro:open-skills`) holds
 * what Ouroboros can use beyond its core, as the rows of one list: Skills (the installed skills), MCP servers, then the
 * skill catalog's Browse and Sources. The section's box decides the layout as in Settings: from 40rem the list is the
 * rail beside the open page (never an empty page: the page last opened on this device, Skills the first time), below
 * it the list alone and a pressed row's page as a level with Back.
 *
 * Its pages draw themselves. Skills, Browse and Sources are the three views of skills.js, one block kept mounted
 * (`viewsHtml`; `views.show(view)` and `views.hide()` say when one is the page on screen). MCP servers (`tools`), a
 * server's page (`tools/<id>`, `tools/new`), a skill's own form (`skill/<name>`), the list of skill forms (`skill`) and
 * the skill sources (`skill-sources`) are pages of the page registry (mcp_settings.js, settings_extension_forms.js),
 * opened by the levels Settings opens its pages with (section_levels.js): a server's page over MCP servers, a skill's
 * form over Skills, the skill sources over Sources. Those old addresses asked of Settings arrive here (settings.js).
 * A door of these pages to the rest of Settings opens it with Back to Extensions. Nothing here reads a catalog or
 * starts a server: the rows' answers come from what skills.js has read and from the passive MCP status read. */

import { fmt } from './i18n.js';
import { sectionHoldsTwoPanes } from './layout.js';
import { renderPageHeader } from './page_header.js';
import { fluentIcon } from './page_icons.js';
import { pageLabel } from './pages.js';
import { bindScrollFade, revealPastFade } from './scroll_fade.js';
import { createSectionLevels, isVisible as visible } from './section_levels.js';
import { readSettings } from './settings_controls.js';
import { resetSecretReveals } from './settings_secrets.js';
import { extensionsPageOf, isDoorId, settingsPageFor } from './settings_pages.js';
import { markView, paintStatusMark, statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
// The registry pages this section opens: each registers its family when it is imported (settings_pages.js).
import { configuredServerCount, mcpNeedsSignIn, mcpServerOn, refreshStatus, retireMcpStatusRead } from './mcp_settings.js';
import './settings_extension_forms.js';

const PLACE = 'skills';   // the registry id (pages.js), the address every other module already uses
const LAST_PAGE_KEY = 'ouroboros.extensions.last-page';
const IN_ICON = fluentIcon('chevron-right', 14, 'regular');

/** The section's first-level pages in the list's order: what this install has, then where more skills come from.
 *  `id` is the page's address; the skill views are `installed`, `browse` and `sources`, MCP servers its registry page. */
export const EXTENSIONS_PAGES = Object.freeze([
    Object.freeze({ id: 'installed', title: 'Skills', group: '', glyph: 'hat-graduation' }),
    Object.freeze({ id: 'tools', title: 'MCP servers', group: '', glyph: 'plug-connected' }),
    Object.freeze({ id: 'browse', title: 'Browse', group: 'Skill catalog', glyph: 'globe' }),
    Object.freeze({ id: 'sources', title: 'Sources', group: 'Skill catalog', glyph: 'link' }),
]);
const VIEWS = new Set(['installed', 'browse', 'sources']);
export { extensionsPageOf };

/** The MCP servers row's answer from the passive status read (`GET /api/mcp/status`): a server whose configuration
 *  could not be loaded has no runtime facts and is not on; a server that waits for a sign-in is the owner's move. */
export function mcpServersAnswer(tools) {
    if (tools === null) return { text: fmt('Not checked'), owner: false };
    if (!tools || typeof tools !== 'object') return null;
    if (tools.enabled === false) return { text: fmt('Off'), owner: false };
    const servers = Array.isArray(tools.servers) ? tools.servers : [];
    if (!servers.length) return { text: fmt('None yet'), owner: false };
    const on = servers.filter(mcpServerOn).length;
    const waiting = servers.filter(mcpNeedsSignIn).length;
    const words = fmt('{on} of {n} on', { on, n: configuredServerCount(servers) });
    return { text: waiting ? `${words} · ${fmt(waiting === 1 ? '{n} needs sign-in' : '{n} need sign-in', { n: waiting })}` : words,
        owner: waiting > 0 };
}

/** The Skills row's answer: how many are installed and how many wait for the owner (the Needs you filter's count). */
export function installedAnswer({ installed = null, needs = 0 } = {}) {
    if (!Number.isFinite(installed)) return null;
    if (!installed) return { text: fmt('None yet'), owner: false };
    const words = fmt('{n} installed', { n: installed });
    return { text: needs ? `${words} · ${fmt(needs === 1 ? '{n} needs you' : '{n} need you', { n: needs })}` : words, owner: needs > 0 };
}

function readAnswer(answer, failed) {
    if (!failed) return answer;
    return { text: answer ? fmt('Last known: {answer}', { answer: answer.text }) : fmt('Could not read'), owner: false };
}

function rowHtml(row) {
    return `<button type="button" class="section-row" data-extensions-open="${escapeHtml(row.id)}">`
        + `<span class="section-row-glyph" aria-hidden="true">${fluentIcon(row.glyph, 18)}</span>`
        + `<span class="section-row-words"><span class="section-row-title">${escapeHtml(row.title)}</span>`
        + '<span class="section-row-answer" data-extensions-answer data-i18n-skip></span></span>'
        + `<span class="section-row-end" aria-hidden="true">${statusMarkHtml('extensions-mark')}${IN_ICON}</span></button>`;
}

function listHtml() {
    return [...new Set(EXTENSIONS_PAGES.map((row) => row.group))].map((group) => '<section class="section-group">'
        + `${group ? `<h3 class="section-group-title">${escapeHtml(group)}</h3>` : ''}<div class="ui-card section-rows">`
        + `${EXTENSIONS_PAGES.filter((row) => row.group === group).map(rowHtml).join('')}</div></section>`).join('');
}

function readLastFirst() {
    try {
        const saved = String(globalThis.localStorage?.getItem(LAST_PAGE_KEY) || '');
        return EXTENSIONS_PAGES.some((row) => row.id === saved) ? saved : 'installed';
    } catch { return 'installed'; }
}

/** A page's name: a first-level page's own, else its registry page's words (a server's name, a skill's form). */
export function extensionsTitle(path) {
    const first = EXTENSIONS_PAGES.find((row) => row.id === path);
    if (first) return first.title;
    const found = settingsPageFor(path);
    const named = typeof found?.title === 'function' ? found.title(path) : found?.title;
    return String(named || path);
}

/**
 * Draw the section into `#content` and take its presses. `viewsHtml` is the skill views' block and `toolbarHtml` their
 * actions for the band's end; `views` is `{ show(view), hide(), placeShown() }` (placeShown: the section came on screen,
 * so its rows' facts are worth reading); `facts()` answers `{ installed, needs, working, sources }` with a failure flag
 * for each listing (`installedFailed`, `sourcesFailed`) from skills.js. Answers open, capture/restore and disposal.
 */
export function createExtensions({ ctx = {}, viewsHtml = '', toolbarHtml = '', views = {}, facts = () => ({}) } = {}) {
    const { state, ws, layers = null, showPage, openSettings, openDashboardTab, setBeforePageLeave } = ctx;
    const label = pageLabel(PLACE);
    const page = document.createElement('div');
    page.id = 'page-skills';  // PLACE's section, named literally for the registry's parity guard
    page.className = 'page app-page-glass section-place extensions-place';
    page.dataset.sectionView = 'home';
    // The band: over the list nothing; the shell's return slot and the page's name; at its end the open
    // skill view's own actions, where Refresh has always stood.
    const leadHtml = `<div class="section-lead extensions-lead">
            <span class="section-lead-col section-lead-rail"></span>
            <span class="section-lead-col section-lead-page"><span class="app-page-return" data-page-return></span><h2 class="app-page-lead-title" data-extensions-lead-name></h2><span class="app-page-toolbar app-page-actions extensions-toolbar">${toolbarHtml}</span></span>
        </div>`;
    page.innerHTML = `${renderPageHeader({ page: 'skills', leadHtml })}
        <div class="settings-shell section-shell extensions-shell" data-section-view="home">
            <nav class="section-rail scroll-fade-y" aria-label="${escapeHtml(label)}" data-extensions-rail><div class="section-list">${listHtml()}</div></nav>
            <div class="settings-levels section-pane" data-extensions-pane>
                <section class="extensions-views" data-extensions-views tabindex="-1" hidden>${viewsHtml}</section>
            </div>
        </div>`;
    document.getElementById('content').appendChild(page);

    const shell = page.querySelector('.extensions-shell');
    const rail = page.querySelector('[data-extensions-rail]');
    const pane = page.querySelector('[data-extensions-pane]');
    const viewsEl = page.querySelector('[data-extensions-views]');
    const header = page.querySelector('.app-page-header');
    const toolbar = header.querySelector('.app-page-toolbar');
    const leadName = page.querySelector('[data-extensions-lead-name]');
    const rows = () => [...rail.querySelectorAll('[data-extensions-open]')];
    const shown = () => state?.activePage === PLACE;
    const panes = () => sectionHoldsTwoPanes(page);

    const pageCtx = { openSettings: (path, options) => fromPage(path, options), openDashboardTab, showPage, layers, ws, state,
        readSettings, onScreen: (path) => sectionLevels.onScreen(path) };
    const sectionLevels = createSectionLevels({ place: PLACE, pane, layers, shown, ctx: pageCtx, title: extensionsTitle,
        own: (path) => (VIEWS.has(path) ? { el: viewsEl } : null), onTitled: () => paintLead() });
    const { levels, top } = sectionLevels;
    const beside = () => levels.length > 0 && visible(rail);
    const rowsOnScreen = () => shown() && !document.hidden && visible(rail);
    let railShown = false;

    let lastFirst = readLastFirst();
    let opening = false;
    let landedBy = null;  // only the levels opened by this jump belong below its return
    let tools;          // the MCP status as last read (null: the read failed)
    let viewOn = '';    // the skill view skills.js was told is on screen
    // Reading places: a first-level page stands where its owner left it when it comes back into view (WebKit drops the
    // place of a box that was hidden). The three skill views share one scroller, so their places are kept by view.
    const viewScroller = viewsEl.querySelector('.skills-scroll');
    const viewPlaces = new Map();
    const pagePlaces = new WeakMap();
    let railPlace = 0;
    let placed = null;
    const scrollerOf = (level) => (level?.own ? viewScroller : level?.el) || null;
    const placeOf = (level) => (level.own ? viewPlaces.get(level.path) : pagePlaces.get(level.el)) || 0;

    /* ---- Levels -------------------------------------------------------------------------------------------------- */
    function open(path, { row = '', under = levels.length, opener = null, quiet = false, remember = true, jump = landedBy } = {}) {
        if (remember && !under && path !== lastFirst) {
            lastFirst = path;
            try { globalThis.localStorage?.setItem(LAST_PAGE_KEY, path); } catch { /* storage blocked: kept for this load */ }
        }
        const level = sectionLevels.open(path, { row, jump, under, opener });
        paint({ row, opened: level, quiet });
        return level;
    }

    function closeLevels(keep, { row = '', quiet = false } = {}) {
        const left = sectionLevels.closeLevels(keep);
        paint({ row, returned: left, quiet });
    }

    function backFrom(level) {
        if (level === top()) closeLevels(levels.length - 1);
    }

    // With two panes and no page open, the page last opened on this device comes back (Skills the first time).
    function land() {
        if (!panes() || levels.length || opening || !shown()) return;
        // Widening a list reached by a jump keeps the landing under that same return. No new navigation occurs.
        const pending = layers?.returnPage?.();
        const jump = pending && [PLACE, `${PLACE}:`].includes(pending.destination) ? pending.id : null;
        const arrive = () => open(lastFirst, { under: 0, jump, quiet: true, remember: false });
        if (layers) layers.batch(arrive);
        else arrive();
    }

    /* ---- Painting ------------------------------------------------------------------------------------------------ */
    function paintLead() {
        const level = top();
        if (leadName) leadName.textContent = level ? extensionsTitle(level.path) : label;
        page.dataset.extensionsDepth = String(levels.length);
        // The band's end holds the open skill view's own actions; MCP servers and an entry's page keep theirs inside.
        if (toolbar) toolbar.hidden = !level?.own;
    }

    // skills.js hears which of its views is the page on screen; covered, left or put away, none is.
    function syncViews() {
        const level = top();
        const want = shown() && level?.own ? level.path : '';
        if (want === viewOn) return;
        viewOn = want;
        if (want) views.show?.(want);
        else views.hide?.();
    }

    // Local levels share the shell's stack; a jump's landing stays below its return.
    function syncLayers() {
        sectionLevels.layer(beside(), backFrom);
    }

    function paint({ row = '', opened = null, returned = null, quiet = false } = {}) {
        const level = top();
        const view = level ? 'page' : 'home';
        page.dataset.sectionView = shell.dataset.sectionView = view;
        sectionLevels.showTop();
        const first = levels[0]?.path || '';
        rows().forEach((each) => {
            if (each.dataset.extensionsOpen === first) each.setAttribute('aria-current', 'page');
            else each.removeAttribute('aria-current');
        });
        paintLead();
        syncViews();  // the shared scroller needs the arriving panel's height before its place can be restored
        const showRail = rowsOnScreen();
        if (showRail && !railShown) queueMicrotask(() => { if (rowsOnScreen()) views.rowsShown?.(); });
        railShown = showRail;
        // The page that has just come (back) into view stands where it was left; one never opened starts at its top.
        const box = scrollerOf(level);
        if (box && level !== placed) box.scrollTop = placeOf(level);
        if (!level || beside()) rail.scrollTop = railPlace;
        placed = level;
        syncLayers();
        if (!level) queueMicrotask(land);
        if (opened && beside()) revealPastFade(rail, rows().find((each) => each.dataset.extensionsOpen === first) || rail);
        // A page opened for one of its rows gives the focus to that row's control (revealRow), else to Back or the page.
        if (row && level) sectionLevels.revealRow(level, row);
        if (quiet) return;
        if (opened && !row) opened.el.focus?.({ preventScroll: true });
        else if (returned) {
            const opener = typeof returned.opener === 'function' ? returned.opener() : returned.opener;
            const to = visible(opener) ? opener
                : level ? level.el
                    : rows().find((each) => each.dataset.extensionsOpen === extensionsPageOf(returned.path));
            to?.focus?.({ preventScroll: true });
        }
        sectionLevels.announce();
    }

    /* ---- Answers: each row's current answer, from reads already made ------------------------------------------- */
    function paintAnswers() {
        let read = {};
        try { read = facts() || {}; } catch { /* a row keeps what it showed */ }
        const answers = {
            installed: readAnswer(installedAnswer(read), read.installedFailed),
            tools: tools === undefined ? null : mcpServersAnswer(tools),
            browse: null,
            sources: readAnswer(Number.isFinite(read.sources) ? { text: fmt('{n} connected', { n: read.sources }), owner: false } : null,
                read.sourcesFailed),
        };
        rows().forEach((row) => {
            const id = row.dataset.extensionsOpen;
            const answer = answers[id];
            const words = row.querySelector('[data-extensions-answer]');
            const text = answer?.text || '';
            if (words.textContent !== text) words.textContent = text;
            // Work in flight on an installed skill keeps the moving mark the Installed tab's count stood for.
            paintStatusMark(row.querySelector('.extensions-mark'),
                markView({ mark: id === 'installed' && !read.installedFailed && read.working ? 'moving' : 'none', owner: Boolean(answer?.owner) }));
        });
    }

    const readTools = () => refreshStatus({ join: true });

    /* ---- Moving ---------------------------------------------------------------------------------------------------- */
    /** Open a page inside this section; external entry uses openFromOutside and the shell's jump. */
    function go(path = '', { row = '', opener = null, quiet = false } = {}) {
        const target = String(path || '');
        if (!target) {
            if (!panes()) closeLevels(0, { quiet });
            else if (levels.length) closeLevels(1, { quiet: true });
            else open(lastFirst, { under: 0, quiet: true, remember: false });
            return true;
        }
        const first = extensionsPageOf(target);
        if (!first) return false;
        if (target === first) {
            if (levels[0]?.path !== first) open(first, { under: 0, row, opener, quiet });
            else closeLevels(1, { row, quiet });
            return true;
        }
        if (levels[0]?.path !== first) open(first, { under: 0, quiet: true });
        const at = sectionLevels.indexOf(target);
        if (at < 0) open(target, { under: underOf(target), row, opener, quiet });
        else {
            if (opener) levels[at].opener = opener;
            closeLevels(at + 1, { row, quiet });
        }
        return true;
    }

    // A page lies over the deepest open level whose address it extends (a skill's form over the list of skill forms),
    // else over its first-level page: another server's page replaces the one open, never stacks on it.
    function underOf(target) {
        for (let at = levels.length - 1; at > 0; at -= 1) if (target.startsWith(`${levels[at].path}/`)) return at + 1;
        return 1;
    }

    // A press on one of these pages that leads somewhere: a page of Extensions opens here; a door leaves for its place
    // with Back to Extensions; any other Settings page opens in Settings and leads back here.
    function fromPage(path = '', options = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, options);
        if (extensionsPageOf(target)) return go(target, options);
        return openSettings?.(target, options);
    }

    async function leave(door, { row = '', ...options } = {}) {
        const [kind, name] = String(door).split(':');
        if (kind === 'skills') return go(String(door).slice('skills:'.length), { row });
        if (kind === 'dashboard') return openDashboardTab?.(name, { row, ...options });
        return showPage?.(kind === 'main' ? 'chat' : name, options);
    }

    /** From outside (app.js `openExtensions`, Settings' door, `ouro:open-skills`): the section is shown, then `path`. */
    async function openFromOutside(path = '', { row = '', ...options } = {}) {
        const target = String(path || '');
        if (target && !extensionsPageOf(target)) return false;
        opening = true;
        try {
            // showPage captures the pressed opener synchronously, before its leave guard awaits (WebKit).
            return await showPage(PLACE, { ...options, row, key: `${PLACE}:${target}`, enter: () => {
                const jump = layers?.returnPage?.();
                landedBy = jump?.destination === `${PLACE}:${target}` ? jump.id : null;
                const at = sectionLevels.indexOf(target);
                // Reused ancestors are part of this landing too. A previously hidden Back for one of them
                // must not appear above the return when the two-pane section narrows.
                if (landedBy) {
                    let keep = at + 1;
                    if (at < 0 && target && levels[0]?.path === extensionsPageOf(target)) keep = underOf(target);
                    if (!target && panes()) keep = 1;
                    for (const level of levels.slice(0, keep)) level.jump = landedBy;
                }
                try { go(target, { row, quiet: true }); }
                finally { landedBy = null; }
                paint({ quiet: true });
                sectionLevels.announce();
                void readTools();
                views.placeShown?.();
            } });
        } finally { opening = false; }
    }

    /** The section's row pressed in the sidebar while it is shown (app.js), as Settings': beside the list a deeper level
     *  returns to its first-level page; in one column the list comes back. */
    function pressedAgain() {
        if (!shown()) return false;
        if (!panes()) { if (levels.length) closeLevels(0); }
        else if (levels.length > 1) closeLevels(1);
        return true;
    }

    /* ---- Wiring ---------------------------------------------------------------------------------------------------- */
    const onRow = (event) => {
        const row = event.target.closest?.('[data-extensions-open]');
        if (!row || !rail.contains(row)) return;
        const id = row.dataset.extensionsOpen;
        // A second press of the chosen row keeps its page beside the list, and returns to it from a level over it.
        if (beside() && levels[0]?.path === id) { if (levels.length > 1) closeLevels(1); return; }
        open(id, { under: 0, opener: row });
    };
    rail.addEventListener('click', onRow);
    const onScroll = (event) => {
        const box = event.target;
        if (box === rail) { railPlace = box.scrollTop; return; }
        const level = top();
        if (!level || box !== scrollerOf(level)) return;
        if (level.own) viewPlaces.set(level.path, box.scrollTop); else pagePlaces.set(level.el, box.scrollTop);
    };
    shell.addEventListener('scroll', onScroll, { capture: true, passive: true });
    const onPageShown = (event) => {
        if (event.detail?.page !== PLACE) {
            railShown = false;
            syncViews();
            syncLayers();
            return;
        }
        if (opening || event.detail.reveal) return;  // the jump's enter/restore announces its actual destination
        paint({ quiet: true });
        sectionLevels.announce();
        void readTools();
        views.placeShown?.();
    };
    const onTools = (event) => {
        tools = event.detail;
        paintAnswers();
        paintLead();   // a server saved under a new name is named so in the band at once
    };
    // The installed listing is not asked for here: skills.js hears the tab's return and the socket's reopening itself
    // and reads it once for the rows and the views alike (its hint); asking too would start a second read.
    const onVisible = () => {
        if (!document.hidden && shown()) {
            void readTools();
            sectionLevels.announce();
        }
    };
    window.addEventListener('ouro:page-shown', onPageShown);
    window.addEventListener('ouro:mcp-status-updated', onTools);
    document.addEventListener('visibilitychange', onVisible);
    const offReconnect = ws?.on?.('open', (event) => {
        retireMcpStatusRead(event);
        if (event?.previouslyConnected) onVisible();
    });
    // A width that brings the list beside the page, or takes it away: the layers, the band and landing follow. Repaint
    // at the next frame, never inside the observer's callback.
    let frame = 0;
    let twoBefore = null;
    const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
        frame ||= requestAnimationFrame(() => {
            frame = 0;
            const now = panes();
            if (now !== twoBefore) { twoBefore = now; paint({ quiet: true }); }
        });
    }) : null;
    resize?.observe(shell);
    const disposeFade = bindScrollFade(rail);
    // Leaving Extensions puts every revealed secret of its pages (an MCP server's credentials) away again.
    const offLeave = setBeforePageLeave?.(({ from }) => {
        if (from === PLACE) resetSecretReveals(page);
        return true;
    });
    const onPageHide = () => resetSecretReveals(page);
    window.addEventListener('pagehide', onPageHide);
    paint({ quiet: true });

    return {
        page, open: openFromOutside, go, fromPage, pressedAgain, paintAnswers, rowsOnScreen,
        capture(opener) {
            const snapshot = sectionLevels.capture();
            const origin = rail.contains(opener) ? [] : levels;
            return { ...snapshot, key: `${PLACE}:${top()?.path || ''}`,
                label: origin.length ? extensionsTitle(origin.at(-1).path) : label,
                title: [label, ...origin.map(level => extensionsTitle(level.path))].join(' › '),
                restore() {
                    snapshot.restore();
                    paint({ quiet: true });
                    sectionLevels.announce();
                    void readTools();
                    views.placeShown?.();
                } };
        },
        destroy() {
            sectionLevels.destroy();
            disposeFade();
            offLeave?.();
            window.removeEventListener('pagehide', onPageHide);
            offReconnect?.();
            resize?.disconnect();
            if (frame) cancelAnimationFrame(frame);
            rail.removeEventListener('click', onRow);
            shell.removeEventListener('scroll', onScroll, { capture: true });
            window.removeEventListener('ouro:page-shown', onPageShown);
            window.removeEventListener('ouro:mcp-status-updated', onTools);
            document.removeEventListener('visibilitychange', onVisible);
        },
    };
}
