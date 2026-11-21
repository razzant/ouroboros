/** Settings: the shell (docs/DESIGN.md "Settings"). The home (settings_home.js) holds Ouroboros's card, search and six
 * groups of rows; a row opens a page, which its module draws into the body the shell gives it (settings_pages.js), or
 * leaves Settings by its door. Every row saves itself (settings_controls.js): nothing here keeps a draft or a Save.
 *
 * What Settings shows is `state.settingsPath` ('' is the home), in memory only: no URL is written and a reload starts
 * at the home (in two panes beside the first-level page last opened on this device, kept in localStorage). An address no page registers (a skill written for the tabs Settings had) never shows an empty page:
 * the home opens with the address's last part searched, its hyphens and underscores read as spaces (`local-model`
 * searches "local model"), and search says where the thing lives now. Pages open in levels. One opened from the
 * home, search or outside keeps its list page under a child address; one opened from a page lies over that page. The
 * shell names the top page layer's Back: the parent level, Settings, or a jump's origin.
 *
 * The page box's width decides the layout (settings.css `@container sectionplace`). Narrow: one level alone, each one
 * `page: true` layer of the shell's stack (layers.js), so Escape, the browser's or Android's Back and Telegram's Back
 * close exactly one level, as Back does; a field that holds focus is left first, so its row commits. Wide: the home
 * beside the level; every level below the first is still a layer. A level under another keeps its place, the home
 * its scroll.
 *
 * `ouro:settings-page-shown` (`detail.path`) says which page is visible: when one opens or is returned to, when
 * Settings is shown, after a reconnect and when the window is seen again. A page reads and repaints on it for its own
 * path; the row helper never rewrites a field that holds focus or typed text. */

import { currentLayout, sectionHoldsTwoPanes } from './layout.js';
import { renderPageHeader } from './page_header.js';
import { revealPastFade, bindScrollFade } from './scroll_fade.js';
import { readSettings } from './settings_controls.js';
import { isDoorId, mountSettingsHome, settingsTitle } from './settings_home.js';
import { retireMcpStatusRead } from './mcp_settings.js';
import { extensionsPageOf, settingsPageFor } from './settings_pages.js';
import { createSectionLevels, isVisible as visible } from './section_levels.js';
import { HOME_ROWS } from './settings_rows.js';
import { resetSecretReveals } from './settings_secrets.js';
// The page modules: each registers its family when it is imported (settings_pages.js).
import './settings_ui.js';
import './settings_accounts.js';
import './subagents_settings.js';
import './model_roles.js';
import './settings_local_model.js';

// The first-level page last opened on this device (docs/DESIGN.md "Settings": a section lands on its first page the
// first time, later on the last one opened on the device). A stored page no longer registered falls back to Models.
const LAST_PAGE_KEY = 'ouroboros.settings.last-page';
function readLastFirst() {
    try { return String(globalThis.localStorage?.getItem(LAST_PAGE_KEY) || ''); } catch { return ''; }
}

const quoted = (value) => String(value).replace(/["\\]/g, '\\$&');

// Before Extensions, the global tool timeout lived beside the MCP timeout on Tools.
const settingsTarget = (path, row) => path === 'tools' && row === 'OUROBOROS_TOOL_TIMEOUT_SEC'
    ? 'computer/advanced' : String(path || '');

export function initSettings(ctx = {}) {
    const { state, ws, layers = null, setBeforePageLeave, showPage, openExtensions, openDashboardTab, stateSnapshots } = ctx;
    state.settingsPath = '';
    const page = document.createElement('div');
    page.id = 'page-settings';
    page.className = 'page app-page-glass section-place';
    // The header over the columns (docs/DESIGN.md "Page header"): no `Settings` word in the band (the sidebar's row
    // names the place). Over the list (and over the home alone) stands its search field; over a page, its name in the
    // header's one title style, after the one Back where Back leads to a level not on screen (settings.css lays the
    // lead on the shell's own columns). A phone's home has one band, `☰ Settings`; its search starts the list.
    const leadHtml = `<div class="settings-lead section-lead" data-settings-lead>
            <span class="settings-lead-col settings-lead-rail section-lead-col section-lead-rail" data-settings-search-slot></span>
            <span class="settings-lead-col settings-lead-page section-lead-col section-lead-page"><span class="app-page-return" data-page-return></span><h2 class="app-page-lead-title" data-settings-lead-name></h2></span>
        </div>`;
    page.dataset.sectionView = 'home';
    page.innerHTML = `${renderPageHeader({ page: 'settings', leadHtml })}
        <div class="settings-shell section-shell" data-settings-view="home" data-section-view="home">
            <div class="settings-home section-rail scroll-fade-y" data-settings-home></div>
            <div class="settings-levels section-pane"></div>
        </div>`;
    document.getElementById('content').appendChild(page);
    const shell = page.querySelector('.settings-shell');
    const header = page.querySelector('.app-page-header');
    const homeEl = page.querySelector('[data-settings-home]');
    const levelsEl = page.querySelector('.settings-levels');
    const leadName = page.querySelector('[data-settings-lead-name]');
    const places = new WeakMap();  // a column → where it was scrolled while visible
    let opening = false;
    let landedBy = null;  // the return of the jump landing now: the levels it shows stay under it
    let besideBefore = false;
    const shown = () => state.activePage === 'settings';
    const beside = () => levels.length > 0 && visible(homeEl);
    const pageCtx = { openSettings: (path, options) => go(path, options), openDashboardTab, showPage, layers, ws, state, readSettings,
        onScreen: (path) => sectionLevels.onScreen(path) };
    const sectionLevels = createSectionLevels({ place: 'settings', pane: levelsEl, layers, shown, ctx: pageCtx,
        title: settingsTitle, onTitled: () => paintLead() });
    const { levels, top, arrive, revealRow } = sectionLevels;
    // Landing (owner 13.11, "1. C"): from 40rem the list and an open page always stand together, never an empty pane;
    // a second press of the chosen row keeps its page when that page is on top, and returns to it from a deeper level
    // (the chosen row is the open page's own: a door from one page of the list to another chooses the other).
    let ownChosen = false;   // the owner pressed the list's Ouroboros row: his page stays until another row is pressed
    let lastFirst = readLastFirst();  // the first-level page last opened on this device
    const pressed = (path, options = {}) => {
        ownChosen = !path;
        const at = path && !options.row && beside() ? levels.findIndex((level) => level.path === path) : -1;
        if (at >= 0) {
            if (levels.length > at + 1) closeLevels(at + 1);
            return true;
        }
        return go(path, { ...options, fresh: true });
    };
    const home = mountSettingsHome(homeEl, { open: pressed,
        ctx: { ...pageCtx, stateSnapshots } });
    // The same search field, with its listeners and query, follows the list between its home and header.
    const searchField = homeEl.querySelector('.settings-search');
    const searchSlot = page.querySelector('[data-settings-search-slot]');
    // From a 40rem box Settings always shows two panes; when the list's Ouroboros row is chosen its pane holds
    // Ouroboros's card. Below 40rem the list is one column.
    const card = homeEl.querySelector('[data-settings-card]');
    const cardHome = card?.parentElement || null;
    const overviewRow = homeEl.querySelector('[data-settings-overview]');
    // The overview pane is drawn the first time two panes show it.
    let overviewEl = null;
    let disposeOverviewFade = () => {};
    function overviewPane() {
        if (overviewEl) return overviewEl;
        overviewEl = document.createElement('section');
        overviewEl.className = 'settings-page scroll-fade-y';
        overviewEl.dataset.settingsPage = '';
        overviewEl.dataset.settingsOverview = '';
        overviewEl.setAttribute('aria-label', 'Ouroboros');
        overviewEl.hidden = true;
        overviewEl.innerHTML = '<div class="settings-column"><div class="settings-page-body"></div></div>';
        levelsEl.append(overviewEl);
        disposeOverviewFade = bindScrollFade(overviewEl);
        return overviewEl;
    }
    const panes = () => sectionHoldsTwoPanes(page);
    function placeSearch() {
        const inHome = currentLayout() === 'stack';
        page.dataset.searchPlace = inHome ? 'list' : 'band';
        const parent = inHome ? cardHome : searchSlot;
        if (!searchField || !parent) return;
        header.toggleAttribute('data-search-in-home', inHome);
        if (searchField.parentElement === parent && (!inHome || parent.firstChild === searchField)) return;
        const active = searchField.contains(document.activeElement) ? document.activeElement : null;
        if (inHome) parent.insertBefore(searchField, parent.firstChild);
        else parent.append(searchField);
        if (active?.isConnected && document.activeElement !== active) active.focus?.({ preventScroll: true });
    }
    // With two panes and no page open, the page last seen on this device comes back (Models on a first visit, or
    // when that page is no longer there).
    let landing = false;
    const ensurePage = async () => {
        if (!panes() || levels.length || ownChosen || opening || !shown() || landing) return;
        landing = true;
        const saved = lastFirst;
        let target = 'models';
        try {
            const registered = !isDoorId(saved) && !extensionsPageOf(saved) && settingsPageFor(saved);
            if (registered && (!registered.exists || await registered.exists(saved))) target = saved;
        } catch { /* An unread registry cannot establish that the stored page still exists. */ }
        landing = false;
        if (!panes() || levels.length || ownChosen || opening || !shown() || saved !== lastFirst) return;
        // Open a resolved page directly: unknown-path search paints the home and queues ensurePage again.
        // Landing never overwrites the device's choice, including while its registry cannot be read. Settings entered
        // from the sidebar or drawer takes the focus into the page it lands on (Back returns it to the row); a row of
        // its own that holds the focus (one a return gave back) keeps it.
        open(target, { under: 0, remember: false, quiet: page.contains(document.activeElement) });
    };
    let panesBefore = null;
    function placeOverview(level) {
        const on = panes();
        panesBefore = on;
        const showing = on && !level;
        if (on) overviewPane();
        if (card) {
            const body = overviewEl?.querySelector('.settings-page-body');
            if (on && card.parentElement !== body) body.append(card);
            else if (!on && cardHome && card.parentElement !== cardHome) cardHome.prepend(card);
        }
        if (overviewRow) {
            overviewRow.hidden = !on;
            const row = overviewRow.querySelector('[data-settings-overview-row]');
            if (showing) row.setAttribute('aria-current', 'page'); else row.removeAttribute('aria-current');
        }
        if (overviewEl) {
            if (showing && overviewEl.hidden) arrive(overviewEl);
            overviewEl.hidden = !showing;
        }
        return showing;
    }

    // A door uses the shell's ordinary jump: no return callback belongs to a link.
    async function leave(door, { row = '', ...options } = {}) {
        const [kind, name] = door.split(':');
        if (kind === 'skills' || door === 'page:skills') return openExtensions?.(kind === 'skills' ? door.slice(7) : '', { row, ...options });
        if (kind === 'dashboard') return openDashboardTab?.(name, { row, ...options });
        return showPage?.(kind === 'main' ? 'chat' : name, options);
    }

    // `view` is what a door asks its page to show first (Permissions' Reviewers: the Agents list on its reviewers).
    function go(path = '', { row = '', fresh = false, opener = null, view = '' } = {}) {
        const target = settingsTarget(path, row);
        if (isDoorId(target)) return leave(target, { row });
        if (extensionsPageOf(target)) return openExtensions?.(target, { row });
        if (target && !settingsPageFor(target)) {
            ownChosen = true; // an explicit unknown address keeps its search, rather than re-entering landing
            closeLevels(0);
            home.search((target.split('/').filter(Boolean).pop() || '').replace(/[-_]+/g, ' ').trim());
            return true;
        }
        const at = fresh ? -1 : levels.findIndex((level) => level.path === target);
        if (!target) closeLevels(0);
        else if (at >= 0) closeLevels(at + 1, { row });
        else open(target, { row, under: fresh ? 0 : levels.length, opener, view });
        return true;
    }

    // `opener` is the control that opened the level, where focus returns: WebKit does not focus a pressed button.
    // A jump-owned or unsaved level comes back as left; a closed clean page is drawn anew.
    function open(path, { row, under, opener, view = '', remember = true, quiet = false }) {
        const parent = path.split('/')[0];
        if (!under && path.includes('/') && HOME_ROWS.some(row => row.to === parent) && settingsPageFor(parent)) {
            open(parent, { under: 0, opener, remember, quiet });
            under = levels.length;
        }
        if (remember && !under && path !== lastFirst) {
            lastFirst = path;
            try { globalThis.localStorage?.setItem(LAST_PAGE_KEY, path); } catch { /* storage blocked: kept for this load */ }
        }
        const level = sectionLevels.open(path, { row, under, opener, view, jump: landedBy });
        paint({ row, opened: level, quiet });
    }

    function closeLevels(keep, { row = '', quiet = false } = {}) {
        const left = sectionLevels.closeLevels(keep);
        paint({ row, returned: left, quiet });
    }

    // The header's lead follows the open level: its name, and Back only where it leads to a level not on screen (one
    // column: its open page; two panes: a deeper level). The shell paints Back from the layer stack.
    function paintLead() {
        const level = top();
        const overview = !level && Boolean(overviewEl && !overviewEl.hidden);
        page.dataset.sectionView = level || overview ? 'page' : 'home';
        // Ouroboros's own pane is named by its card: the band over it holds no title.
        if (leadName) leadName.textContent = level ? settingsTitle(level.path) : '';
        page.dataset.settingsDepth = String(levels.length);
    }

    function backFrom(level) {
        if (level !== top()) return;
        closeLevels(levels.length - 1);
    }

    function paint({ row = '', opened = null, returned = null, quiet = false } = {}) {
        const level = top();
        state.settingsPath = level?.path || '';
        const overview = placeOverview(level);
        placeSearch();
        shell.dataset.settingsView = shell.dataset.sectionView = level || overview ? 'page' : 'home';
        if (!level) queueMicrotask(ensurePage);
        paintLead();
        sectionLevels.showTop();
        const column = level ? level.el : overview ? overviewEl : homeEl;
        column.scrollTop = places.get(column) || 0;
        if (beside() || overview || !level) homeEl.scrollTop = places.get(homeEl) || 0;
        besideBefore = beside();
        home.mark(levels.map((each) => each.path).reverse());  // the open page's own row, else the nearest under it
        // A chosen row is revealed; a quiet layout change (including a section shown again) keeps the rail's passage.
        if ((opened && beside()) || (overview && !quiet)) revealInRail();
        syncLayers();
        if (!quiet) {
            if (row && level) revealRow(level, row);
            else if (opened) {  // beside the home too: a keyboard reaches the page at once, and Back returns to the row
                opened.el.focus({ preventScroll: true });
            } else if (returned) {
                const to = visible(returned.opener) ? returned.opener
                    : level ? level.el
                        : homeEl.querySelector(`[data-settings-open="${quoted(returned.path.split('/')[0])}"]`);
                to?.focus?.({ preventScroll: true });
            }
            announce();
        }
    }

    function revealInRail() {
        const chosen = homeEl.querySelector('[aria-current="page"]');
        if (!chosen) return;
        revealPastFade(homeEl, chosen);
        places.set(homeEl, homeEl.scrollTop);
    }

    function syncLayers() { sectionLevels.layer(beside(), backFrom); }

    // The home reads its answers when it comes on screen (Settings shown, the window seen again, a reconnect, one column
    // back on the list, a width that brings it beside the page), not at a press beside it; off screen its reads are
    // dropped. A page reads on the signal for its own path.
    let listOn = false;
    function look(again = false) {
        const on = shown() && !document.hidden && visible(homeEl);
        if (on && (again || !listOn)) void home.refresh();
        else if (!on && listOn) home.hide();
        listOn = on;
    }
    function announce(again = false) {
        if (!shown()) return;
        sectionLevels.announce();
        look(again);
    }

    // From outside Settings (app.js `openSettings`): Settings is shown first, and refused when the page left vetoes.
    // A path already open keeps its level and its place (a Dashboard tab's Back returns there); any other starts anew.
    async function openFromOutside(path = '', { row = '', ...options } = {}) {
        const target = settingsTarget(path, row);
        if (isDoorId(target)) return leave(target, { row, ...options });
        if (extensionsPageOf(target)) return openExtensions?.(target, { row, ...options });
        opening = true;
        try {
            return await showPage('settings', { ...options, row, key: `settings:${target}`, enter: () => {
                const jump = layers?.returnPage?.();
                landedBy = jump?.destination === `settings:${target}` ? jump.id : null;
                const at = levels.findIndex((level) => level.path === target);
                if (landedBy) levels.slice(0, at + 1).forEach((level) => { level.jump = landedBy; });
                try {
                    if (!target || at >= 0) closeLevels(target ? at + 1 : 0, { row });
                    else go(target, { row, fresh: true });
                } finally { landedBy = null; }
            } });
        } finally { opening = false; }
    }

    // Settings pressed in the sidebar while it is shown (app.js). Two panes: the list beside a first-level page (or
    // Ouroboros's card) stays as it stands, nothing closes or repaints; a deeper level returns to its section's
    // first-level page. One column: the list.
    function pressedAgain() {
        if (!shown()) return false;
        if (!panes()) { if (levels.length) closeLevels(0); }
        else if (levels.length > 1) closeLevels(1);
        return true;
    }

    const onScroll = (event) => {
        const box = event.target;
        if (box === homeEl || (overviewEl && box === overviewEl) || levels.some((level) => level.el === box)) places.set(box, box.scrollTop);
    };
    const onPageShown = (event) => {
        // Settings opened from outside syncs its layers once its landing is known (openFromOutside's `enter`).
        if (event.detail?.page === 'settings' && (opening || event.detail.reveal)) return;
        syncLayers();
        if (event.detail?.page !== 'settings') look();
        else { announce(); ensurePage(); }
    };
    const onVisible = () => (document.hidden ? look() : announce());
    const offReconnect = ws?.on?.('open', (event) => {
        retireMcpStatusRead(event);
        announce(Boolean(event?.previouslyConnected));
    });
    // A width that puts the home beside the page, or takes it away: the layers follow, the home comes back where it
    // was left, and reads when it comes on screen.
    const resized = () => {
        if (panes() !== panesBefore) paint({ quiet: true });
        else {
            placeSearch();
            if (beside() !== besideBefore) {
                besideBefore = beside();
                if (besideBefore) homeEl.scrollTop = places.get(homeEl) || 0;
                paintLead();
                syncLayers();
            }
        }
        look();
    };
    // A new shell width or root font can move the two panes' 40rem. Repaint at the next frame: moving Search
    // between the header and list changes these observed boxes again, which a resize callback must not do.
    let frame = 0;
    const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
        frame ||= requestAnimationFrame(() => { frame = 0; resized(); });
    }) : null;
    [shell, levelsEl, homeEl.querySelector('[data-settings-list]')].forEach((box) => box && resize?.observe(box));
    const disposeFade = bindScrollFade(homeEl);
    shell.addEventListener('scroll', onScroll, { capture: true, passive: true });
    window.addEventListener('ouro:page-shown', onPageShown);
    document.addEventListener('visibilitychange', onVisible);
    // Leaving Settings puts every revealed secret away again.
    setBeforePageLeave?.(({ from }) => {
        if (from === 'settings') resetSecretReveals(page);
        return true;
    });
    window.addEventListener('pagehide', (event) => {
        resetSecretReveals(page);
        if (event.persisted) return;
        resize?.disconnect();
        if (frame) cancelAnimationFrame(frame);
        disposeFade();
        disposeOverviewFade?.();
        offReconnect?.();
        sectionLevels.destroy();
        window.removeEventListener('ouro:page-shown', onPageShown);
        document.removeEventListener('visibilitychange', onVisible);
        home.dispose();
    });

    // Loopback-only debug hook, like app.js's __ouroWs: Settings has no URL, so browser tests open a path through it.
    window.__ouroSettings = { open: openFromOutside };
    return { page, open: openFromOutside, pressedAgain, stateApplied: (data) => home.stateApplied(data),
        capture(opener) {
            const snapshot = sectionLevels.capture(), own = ownChosen;
            const origin = homeEl.contains(opener) ? [] : levels;
            return { label: origin.length ? settingsTitle(origin.at(-1).path) : 'Settings',
                title: ['Settings', ...origin.map((level) => settingsTitle(level.path))].join(' › '),
                retains: snapshot.retains, release: snapshot.release,
                restore() {
                    snapshot.restore();
                    ownChosen = own;
                    paint({ quiet: true });
                    announce();
                } };
        } };
}
