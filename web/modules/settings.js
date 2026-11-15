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
import { settingsPageFor } from './settings_pages.js';
import { HOME_ROWS } from './settings_rows.js';
import { resetSecretReveals } from './settings_secrets.js';
import { setResult } from './ui_primitives.js';
// The page modules: each registers its family when it is imported (settings_pages.js).
import './settings_ui.js';
import './settings_accounts.js';
import './subagents_settings.js';
import './mcp_settings.js';
import './model_roles.js';
import './settings_local_model.js';
import './settings_extension_forms.js';

// The first-level page last opened on this device (docs/DESIGN.md "Settings": a section lands on its first page the
// first time, later on the last one opened on the device). A stored page no longer registered falls back to Models.
const LAST_PAGE_KEY = 'ouroboros.settings.last-page';
function readLastFirst() {
    try { return String(globalThis.localStorage?.getItem(LAST_PAGE_KEY) || ''); } catch { return ''; }
}

const quoted = (value) => String(value).replace(/["\\]/g, '\\$&');
const visible = (el) => Boolean(el?.isConnected && el.getClientRects?.().length);

export function initSettings(ctx = {}) {
    const { state, ws, layers = null, setBeforePageLeave, showPage, openDashboardTab, stateSnapshots } = ctx;
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
    const levels = [];
    const aside = new Map();  // path → a level with unsaved words or owned by a jump's return
    const layered = new Set();
    const places = new WeakMap();  // a column → where it was scrolled while visible
    let opening = false;
    let besideBefore = false;
    let serial = 0;
    const shown = () => state.activePage === 'settings';
    const top = () => levels[levels.length - 1] || null;
    const beside = () => levels.length > 0 && visible(homeEl);
    const pageCtx = { openSettings: (path, options) => go(path, options), openDashboardTab, showPage, layers, ws, state, readSettings };
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
            const registered = !isDoorId(saved) && settingsPageFor(saved);
            if (registered && (!registered.exists || await registered.exists(saved))) target = saved;
        } catch { /* An unread registry cannot establish that the stored page still exists. */ }
        landing = false;
        if (!panes() || levels.length || ownChosen || opening || !shown() || saved !== lastFirst) return;
        // Open a resolved page directly: unknown-path search paints the home and queues ensurePage again.
        // Landing never overwrites the device's choice, including while its registry cannot be read.
        open(target, { under: 0, remember: false });
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
        if (kind === 'dashboard') return openDashboardTab?.(name, { row, ...options });
        return showPage?.(kind === 'main' ? 'chat' : name, options);
    }

    // `view` is what a door asks its page to show first (Permissions' Reviewers: the Agents list on its reviewers).
    function go(path = '', { row = '', fresh = false, opener = null, view = '' } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row });
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
    function open(path, { row, under, opener, view = '', remember = true }) {
        const parent = path.split('/')[0];
        if (!under && path.includes('/') && HOME_ROWS.some(row => row.to === parent) && settingsPageFor(parent)) {
            open(parent, { under: 0, opener, remember });
            under = levels.length;
        }
        if (remember && !under && path !== lastFirst) {
            lastFirst = path;
            try { globalThis.localStorage?.setItem(LAST_PAGE_KEY, path); } catch { /* storage blocked: kept for this load */ }
        }
        while (levels.length > under) close(levels.pop());
        const focused = document.activeElement === document.body ? null : document.activeElement;
        const kept = aside.get(path);
        aside.delete(path);
        const level = Object.assign(kept || { path, id: `settings-level-${++serial}`, view }, { opener: opener || focused });
        if (!kept) {
            level.el = document.createElement('section');
            level.el.className = 'settings-page scroll-fade-y';
            level.el.dataset.settingsPage = path;
            level.el.tabIndex = -1;
            level.el.setAttribute('aria-label', settingsTitle(path));
            level.el.innerHTML = '<div class="settings-column"><div class="settings-page-body"></div></div>';
            level.disposeFade = bindScrollFade(level.el);
        }
        // Only the arriving page moves (settings.css): the one that leaves is gone at once.
        arrive(level.el);
        levelsEl.append(level.el);
        levels.push(level);
        if (!kept) mount(level, row);
        paint({ row, opened: level });
    }

    function mount(level, row) {
        const found = settingsPageFor(level.path);  // `go` opens a level only for a path a page registered
        const body = level.el.querySelector('.settings-page-body');
        const failed = (err) => {
            console.error('settings: a page failed to draw', level.path, err);
            const line = document.createElement('p');
            line.className = 'ui-result';
            body.replaceChildren(line);
            setResult(line, { text: `This page could not be shown: ${err?.message || err}`, tone: 'danger' });
        };
        try {
            Promise.resolve(found.mount(body, { path: level.path, row, view: level.view || '', ctx: pageCtx })).then((handle) => {
                if (level.closed) { handle?.unmount?.(); return; }
                level.handle = handle;
                // A page that learns its name while it draws (a skill's form) is named again.
                level.el.setAttribute('aria-label', settingsTitle(level.path));
                if (level === top()) paintLead();
            }, failed);
        } catch (err) { failed(err); }
    }

    const heldByJump = level => layers?.ids().some(id => layers.get(id)?.returnTo?.retains?.(level.el));
    function disposeLevel(level) {
        level.closed = true;
        level.disposeFade?.();
        try { level.handle?.unmount?.(); } catch (err) { console.error('settings: a page failed to let go', level.path, err); }
    }
    function releaseAside() {
        for (const [path, level] of aside) {
            if (level.handle?.unsaved?.() || heldByJump(level)) continue;
            aside.delete(path);
            disposeLevel(level);
        }
    }

    // Closing a clean page releases its bindings. Only unsaved words or an actual jump retain the same nodes.
    function close(level) {
        const at = document.activeElement;
        if (at && level.el.contains(at)) at.blur?.();
        level.watch?.disconnect();
        level.el.remove();
        if (layered.delete(level.id)) layers?.remove(level.id);
        if (level.handle?.unsaved?.() || heldByJump(level)) aside.set(level.path, level);
        else disposeLevel(level);
    }

    function closeLevels(keep, { row = '', quiet = false } = {}) {
        const left = levels.length > keep ? levels[keep] : null;
        while (levels.length > keep) close(levels.pop());
        paint({ row, returned: left, quiet });
    }

    // The arriving page: held at its start state for its first paint, then it comes in (opacity and 8 px from the
    // right on --motion-fade, settings.css; reduced motion keeps the fade alone). Nothing waits on it.
    function arrive(el) {
        el.dataset.settingsArrive = 'wait';
        const run = () => {
            if (el.dataset.settingsArrive !== 'wait') return;
            el.dataset.settingsArrive = 'in';
            el.addEventListener('animationend', () => { delete el.dataset.settingsArrive; }, { once: true });
            setTimeout(() => { delete el.dataset.settingsArrive; }, 600);
        };
        if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => requestAnimationFrame(run));
        else delete el.dataset.settingsArrive;
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

    // The row a page was opened for: its fold opens, it scrolls into view, its control takes focus. A page drawn after
    // a read gets it when the row appears.
    function revealRow(level, id) {
        const find = () => level.el.querySelector(`[data-setting-row="${quoted(id)}"]`);
        const reveal = (node) => {
            for (let fold = node.closest('details'); fold; fold = fold.parentElement?.closest('details')) fold.open = true;
            node.scrollIntoView?.({ block: 'center' });
            const control = node.querySelector('[data-setting-control], [data-setting-value], button, input, select, textarea');
            (control || node).focus?.({ preventScroll: true });
        };
        const node = find();
        if (node || typeof MutationObserver !== 'function') { if (node) reveal(node); return; }
        level.watch = new MutationObserver(() => {
            const hit = find();
            if (!hit) return;
            level.watch.disconnect();
            reveal(hit);
        });
        level.watch.observe(level.el, { childList: true, subtree: true });
    }

    function paint({ row = '', opened = null, returned = null, quiet = false } = {}) {
        const level = top();
        state.settingsPath = level?.path || '';
        const overview = placeOverview(level);
        placeSearch();
        shell.dataset.settingsView = shell.dataset.sectionView = level || overview ? 'page' : 'home';
        if (!level) queueMicrotask(ensurePage);
        paintLead();
        for (const each of levels) each.el.hidden = each !== level;
        const column = level ? level.el : overview ? overviewEl : homeEl;
        column.scrollTop = places.get(column) || 0;
        if (beside() || overview || !level) homeEl.scrollTop = places.get(homeEl) || 0;
        besideBefore = beside();
        home.mark(levels.map((each) => each.path).reverse());  // the open page's own row, else the nearest under it
        // The home narrowed into the rail at once: the pressed row is brought into the rail's view.
        if ((opened && beside()) || overview) revealInRail();
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

    // Every level that hides the one under it is a layer, so Escape and the system's (and Telegram's) Back close the
    // top one: in one column every level; in two panes every level below the first (the first stands beside the list,
    // which is not hidden). Settings put away: none.
    function syncLayers() {
        if (!layers) return;
        const wanted = !shown() ? [] : beside() || layers.returnPage?.() ? levels.slice(1) : levels;
        layers.batch(() => {
            for (const id of [...layered]) {
                if (wanted.some((level) => level.id === id)) continue;
                layers.remove(id);
                layered.delete(id);
            }
            for (const level of wanted) {
                layers.push({ id: level.id, el: level.el, page: true, modal: false, focus: false, opener: null,
                    before: layers.returnPage?.()?.id,
                    back: { label: levels.indexOf(level) > 0 ? settingsTitle(levels[levels.indexOf(level) - 1].path) : 'Settings' },
                    onClose: () => backFrom(level) });
                layered.add(level.id);
            }
        });
    }

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
        window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: state.settingsPath } }));
        look(again);
    }

    // From outside Settings (app.js `openSettings`): Settings is shown first, and refused when the page left vetoes.
    // A path already open keeps its level and its place (a Dashboard tab's Back returns there); any other starts anew.
    async function openFromOutside(path = '', { row = '', ...options } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row, ...options });
        opening = true;
        try {
            return await showPage('settings', { ...options, key: `settings:${target}`, enter: () => {
                const at = levels.findIndex((level) => level.path === target);
                if (!target || at >= 0) closeLevels(target ? at + 1 : 0, { row });
                else go(target, { row, fresh: true });
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
        syncLayers();
        if (event.detail?.page !== 'settings') look();
        else if (!opening) { announce(); ensurePage(); }
    };
    const onVisible = () => (document.hidden ? look() : announce());
    const offReconnect = ws?.on?.('open', (event) => announce(Boolean(event?.previouslyConnected)));  // a reconnect, not the first open
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
        for (const level of [...levels, ...aside.values()]) disposeLevel(level);
        window.removeEventListener('ouro:page-shown', onPageShown);
        document.removeEventListener('visibilitychange', onVisible);
        home.dispose();
    });

    // Loopback-only debug hook, like app.js's __ouroWs: Settings has no URL, so browser tests open a path through it.
    window.__ouroSettings = { open: openFromOutside };
    return { page, open: openFromOutside, pressedAgain,
        capture() {
            const kept = [...levels], own = ownChosen;
            return { title: ['Settings', ...levels.map((level) => settingsTitle(level.path))].join(' › '),
                retains: el => kept.some(level => level.el === el), release: releaseAside,
                restore() {
                    let shared = 0;
                    while (shared < kept.length && levels[shared] === kept[shared]) shared++;
                    closeLevels(shared, { quiet: true });
                    ownChosen = own;
                    kept.forEach((level, under) => { if (under >= shared) open(level.path, { under, remember: false }); });
                    paint({ quiet: true });
                } };
        } };
}
