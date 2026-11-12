/** Settings: the shell (docs/DESIGN.md "Settings"). The home (settings_home.js) holds Ouroboros's card, search and six
 * groups of rows; a row opens a page, which its module draws into the body the shell gives it (settings_pages.js), or
 * leaves Settings by its door. Every row saves itself (settings_controls.js): nothing here keeps a draft or a Save.
 *
 * What Settings shows is `state.settingsPath` ('' is the home), in memory only: no URL is written and a reload starts
 * at the home. An address no page registers (a skill written for the tabs Settings had) never shows an empty page:
 * the home opens with the address's last part searched, its hyphens and underscores read as spaces (`local-model`
 * searches "local model"), and search says where the thing lives now. Pages open in levels. One opened from the
 * home, a search result or outside Settings is the first level; one opened from a page lies over that page. The
 * first row of every level is Back and names where it leads: the level under it, the home ("Settings"), or the
 * place outside Settings that opened it (`back: { label, go }`).
 *
 * The page box's width decides the layout (settings.css `@container settingspage`). Narrow: one level alone, each one
 * `page: true` layer of the shell's stack (layers.js), so Escape, the browser's or Android's Back and Telegram's Back
 * close exactly one level, as Back does; a field that holds focus is left first, so its row commits. Wide: the home
 * beside the level, and no layer. A level under another keeps its place, the home its scroll.
 *
 * `ouro:settings-page-shown` (`detail.path`) says which page is visible: when one opens or is returned to, when
 * Settings is shown, after a reconnect and when the window is seen again. A page reads and repaints on it for its own
 * path; the row helper never rewrites a field that holds focus or typed text. */

import { settingsReturn } from './dashboard.js';
import { renderPageHeader } from './page_header.js';
import { SHELL_ICONS } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { readSettings } from './settings_controls.js';
import { isDoorId, mountSettingsHome, settingsTitle } from './settings_home.js';
import { settingsPageFor } from './settings_pages.js';
import { resetSecretReveals } from './settings_secrets.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { setResult } from './ui_primitives.js';
// The page modules: each registers its family when it is imported (settings_pages.js).
import './settings_ui.js';
import './settings_accounts.js';
import './subagents_settings.js';
import './mcp_settings.js';
import './model_roles.js';
import './settings_local_model.js';
import './settings_extension_forms.js';

const quoted = (value) => String(value).replace(/["\\]/g, '\\$&');
const visible = (el) => Boolean(el?.isConnected && el.getClientRects?.().length);

export function initSettings(ctx = {}) {
    const { state, ws, layers = null, setBeforePageLeave, showPage, openDashboardTab, stateSnapshots } = ctx;
    state.settingsPath = '';
    const page = document.createElement('div');
    page.id = 'page-settings';
    page.className = 'page app-page-glass';
    page.innerHTML = `${renderPageHeader({ page: 'settings' })}
        <div class="settings-shell" data-settings-view="home">
            <div class="settings-home scroll-fade-y" data-settings-home></div>
            <div class="settings-levels"></div>
        </div>`;
    document.getElementById('content').appendChild(page);
    const shell = page.querySelector('.settings-shell');
    const homeEl = page.querySelector('[data-settings-home]');
    const levelsEl = page.querySelector('.settings-levels');
    const levels = [];
    const aside = new Map();  // path → a level put aside with unsaved words
    const layered = new Set();
    const places = new WeakMap();  // a column → where it was scrolled while visible
    let opening = false;
    let besideBefore = false;
    let serial = 0;
    const shown = () => state.activePage === 'settings';
    const top = () => levels[levels.length - 1] || null;
    const beside = () => levels.length > 0 && visible(homeEl);
    const pageCtx = { openSettings: (path, options) => go(path, options), openDashboardTab, showPage, layers, ws, state, readSettings };
    const home = mountSettingsHome(homeEl, { open: (path, options = {}) => go(path, { ...options, fresh: true }),
        ctx: { ...pageCtx, stateSnapshots } });

    // A door out of Settings. Pressed inside Settings (`back`: the path it was pressed on), it gives where it leads a
    // first-row Back to that path: a Dashboard tab draws it in its panel, a page of its own (Circles, Skills) takes the
    // same row (dashboard.js settingsReturn) under its header and its tabs, above what scrolls: inside the Dashboard's
    // shell where the page has one. Leaving that page any other way drops it (onPageShown). A door asked for from
    // outside Settings (a capsule's `dashboard:costs`) gets no Back: nobody came from Settings. `main:context` shows
    // Main and does nothing else: the Context capsule is the room's.
    const doorBack = settingsReturn({ layers, openSettings: (path) => openFromOutside(path), id: 'settings-door-back' });
    async function leave(door, { row = '', back } = {}) {
        const [kind, name] = door.split(':');
        const inside = typeof back === 'string';
        if (kind === 'dashboard') return openDashboardTab?.(name, inside ? { row, back } : { row });
        const left = await showPage?.(kind === 'main' ? 'chat' : name);
        const landed = left !== false && kind === 'page' && inside ? document.getElementById(`page-${name}`) : null;
        let first = landed && [...landed.children].find((part) => part.matches('.app-page-header'))?.nextElementSibling;
        if (first?.matches('.app-page-tabs')) first = first.nextElementSibling;
        if (first?.matches('.dashboard-shell')) doorBack.show(first, back);
        else if (first) doorBack.show(landed, back, first);
        return left;
    }

    function go(path = '', { row = '', back = null, fresh = false, opener = null } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row, back: state.settingsPath });
        if (target && !settingsPageFor(target)) {
            closeLevels(0);
            home.search((target.split('/').filter(Boolean).pop() || '').replace(/[-_]+/g, ' ').trim());
            return true;
        }
        const at = back || fresh ? -1 : levels.findIndex((level) => level.path === target);
        if (!target) closeLevels(0);
        else if (at >= 0) closeLevels(at + 1, { row });
        else open(target, { row, back, under: back || fresh ? 0 : levels.length, opener });
        return true;
    }

    // `opener` is the control that opened the level, where focus returns: WebKit does not focus a pressed button.
    // A level put aside with unsaved words (`close`) comes back as it was left; any other is drawn anew.
    function open(path, { row, back, under, opener }) {
        while (levels.length > under) close(levels.pop());
        const focused = document.activeElement === document.body ? null : document.activeElement;
        const kept = aside.get(path);
        aside.delete(path);
        const level = Object.assign(kept || { path, id: `settings-level-${++serial}` }, { back, opener: opener || focused });
        if (!kept) {
            level.el = document.createElement('section');
            level.el.className = 'settings-page';
            level.el.dataset.settingsPage = path;
            level.el.setAttribute('aria-labelledby', `${level.id}-title`);
            level.el.innerHTML = `<div class="settings-back-row"><button type="button" class="settings-back ui-hit-area" data-settings-back>
                ${SHELL_ICONS.back}<span data-settings-back-to></span></button></div>
                <div class="settings-column"><h2 class="settings-page-title" id="${level.id}-title">${escapeHtml(settingsTitle(path))}</h2>
                <div class="settings-page-body"></div></div>`;
            level.el.querySelector('[data-settings-back]').addEventListener('click', () => backFrom(level));
        }
        const leadsTo = back?.label || (levels.length ? settingsTitle(top().path) : 'Settings');
        level.el.querySelector('[data-settings-back]').setAttribute('aria-label', `Back to ${leadsTo}`);
        level.el.querySelector('[data-settings-back-to]').textContent = leadsTo;
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
            Promise.resolve(found.mount(body, { path: level.path, row, ctx: pageCtx })).then((handle) => {
                if (level.closed) { handle?.unmount?.(); return; }
                level.handle = handle;
                // A page that learns its name while it draws (a skill's form) is titled again.
                level.el.querySelector('.settings-page-title').textContent = settingsTitle(level.path);
            }, failed);
        } catch (err) { failed(err); }
    }

    // A level closes: the field it holds is left first (its row commits on that change), then its page is let go. A
    // page that says it holds unsaved words (an editor with its own Save) is put aside whole instead: its path opens
    // it again as it was left, while the app is open.
    function close(level) {
        const at = document.activeElement;
        if (at && level.el.contains(at)) at.blur?.();
        level.watch?.disconnect();
        level.el.remove();
        if (layered.delete(level.id)) layers?.remove(level.id);
        if (level.handle?.unsaved?.()) { aside.set(level.path, level); return; }
        level.closed = true;
        try { level.handle?.unmount?.(); } catch (err) { console.error('settings: a page failed to let go', level.path, err); }
    }

    function closeLevels(keep, { row = '', quiet = false } = {}) {
        const left = levels.length > keep ? levels[keep] : null;
        while (levels.length > keep) close(levels.pop());
        paint({ row, returned: left, quiet });
    }

    function backFrom(level) {
        if (level !== top()) return;
        if (levels.length === 1 && level.back) {
            closeLevels(0, { quiet: true });
            level.back.go?.();
            return;
        }
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
        shell.dataset.settingsView = level ? 'page' : 'home';
        for (const each of levels) each.el.hidden = each !== level;
        const column = level ? level.el : homeEl;
        column.scrollTop = places.get(column) || 0;
        if (beside() || !level) homeEl.scrollTop = places.get(homeEl) || 0;
        besideBefore = beside();
        home.mark(levels[0]?.path || '');
        syncLayers();
        if (!quiet) {
            if (row && level) revealRow(level, row);
            else if (opened) {  // beside the home too: a keyboard reaches the page at once, and Back returns to the row
                opened.el.querySelector('[data-settings-back]')?.focus({ preventScroll: true });
            } else if (returned) {
                const to = visible(returned.opener) ? returned.opener
                    : level ? level.el.querySelector('[data-settings-back]')
                        : homeEl.querySelector(`[data-settings-open="${quoted(returned.path.split('/')[0])}"]`);
                to?.focus?.({ preventScroll: true });
            }
            announce();
        }
    }

    // Narrow: every level is a layer, so Escape and the system's Back close the top one. Wide, or Settings put away: none.
    function syncLayers() {
        if (!layers) return;
        const wanted = shown() && !beside() ? levels : [];
        layers.batch(() => {
            for (const id of [...layered]) {
                if (wanted.some((level) => level.id === id)) continue;
                layers.remove(id);
                layered.delete(id);
            }
            for (const level of wanted) {
                layers.push({ id: level.id, el: level.el, page: true, modal: false, focus: false, opener: null,
                    onClose: () => backFrom(level) });
                layered.add(level.id);
            }
        });
    }

    // The home reads its answers again whenever it is on screen; a page reads on the signal for its own path.
    function announce() {
        if (!shown()) return;
        if (visible(homeEl)) void home.refresh();
        window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: state.settingsPath } }));
    }

    // From outside Settings (app.js `openSettings`): Settings is shown first, and refused when the page left vetoes.
    // A path already open keeps its level and its place (a Dashboard tab's Back returns there); any other starts anew.
    async function openFromOutside(path = '', { row = '', back = null } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row });
        opening = true;
        try {
            if (!await showPage('settings')) return false;
        } finally { opening = false; }
        const at = levels.findIndex((level) => level.path === target);
        if (!target || (at >= 0 && !back)) closeLevels(target ? at + 1 : 0, { row });
        else go(target, { row, back, fresh: true });
        return true;
    }

    const onScroll = (event) => {
        const box = event.target;
        if (box === homeEl || levels.some((level) => level.el === box)) places.set(box, box.scrollTop);
    };
    const onPageShown = (event) => {
        doorBack.clear();  // a door's page was left (its own showing came before its row was drawn)
        syncLayers();
        if (event.detail?.page === 'settings' && !opening) announce();
    };
    const onVisible = () => { if (!document.hidden) announce(); };
    const offReconnect = ws?.on?.('open', () => announce());
    // A width that puts the home beside the page, or takes it away: the layers follow, and the home comes back
    // where it was left.
    const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
        if (beside() === besideBefore) return;
        besideBefore = beside();
        if (besideBefore) homeEl.scrollTop = places.get(homeEl) || 0;
        syncLayers();
    }) : null;
    resize?.observe(shell);
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
        disposeFade();
        offReconnect?.();
        doorBack.clear();
        window.removeEventListener('ouro:page-shown', onPageShown);
        document.removeEventListener('visibilitychange', onVisible);
        home.dispose();
    });

    // Loopback-only debug hook, like app.js's __ouroWs: Settings has no URL, so browser tests open a path through it.
    window.__ouroSettings = { open: openFromOutside };
    return { page, open: openFromOutside };
}
