import { bindTabStrip, placePageBack, renderBackControl, renderPageHeader } from './page_header.js';
import { sectionHoldsTwoPanes } from './layout.js';
import { fluentIcon } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { bindSettingRows, readSettings, settingsRow } from './settings_controls.js';
import { homeAnswers, homeFactsSeen } from './settings_home.js';
import { rowsOf } from './settings_rows.js';
import { paintStatusMark, statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr as escape } from './ui_primitives.js';

// By importance and expected use (docs/DESIGN.md §5): the log first and by default, then what
// runs now, what Ouroboros does on its own (consciousness and its places) and what it costs, then
// updates, the owner-measured Storage and Evolution (off by default).
const DASHBOARD_TABS = [
    { value: 'logs', label: 'Logs' },
    { value: 'activity', label: 'Activity' },
    { value: 'consciousness', label: 'Consciousness' },
    { value: 'costs', label: 'Costs' },
    { value: 'updates', label: 'Updates' },
    { value: 'storage', label: 'Storage' },
    { value: 'evolution', label: 'Evolution' },
];
// Static guard markers: rowHtml emits data-dashboard-tab="logs",
// data-dashboard-tab="activity", data-dashboard-tab="consciousness", data-dashboard-tab="costs",
// data-dashboard-tab="updates", data-dashboard-tab="storage", and
// data-dashboard-tab="evolution" from DASHBOARD_TABS at runtime.
const BACK_LAYER = 'dashboard-back';
const LEVEL_LAYER = 'dashboard-level';
// Each page's glyph (Fluent Filled, as the Settings rows that lead to the same pages wear).
const ROW_GLYPHS = {
    logs: 'document-text',
    activity: 'timeline',
    consciousness: 'brain-circuit',
    costs: 'wallet',
    updates: 'arrow-circle-up',
    storage: 'hard-drive',
    evolution: 'branch-fork',
};
const IN_ICON = fluentIcon('chevron-right', 14, 'regular');

/** One page of the Dashboard as a row of the app's section list (ui.css "The section list"), a tab of its vertical
 *  tablist: its glyph, its name, its answer line (filled where Settings already knows one) and its end. */
function rowHtml({ value, label }, active) {
    return `<button type="button" class="section-row dashboard-tab${active ? ' active' : ''}" role="tab" id="dashboard-tab-${value}"`
        + ` aria-controls="dashboard-panel-${value}" aria-selected="${active}" tabindex="${active ? 0 : -1}" data-dashboard-tab="${value}">`
        + `<span class="section-row-glyph" aria-hidden="true">${fluentIcon(ROW_GLYPHS[value], 18)}</span>`
        + `<span class="section-row-words"><span class="section-row-title">${escape(label)}</span>`
        + `<span class="section-row-answer" data-dashboard-answer></span></span>`
        + `<span class="section-row-end" aria-hidden="true">${statusMarkHtml('dashboard-row-mark')}${IN_ICON}</span></button>`;
}

/**
 * A tab's settings: its rows of settings_rows.js (page `dashboard:<tab>`), each saving itself, drawn once into `host`,
 * which lies outside every region the tab repaints. `cards` are the objects they stand in, `{ title, lead, label, rows,
 * fold, solo }`: `rows` names a card's rows by id (one card without it takes the rest), `fold` names the fold of its
 * rare rows, and a `solo` card is one row drawn as the card itself (its name the title, its meta the lead, its control
 * the body). Without `cards` the rows go straight into `host`, a part of a card the tab already has. The rows are read
 * when the tab is shown, painted after a save on any page and read again after one of their own. `custom(row)` draws a
 * custom control, `openDoor(row)` takes a door's press, `onSaved(rowId)` hears this tab's own saves, and
 * `words(settings)` says what a row adds from the last read: `{ [rowId]: { caption, value } }`, the caption in its meta
 * line and a door's value before its chevron. Answers `{ dispose }`.
 */
export function mountTabSettings(host, tab, { state, cards = null, custom = null, openDoor = null, words = null, onPaint = null, onSaved = null } = {}) {
    const page = `dashboard:${tab}`;
    const ctx = { custom, openDoor };
    const named = new Set((cards || []).flatMap((card) => card.rows || []));
    const draw = (card, inFold) => rowsOf(page, { fold: inFold }).filter((row) => (card.rows ? card.rows.includes(row.id) : !named.has(row.id)))
        .map((row) => settingsRow(row, ctx)).join('');
    const body = (card) => {
        const folded = draw(card, true);
        return `${card.title ? `<h3 class="dashboard-card-title">${escape(card.title)}</h3>` : ''}`
            + `${card.lead ? `<p class="dashboard-settings-lead">${escape(card.lead)}</p>` : ''}${draw(card, false)}`
            + (folded ? `<details class="ui-disclosure dashboard-settings-fold" data-settings-fold><summary>${escape(card.fold)}</summary>${folded}</details>` : '');
    };
    host.innerHTML = cards ? cards.map((card) => `<section class="ui-card dashboard-settings"${card.solo ? ' data-solo' : ''}`
        + ` aria-label="${escape(card.label || card.title)}">${body(card)}</section>`).join('') : body({});
    const rows = bindSettingRows(host, ctx);
    const nodes = Array.from(host.querySelectorAll('[data-setting-row]'));
    const shown = () => !document.hidden && state?.activePage === 'dashboard' && state?.dashboardActiveSubtab === tab;
    let settings = null;
    let reads = 0;
    // A row's own words from the last read: made once, then only their text changes.
    const say = (node, kind, text) => {
        let span = node.querySelector(`[data-row-${kind}]`);
        if (!span && text) {
            span = host.ownerDocument.createElement('span');
            span.className = `settings-row-${kind}`;
            span.setAttribute(`data-row-${kind}`, '');
            const box = node.querySelector(kind === 'caption' ? '.ui-row-meta' : '.ui-row-actions');
            box.insertBefore(span, kind === 'caption' ? null : box.firstChild);  // a caption ends the meta line
        }
        if (span && span.textContent !== (text || '')) span.textContent = text || '';
    };
    const paint = () => {
        rows.paint();
        const said = (settings && words?.(settings)) || {};
        nodes.forEach((node) => {
            say(node, 'caption', said[node.dataset.settingRow]?.caption);
            say(node, 'value', said[node.dataset.settingRow]?.value);
        });
        // A card whose every row is left out (a managed deployment's channel) takes no room.
        (cards ? Array.from(host.children) : [host]).forEach((card) => {
            card.hidden = !Array.from(card.querySelectorAll('[data-setting-row]')).some((node) => !node.hidden);
        });
        onPaint?.(settings);
    };
    const read = async () => {
        const mine = ++reads;
        try { settings = await readSettings(); } catch { /* each row keeps what it showed */ }
        if (mine === reads) paint();
    };
    const onShown = () => (shown() ? void read() : rows.leave());
    const onUpdated = (event) => {
        const { source, page: home, row } = event.detail || {};
        if (source === 'settings-row' && home === page) { onSaved?.(row); paint(); void read(); }
        else if (source === 'settings-row') paint();
        else if (shown()) void read();
    };
    window.addEventListener('ouro:dashboard-subtab-shown', onShown);
    window.addEventListener('ouro:page-shown', onShown);
    window.addEventListener('ouro:settings-updated', onUpdated);
    paint();
    if (shown()) void read();
    return {
        dispose() {
            rows.dispose();
            window.removeEventListener('ouro:dashboard-subtab-shown', onShown);
            window.removeEventListener('ouro:page-shown', onShown);
            window.removeEventListener('ouro:settings-updated', onUpdated);
        },
    };
}

/** Bring a row of the panel into view and its control under the focus: a row behind a fold opens it first. */
export function focusSettingRow(panel, id) {
    const node = Array.from(panel.querySelectorAll('[data-setting-row]')).find((row) => row.dataset.settingRow === id);
    if (!node) return false;
    const fold = node.closest('details');
    if (fold) fold.open = true;
    node.scrollIntoView?.({ block: 'center' });
    const control = node.matches('[data-setting-door]') ? node : node.querySelector('[data-setting-control], [data-setting-value], button');
    control?.focus?.({ preventScroll: true });
    return true;
}

/**
 * The way back to Settings from a place a door of Settings opened (PINS §2): a Dashboard tab, or a page of its own
 * (Circles, Skills: settings.js `leave`). The app's one Back (page_header.js renderBackControl), "Back to Settings",
 * icon only, in the place's header Back slot (page_header.js placePageBack): before its tabs or its location on a desk,
 * at the start of the row under ☰ on a phone, never inside a lead that scrolls. A page with no header keeps it as a
 * first row of `host`. One `page: true` layer (`id`), so Escape and the system's Back return there too. Every way of
 * returning leaves a focused field first, so a typed value is saved as it would be on leaving it; leaving the tab or
 * the page any other way drops it (its owner calls `clear`). Answers `{ show(host, path, before), clear() }`.
 */
export function settingsReturn({ layers = null, openSettings = null, id = BACK_LAYER, place = null } = {}) {
    let control = null;
    let scope = null;
    let unplace = null;
    const clear = () => {
        unplace?.();
        unplace = null;
        control?.remove();
        control = null;
        scope = null;
        layers?.remove(id);
    };
    const go = (path) => {
        const field = document.activeElement;
        if (field && scope?.contains(field)) field.blur?.();
        clear();
        void openSettings?.(path);
    };
    const show = (host, path, before = host.firstChild) => {
        clear();
        const holder = document.createElement('div');
        holder.innerHTML = renderBackControl({ to: 'Settings', attrs: 'data-dashboard-back' });
        const button = holder.querySelector('[data-dashboard-back]') || holder.firstElementChild;
        button.addEventListener('click', () => go(path));
        const page = host.closest?.('.page') || host;
        const header = page.querySelector(':scope > .app-page-header');
        // A place that keeps its own Back slot (the Dashboard's, before the open page's name) places it there.
        const own = place?.(button);
        if (own) {
            unplace = own;
            control = button;
        } else if (header?.querySelector('[data-page-back]')) {
            unplace = placePageBack(header, button);
            control = button;
        } else {
            control = document.createElement('div');
            control.className = 'dashboard-back';
            control.append(button);
            host.insertBefore(control, before);
        }
        scope = host;
        layers?.push({ id, el: host, page: true, modal: false, focus: false, opener: null, onClose: () => go(path) });
    };
    return { show, clear };
}

/**
 * The Dashboard: a place with seven pages, drawn as Settings is (ui.css "The section list"). Its pages are the rows of
 * one list, a vertical tablist whose rows keep the `[data-dashboard-tab]` hooks; the open page is that row's panel.
 * The place's box decides the layout: from 40rem the list stands beside the open page (never an empty page: the one
 * last shown, Logs the first time) and the header's band names that page over it; narrower, the Dashboard opens as
 * its list, and a pressed row opens its page as a level (`☰` gives way to Back, then the page's name, in one row),
 * one `page: true` layer, so Escape, the system's Back and Telegram's Back return to the list. A page opened from a
 * Settings row (`activateTab(name, { back })`) gets "Back to Settings" before its name instead. Each row says its
 * page's current answer where the Settings home already knows it (settings_home.js homeAnswers): no read starts here.
 */
export function initDashboard({ state, layers = null, openSettings = null, stateSnapshots = null }) {
    const active = DASHBOARD_TABS.some((tab) => tab.value === state.dashboardActiveSubtab) ? state.dashboardActiveSubtab : 'logs';
    const page = document.createElement('div');
    page.id = 'page-dashboard';
    page.className = 'page app-page-glass section-place';
    page.dataset.sectionView = 'page';
    // The band: over the list nothing; over the page `(‹) <page>` in the header's one title style.
    const leadHtml = `<div class="section-lead dashboard-lead">
            <span class="section-lead-col section-lead-rail"></span>
            <span class="section-lead-col section-lead-page"><span class="dashboard-lead-back" data-dashboard-back-slot></span>${renderBackControl({ to: 'Dashboard', attrs: 'data-dashboard-list-back' })}<span class="app-page-lead-title" data-dashboard-lead-name></span></span>
        </div>`;
    page.innerHTML = `
        ${renderPageHeader({ page: 'dashboard', leadHtml })}
        <div class="dashboard-shell section-shell" data-section-view="page">
            <nav class="dashboard-rail section-rail scroll-fade-y" aria-label="Dashboard">
                <div class="section-list"><section class="section-group">
                    <div class="ui-card section-rows dashboard-tabs" role="tablist" aria-orientation="vertical" aria-label="Dashboard views">
                        ${DASHBOARD_TABS.map((tab) => rowHtml(tab, tab.value === active)).join('')}
                    </div>
                </section></div>
            </nav>
            <div class="dashboard-panels section-pane">
                ${DASHBOARD_TABS.map((tab) => `<section class="dashboard-panel"
                    data-dashboard-panel="${tab.value}" id="dashboard-panel-${tab.value}"
                    role="tabpanel" aria-labelledby="dashboard-tab-${tab.value}" hidden></section>`).join('')}
            </div>
        </div>
    `;
    document.getElementById('content').appendChild(page);

    const shell = page.querySelector('.dashboard-shell');
    const rail = page.querySelector('.dashboard-rail');
    const list = page.querySelector('.dashboard-tabs');
    const panelsEl = page.querySelector('.dashboard-panels');
    const panels = Array.from(page.querySelectorAll('.dashboard-panel'));
    const rows = Array.from(list.querySelectorAll('[data-dashboard-tab]'));
    const leadName = page.querySelector('[data-dashboard-lead-name]');
    const listBack = page.querySelector('[data-dashboard-list-back]');
    const backSlot = page.querySelector('[data-dashboard-back-slot]');
    const tabStrip = bindTabStrip(list, { dataAttr: 'data-dashboard-tab', onChange: (name) => showTab(name) });
    const disposeFade = bindScrollFade(rail);
    // Two panes from a 40rem box where the engine draws them (ui.css @container sectionplace), as Settings.
    const panes = () => sectionHoldsTwoPanes(page);
    let level = false;   // one column: the open page stands as a level over the list
    let layered = false;

    // The way back to Settings stands in the band's Back slot, before the page's name.
    const back = settingsReturn({ layers, openSettings, place: (button) => {
        backSlot.replaceChildren(button);
        paint();
        return () => { backSlot.replaceChildren(); paint(); };
    } });

    function syncLayer() {
        if (!layers) return;
        const wanted = level && !panes() && state.activePage === 'dashboard' && !backSlot.firstChild;
        if (wanted === layered) return;
        layered = wanted;
        if (wanted) layers.push({ id: LEVEL_LAYER, el: panelsEl, page: true, modal: false, focus: false, opener: null, onClose: () => closeLevel() });
        else layers.remove(LEVEL_LAYER);
    }

    function paint() {
        const view = panes() || level ? 'page' : 'home';
        page.dataset.sectionView = view;
        shell.dataset.sectionView = view;
        const tab = DASHBOARD_TABS.find((each) => each.value === state.dashboardActiveSubtab);
        if (leadName) leadName.textContent = tab?.label || '';
        if (listBack) listBack.hidden = panes() || !level || Boolean(backSlot.firstChild);
        syncLayer();
    }

    // What a row says: the answer the Settings home gives its row for the same page, from the app's last state
    // snapshot and whatever the home has read; a page with nothing known says nothing (its row is one line).
    function paintAnswers() {
        const seen = homeFactsSeen();
        const facts = { ...seen, state: stateSnapshots?.latest?.() || seen.state };
        let answers = {};
        try { answers = homeAnswers(facts); } catch { /* a row keeps what it showed */ }
        rows.forEach((row) => {
            const answer = answers[`dashboard:${row.dataset.dashboardTab}`];
            const words = row.querySelector('[data-dashboard-answer]');
            const text = answer?.text || '';
            if (words.textContent !== text) words.textContent = text;
            paintStatusMark(row.querySelector('.dashboard-row-mark'), answer?.owner ? { shape: 'dot', hue: 'owner' } : {});
        });
    }

    function openLevel() {
        if (panes()) return;
        level = true;
        paint();
        (listBack && !listBack.hidden ? listBack : panelsEl).focus?.({ preventScroll: true });
    }

    function closeLevel() {
        if (!level) return;
        level = false;
        back.clear();
        paint();
        rows.find((row) => row.dataset.dashboardTab === state.dashboardActiveSubtab)?.focus?.({ preventScroll: true });
    }
    listBack?.addEventListener('click', () => closeLevel());

    // A row pressed in one column opens its page as a level (pressed again, the page it already shows).
    list.addEventListener('click', (event) => {
        const row = event.target.closest?.('[data-dashboard-tab]');
        if (row && rows.includes(row) && !panes()) openLevel();
    });

    const onPageShown = (event) => {
        if (event.detail?.page !== 'dashboard') {
            back.clear();
            level = false;
            paint();
            return;
        }
        // The Dashboard comes into view: in one column as its list (a deep link opens its page right after).
        level = false;
        paint();
        paintAnswers();
    };
    const onShownTab = () => paintAnswers();
    window.addEventListener('ouro:page-shown', onPageShown);
    window.addEventListener('ouro:dashboard-subtab-shown', onShownTab);
    window.addEventListener('ouro:settings-page-shown', onShownTab);
    // The answers follow the state the app already reads (every 20 s, faster while a room shows): no read of their own.
    const answerTimer = setInterval(() => { if (state.activePage === 'dashboard' && !document.hidden) paintAnswers(); }, 5000);
    answerTimer?.unref?.();  // a test runner's clock is never kept alive by it
    // The place's box repaints at once. A change of the root font moves the two panes' 40rem while the box keeps its
    // width: the pane gains or loses the list beside it and the list's rows resize, and those repaint at the next frame
    // (the repaint resizes them again, a loop the observer would refuse).
    let frame = 0;
    const resize = typeof ResizeObserver === 'function' ? new ResizeObserver((entries) => {
        if (entries.some((entry) => entry.target === page)) paint();
        else frame ||= requestAnimationFrame(() => { frame = 0; paint(); });
    }) : null;
    [page, panelsEl, list].forEach((box) => resize?.observe(box));

    /** Show a tab's page. `back`, a Settings path ('' is its home), gives the page a Back there before its name;
     *  without it there is none. `row` brings that row into view and focuses its control. `asLevel`: in one column
     *  the page is shown as a level, made visible before its row is revealed (a hidden pane scrolls and focuses
     *  nothing). */
    function showTab(tabName, { row = '', back: path, asLevel = false } = {}) {
        const name = tabName || 'logs';
        if (!tabStrip.select(name)) return false;
        const panel = panels.find((candidate) => candidate.dataset.dashboardPanel === name);
        // One change of the layer stack: a level that Back to Settings replaces at once is never pushed and removed
        // first (Telegram's Back would show, hide and show again).
        const steps = () => {
            back.clear();
            panels.forEach((each) => {
                const on = each.dataset.dashboardPanel === name;
                each.classList.toggle('active', on);
                each.hidden = !on;
            });
            state.dashboardActiveSubtab = name;
            if (asLevel && !panes()) level = true;
            paint();
            window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: name } }));
            if (typeof path === 'string') back.show(panel, path);
        };
        if (layers?.batch) layers.batch(steps); else steps();
        if (row) focusSettingRow(panel, row);
        return true;
    }

    /** From outside the list (a deep link, a Settings row): the page is shown, as a level in one column. */
    function activateTab(tabName, options = {}) {
        showTab(tabName, { ...options, asLevel: true });
    }

    showTab(active);
    paintAnswers();
    return {
        page,
        activateTab,
        destroy() {
            tabStrip.destroy();
            disposeFade();
            back.clear();
            resize?.disconnect();
            if (frame) cancelAnimationFrame(frame);
            clearInterval(answerTimer);
            if (layered) layers?.remove(LEVEL_LAYER);
            window.removeEventListener('ouro:page-shown', onPageShown);
            window.removeEventListener('ouro:dashboard-subtab-shown', onShownTab);
            window.removeEventListener('ouro:settings-page-shown', onShownTab);
        },
    };
}

