import { bindPageHeader, bindTabStrip, renderPageHeader, renderTabStrip } from './page_header.js';
import { SHELL_ICONS } from './page_icons.js';
import { bindSettingRows, readSettings, settingsRow } from './settings_controls.js';
import { rowsOf } from './settings_rows.js';
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
// Static guard markers: renderTabStrip emits data-dashboard-tab="logs",
// data-dashboard-tab="activity", data-dashboard-tab="consciousness", data-dashboard-tab="costs",
// data-dashboard-tab="updates", data-dashboard-tab="storage", and
// data-dashboard-tab="evolution" from DASHBOARD_TABS at runtime.
const BACK_LAYER = 'dashboard-back';

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
            box.insertBefore(span, kind === 'caption' ? node.querySelector('[data-setting-result]') : box.firstChild);
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
 * (Circles, Skills: settings.js `leave`). A first row that names Settings, above the place's scroller, drawn as a
 * Settings page draws its own Back (settings.css), and one `page: true` layer (`id`), so Escape and the system's Back
 * return there too. Every way of returning leaves a focused field first, so a typed value is saved as it would be on
 * leaving it; leaving the tab or the page any other way drops the row (its owner calls `clear`). Answers
 * `{ show(host, path, before), clear() }`: the row goes into `host` before `before`, its first child when none is named.
 */
export function settingsReturn({ layers = null, openSettings = null, id = BACK_LAYER } = {}) {
    let row = null;
    const clear = () => {
        row?.remove();
        row = null;
        layers?.remove(id);
    };
    const go = (path) => {
        const field = document.activeElement;
        if (field && row?.parentElement?.contains(field)) field.blur?.();
        clear();
        void openSettings?.(path);
    };
    const show = (host, path, before = host.firstChild) => {
        clear();
        row = document.createElement('div');
        row.className = 'settings-back-row dashboard-back';
        row.innerHTML = `<button type="button" class="settings-back ui-hit-area" data-dashboard-back aria-label="Back to Settings">${SHELL_ICONS.back}<span>Settings</span></button>`;
        row.querySelector('[data-dashboard-back]').addEventListener('click', () => go(path));
        host.insertBefore(row, before);
        layers?.push({ id, el: host, page: true, modal: false, focus: false, opener: null, onClose: () => go(path) });
    };
    return { show, clear };
}

export function initDashboard({ state, layers = null, openSettings = null }) {
    const page = document.createElement('div');
    page.id = 'page-dashboard';
    page.className = 'page app-page-glass';
    page.innerHTML = `
        ${renderPageHeader({
            page: 'dashboard',
            tabsHtml: renderTabStrip({
                items: DASHBOARD_TABS.map((tab) => ({
                    ...tab,
                    tabId: `dashboard-tab-${tab.value}`,
                    panelId: `dashboard-panel-${tab.value}`,
                })),
                active: state.dashboardActiveSubtab || 'logs',
                dataAttr: 'data-dashboard-tab',
                ariaLabel: 'Dashboard views',
                stripClass: 'dashboard-tabs',
                tabClass: 'dashboard-tab',
            }),
        })}
        <div class="dashboard-shell">
            <div class="dashboard-panels">
                ${DASHBOARD_TABS.map((tab) => `<section class="dashboard-panel"
                    data-dashboard-panel="${tab.value}" id="dashboard-panel-${tab.value}"
                    role="tabpanel" aria-labelledby="dashboard-tab-${tab.value}" hidden></section>`).join('')}
            </div>
        </div>
    `;
    document.getElementById('content').appendChild(page);

    const panels = Array.from(page.querySelectorAll('.dashboard-panel'));
    const tabStrip = bindTabStrip(page.querySelector('.dashboard-tabs'), {
        dataAttr: 'data-dashboard-tab',
        onChange: (name) => activateTab(name),
    });
    // On a desk the six tabs are the bar's start (docs/DESIGN.md "Page header").
    const disposeBar = bindPageHeader(page.querySelector('.app-page-header'));
    const back = settingsReturn({ layers, openSettings });
    const onPageShown = (event) => { if (event.detail?.page !== 'dashboard') back.clear(); };
    window.addEventListener('ouro:page-shown', onPageShown);

    /** Show a tab. `back`, a Settings path ('' is its home), gives the tab a first-row Back there; without it there is
     *  none. `row` brings that row into view and focuses its control. */
    function activateTab(tabName, { row = '', back: path } = {}) {
        const name = tabName || 'logs';
        if (!tabStrip.select(name)) return;
        back.clear();
        panels.forEach((panel) => {
            const active = panel.dataset.dashboardPanel === name;
            panel.classList.toggle('active', active);
            panel.hidden = !active;
        });
        state.dashboardActiveSubtab = name;
        window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: name } }));
        const panel = panels.find((candidate) => candidate.dataset.dashboardPanel === name);
        if (typeof path === 'string') back.show(panel, path);
        if (row) focusSettingRow(panel, row);
    }

    activateTab(DASHBOARD_TABS.some((tab) => tab.value === state.dashboardActiveSubtab)
        ? state.dashboardActiveSubtab : 'logs');
    return {
        page,
        activateTab,
        destroy() { tabStrip.destroy(); disposeBar(); back.clear(); window.removeEventListener('ouro:page-shown', onPageShown); },
    };
}
