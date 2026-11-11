import { bindPageHeader, bindTabStrip, renderPageHeader, renderTabStrip } from './page_header.js';
import { SHELL_ICONS } from './page_icons.js';
import { bindSettingRows, readSettings, settingsRow } from './settings_controls.js';
import { rowsOf } from './settings_rows.js';

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
 * A tab's settings: its rows of settings_rows.js (page `dashboard:<tab>`) drawn once into `host`, which lies outside
 * every region the tab repaints, the rare ones in a fold named `fold`; each row saves itself. The rows are read when
 * the tab is shown, painted again after a save on any page and read again after one of their own. `custom` draws a
 * custom control, `openDoor(row)` takes a door's press, `onPaint(settings)` follows each paint with the last read,
 * `onSaved(rowId)` hears this tab's own saves. Answers `{ dispose }`.
 */
export function mountTabSettings(host, tab, { state, title = '', fold = '', custom = null, openDoor = null, onPaint = null, onSaved = null } = {}) {
    const page = `dashboard:${tab}`;
    const ctx = { custom, openDoor };
    const draw = (inFold) => rowsOf(page, { fold: inFold }).map((row) => settingsRow(row, ctx)).join('');
    const folded = draw(true);
    host.innerHTML = `${title ? `<h3 class="dashboard-card-title">${title}</h3>` : ''}${draw(false)}`
        + (folded ? `<details class="ui-disclosure dashboard-settings-more"><summary>${fold}</summary>${folded}</details>` : '');
    const rows = bindSettingRows(host, ctx);
    const shown = () => !document.hidden && state?.activePage === 'dashboard' && state?.dashboardActiveSubtab === tab;
    let settings = null;
    let reads = 0;
    const paint = () => {
        rows.paint();
        // A card whose every row is left out (a managed deployment's channel) takes no room.
        host.hidden = !Array.from(host.querySelectorAll('[data-setting-row]')).some((node) => !node.hidden);
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
        if (source === 'settings-row' && home === page) { onSaved?.(row); void read(); }
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
 * The way back to Settings from a tab it opened (PINS §2): a first row "‹ Settings" above the tab's scroller and one
 * `page: true` layer, so Escape and the system's Back return there too, on a desk and on a phone alike. Every way of
 * returning blurs a focused field first, so a typed value is saved as it would be on leaving it; leaving the tab or
 * the Dashboard any other way drops it. Answers `{ show(panel, path), clear() }`.
 */
export function settingsReturn({ layers = null, openSettings = null } = {}) {
    let row = null;
    const clear = () => {
        row?.remove();
        row = null;
        layers?.remove(BACK_LAYER);
    };
    const go = (path) => {
        const field = document.activeElement;
        if (field && row?.parentElement?.contains(field)) field.blur?.();
        clear();
        void openSettings(path);
    };
    const show = (panel, path) => {
        clear();
        row = document.createElement('div');
        row.className = 'dashboard-back';
        row.innerHTML = `<button type="button" class="btn btn-ghost btn-sm" data-dashboard-back><span aria-hidden="true">${SHELL_ICONS.back}</span><span>Settings</span></button>`;
        row.querySelector('[data-dashboard-back]').addEventListener('click', () => go(path));
        panel.insertBefore(row, panel.firstChild);
        layers?.push({ id: BACK_LAYER, el: panel, page: true, modal: false, focus: false, opener: null, onClose: () => go(path) });
    };
    return { show, clear };
}

export function initDashboard({ state, layers = null, openSettings = null, showPage = null }) {
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
    // Until Settings opens by path, its page is where Back leads.
    const back = settingsReturn({ layers, openSettings: openSettings || (() => showPage?.('settings')) });
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
    page.activateDashboardTab = activateTab;
    return {
        page,
        activateTab,
        destroy() { tabStrip.destroy(); disposeBar(); back.clear(); window.removeEventListener('ouro:page-shown', onPageShown); },
    };
}
