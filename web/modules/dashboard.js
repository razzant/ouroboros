import { renderPageHeader, renderTabStrip } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';

const DASHBOARD_TABS = [
    { value: 'logs', label: 'Logs' },
    { value: 'evolution', label: 'Evolution' },
    { value: 'costs', label: 'Costs' },
    { value: 'updates', label: 'Updates' },
    { value: 'activity', label: 'Activity' },
];
// Static guard markers: renderTabStrip emits data-dashboard-tab="logs",
// data-dashboard-tab="evolution", data-dashboard-tab="costs",
// data-dashboard-tab="updates", and data-dashboard-tab="activity" from
// DASHBOARD_TABS at runtime.

export function initDashboard({ state }) {
    const page = document.createElement('div');
    page.id = 'page-dashboard';
    page.className = 'page app-page-glass';
    page.innerHTML = `
        ${renderPageHeader({
            title: 'Dashboard',
            icon: PAGE_ICONS.dashboard,
            description: 'Monitor logs, evolution, costs, activity, and update state from one view.',
            tabsHtml: renderTabStrip({
                items: DASHBOARD_TABS,
                active: state.dashboardActiveSubtab || 'logs',
                dataAttr: 'data-dashboard-tab',
                ariaLabel: 'Dashboard views',
                stripClass: 'dashboard-tabs',
                tabClass: 'dashboard-tab',
            }),
        })}
        <div class="dashboard-shell">
            <div class="dashboard-panels">
                <section class="dashboard-panel active" data-dashboard-panel="logs" id="dashboard-panel-logs"></section>
                <section class="dashboard-panel" data-dashboard-panel="evolution" id="dashboard-panel-evolution"></section>
                <section class="dashboard-panel" data-dashboard-panel="costs" id="dashboard-panel-costs"></section>
                <section class="dashboard-panel" data-dashboard-panel="updates" id="dashboard-panel-updates"></section>
                <section class="dashboard-panel" data-dashboard-panel="activity" id="dashboard-panel-activity"></section>
            </div>
        </div>
    `;
    document.getElementById('content').appendChild(page);

    const tabs = Array.from(page.querySelectorAll('.dashboard-tab'));
    const panels = Array.from(page.querySelectorAll('.dashboard-panel'));

    function activateTab(tabName) {
        const name = tabName || 'logs';
        tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.dashboardTab === name));
        panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.dashboardPanel === name));
        state.dashboardActiveSubtab = name;
        window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: name } }));
    }

    tabs.forEach((tab) => {
        tab.addEventListener('click', () => activateTab(tab.dataset.dashboardTab));
    });
    state.dashboardActiveSubtab = state.dashboardActiveSubtab || 'logs';
    page.activateDashboardTab = activateTab;
    return {
        page,
        activateTab,
    };
}
