import { bindTabStrip, renderPageHeader, renderTabStrip } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';

// By importance and expected use (docs/DESIGN.md §5): the log first and by default, then what
// runs now and what it costs, then updates and Evolution (off by default).
const DASHBOARD_TABS = [
    { value: 'logs', label: 'Logs' },
    { value: 'activity', label: 'Activity' },
    { value: 'costs', label: 'Costs' },
    { value: 'updates', label: 'Updates' },
    { value: 'evolution', label: 'Evolution' },
];
// Static guard markers: renderTabStrip emits data-dashboard-tab="logs",
// data-dashboard-tab="activity", data-dashboard-tab="costs",
// data-dashboard-tab="updates", and data-dashboard-tab="evolution" from
// DASHBOARD_TABS at runtime.

export function initDashboard({ state }) {
    const page = document.createElement('div');
    page.id = 'page-dashboard';
    page.className = 'page app-page-glass';
    page.innerHTML = `
        ${renderPageHeader({
            title: 'Dashboard',
            icon: PAGE_ICONS.dashboard,
            description: 'Monitor logs, activity, costs, updates, and evolution from one view.',
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
        onChange: activateTab,
    });

    function activateTab(tabName) {
        const name = tabName || 'logs';
        if (!tabStrip.select(name)) return;
        panels.forEach((panel) => {
            const active = panel.dataset.dashboardPanel === name;
            panel.classList.toggle('active', active);
            panel.hidden = !active;
        });
        state.dashboardActiveSubtab = name;
        window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: name } }));
    }

    activateTab(DASHBOARD_TABS.some((tab) => tab.value === state.dashboardActiveSubtab)
        ? state.dashboardActiveSubtab : 'logs');
    page.activateDashboardTab = activateTab;
    return {
        page,
        activateTab,
        destroy: tabStrip.destroy,
    };
}
