/** Web UI orchestrator: shared state, navigation, page init, WS startup. */

import { createWS } from './modules/ws.js';
import { apiFetch } from './modules/api_client.js';
import { loadVersion, initMatrixRain } from './modules/utils.js';
import { initChat, createChatInstance } from './modules/chat.js';
import { initFiles } from './modules/files.js';

import { initLogs } from './modules/logs.js';
import { initEvolution } from './modules/evolution.js';
import { initSettings } from './modules/settings.js';
import { initCosts } from './modules/costs.js';
import { initSkills } from './modules/skills.js';
import { initWidgets } from './modules/widgets.js';
import { initUpdates } from './modules/updates.js';
import { initActivity } from './modules/activity.js';
import { initUpdateStatus } from './modules/update_status.js';
import { initDashboard } from './modules/dashboard.js';
import { hydrateNavIcons } from './modules/page_icons.js';

import { initOnboardingOverlay } from './modules/onboarding_overlay.js';

const state = {
    messages: [],
    logs: [],
    dashboard: {},
    activeFilters: { tools: true, llm: true, errors: true, tasks: true, system: true, consciousness: true },
    unreadCount: 0,
    activePage: 'chat',
    settingsActiveSubtab: 'providers',
    dashboardActiveSubtab: 'logs',
    beforePageLeave: null,
    // Project-thread isolation SSOT for the live WS fan-out. Initialized to an
    // empty Set (never undefined) so chat.js::isMyThread is deterministic before
    // the first /api/state response; populated by renderProjectsNav.
    projectChatIds: new Set(),
};

// Connect only after modules register listeners.
const ws = createWS();
const beforePageLeaveHandlers = [];
let settingsControls = null;
let dashboardControls = null;
const navState = {
    activeProjectId: null,
    projectsExpanded: true,
    mobileDrawerOpen: false,
};
const primarySidebar = document.getElementById('primary-sidebar');
const navDrawerBackdrop = document.getElementById('nav-drawer-backdrop');
const projectPanelBackdrop = document.getElementById('project-panel-backdrop');
const projectPanel = document.getElementById('project-panel');
const projectPanelBody = document.getElementById('project-panel-body');
const projectPanelTitle = document.getElementById('project-panel-title');
const navProjects = document.getElementById('nav-projects');
const navProjectsToggle = document.getElementById('nav-projects-toggle');
const navProjectsCount = document.getElementById('nav-projects-count');
const navProjectsList = document.getElementById('nav-projects-list');
const projectInstances = new Map();
let knownProjectsJson = '';
let lastProjectRows = [];
let projectPanelHideTimer = null;

function setMobileDrawerOpen(open, { sync = true } = {}) {
    navState.mobileDrawerOpen = Boolean(open);
    if (sync) syncNavigationState();
}

async function showPage(name, options = {}) {
    const pageName = String(name || '').trim();
    if (!pageName) return false;
    const changingPage = state.activePage !== pageName;
    if (changingPage) {
        for (const handler of beforePageLeaveHandlers) {
            const canLeave = await handler({ from: state.activePage, to: pageName });
            if (canLeave === false) return false;
        }
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${pageName}`)?.classList.add('active');
        state.activePage = pageName;
        window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: pageName } }));
        if (pageName === 'chat') {
            state.unreadCount = 0;
            updateUnreadBadge();
        }
    }
    if (options.closeProject !== false) closeProjectPanel({ sync: false });
    if (options.closeDrawer !== false) navState.mobileDrawerOpen = false;
    syncNavigationState();
    return true;
}

async function openSettingsTab(tabName) {
    await showPage('settings');
    if (settingsControls && typeof settingsControls.activateTab === 'function') {
        settingsControls.activateTab(tabName);
    }
}

async function openDashboardTab(tabName) {
    await showPage('dashboard');
    if (dashboardControls && typeof dashboardControls.activateTab === 'function') {
        dashboardControls.activateTab(tabName);
    }
}

function updateUnreadBadge() {
    const btn = document.querySelector('[data-nav-page="chat"]');
    let badge = btn?.querySelector('.unread-badge');
    if (state.unreadCount > 0 && state.activePage !== 'chat') {
        if (!badge) {
            badge = document.createElement('span');
            badge.className = 'unread-badge';
            btn.appendChild(badge);
        }
        badge.textContent = state.unreadCount > 99 ? '99+' : state.unreadCount;
    } else if (badge) {
        badge.remove();
    }
}

function syncNavigationState() {
    const activeProjectId = navState.activeProjectId;
    const drawerOpen = Boolean(navState.mobileDrawerOpen);
    document.body.classList.toggle('nav-drawer-open', drawerOpen);
    primarySidebar?.classList.toggle('open', drawerOpen);
    document.querySelectorAll('[data-mobile-nav-toggle]').forEach((button) => {
        button.setAttribute('aria-expanded', drawerOpen ? 'true' : 'false');
    });
    if (navDrawerBackdrop) navDrawerBackdrop.hidden = !drawerOpen;

    document.querySelectorAll('[data-nav-page]').forEach((button) => {
        const isActive = !activeProjectId && button.dataset.navPage === state.activePage;
        button.classList.toggle('active', isActive);
        if (isActive) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
    navProjectsToggle?.classList.toggle('active', Boolean(activeProjectId));
    navProjectsToggle?.setAttribute('aria-expanded', navState.projectsExpanded ? 'true' : 'false');
    navProjectsList.hidden = !navState.projectsExpanded;
    document.querySelectorAll('[data-project-id]').forEach((button) => {
        const isActive = button.dataset.projectId === activeProjectId;
        button.classList.toggle('active', isActive);
        if (isActive) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
    if (projectPanel) {
        if (projectPanelHideTimer) {
            clearTimeout(projectPanelHideTimer);
            projectPanelHideTimer = null;
        }
        if (activeProjectId) {
            projectPanel.hidden = false;
            if (projectPanelBackdrop) projectPanelBackdrop.hidden = false;
            requestAnimationFrame(() => {
                projectPanel.classList.add('open');
                projectPanelBackdrop?.classList.add('open');
            });
        } else {
            projectPanel.classList.remove('open');
            projectPanelBackdrop?.classList.remove('open');
            projectPanelHideTimer = setTimeout(() => {
                projectPanel.hidden = true;
                if (projectPanelBackdrop) projectPanelBackdrop.hidden = true;
                projectPanelHideTimer = null;
            }, 220);
        }
        document.body.classList.toggle('project-panel-open', Boolean(activeProjectId));
    }
}

document.querySelectorAll('[data-nav-page]').forEach(btn => {
    btn.addEventListener('click', () => {
        showPage(btn.dataset.navPage);
    });
});
document.addEventListener('click', (event) => {
    const toggle = event.target.closest('[data-mobile-nav-toggle]');
    if (!toggle) return;
    setMobileDrawerOpen(!navState.mobileDrawerOpen);
});
navDrawerBackdrop?.addEventListener('click', () => setMobileDrawerOpen(false));
hydrateNavIcons();

const ctx = {
    ws,
    state,
    updateUnreadBadge,
    showPage,
    openSettingsTab,
    openDashboardTab,
    setBeforePageLeave: (handler) => {
        if (typeof handler !== 'function') return () => {};
        beforePageLeaveHandlers.push(handler);
        return () => {
            const idx = beforePageLeaveHandlers.indexOf(handler);
            if (idx >= 0) beforePageLeaveHandlers.splice(idx, 1);
        };
    },
};

initChat(ctx);
initFiles(ctx);

// ---------------------------------------------------------------------------
// Multi-project navigation + right thread panel (v6.32.0). Projects come from
// /api/state; each opens as a chat instance bound to its project chat_id.
// Navigation is one state machine now: page, project, and mobile drawer are
// synchronized together so Utilities and Projects can't remain active at once.
// ---------------------------------------------------------------------------
function closeProjectPanel({ sync = true } = {}) {
    navState.activeProjectId = null;
    for (const inst of projectInstances.values()) inst.page.hidden = true;
    if (sync) syncNavigationState();
}

async function openProjectPanel(project, { closeDrawer = true } = {}) {
    if (!project?.id) return;
    if (navState.activeProjectId === project.id) {
        closeProjectPanel();
        return;
    }
    const movedToChat = await showPage('chat', { closeProject: false, closeDrawer: false });
    if (movedToChat === false) return;
    navState.activeProjectId = project.id;
    markProjectViewed(project.id);
    projectPanelTitle.textContent = project.name || project.id;
    let inst = projectInstances.get(project.id);
    if (!inst) {
        inst = createChatInstance({
            ...ctx,
            chatId: Number(project.chat_id) || 1,
            projectId: project.id,
            idPrefix: `pchat-${project.id}`,
            mountEl: projectPanelBody,
            asPanel: true,
            title: project.name || project.id,
        });
        projectInstances.set(project.id, inst);
    }
    for (const [pid, other] of projectInstances) other.page.hidden = pid !== project.id;
    if (closeDrawer) navState.mobileDrawerOpen = false;
    syncNavigationState();
    // Restore this thread's scroll instead of leaving it at the top (P7). Runs
    // after the panel is shown so the column has real geometry to scroll.
    inst.restoreScrollPosition?.();
}

document.getElementById('project-panel-close')?.addEventListener('click', () => closeProjectPanel());
projectPanelBackdrop?.addEventListener('click', () => closeProjectPanel());
navProjectsToggle?.addEventListener('click', () => {
    navState.projectsExpanded = !navState.projectsExpanded;
    syncNavigationState();
});

function renderProjectsNav(projects, projectChatIds) {
    const all = projects || [];
    // Isolation fan-out SSOT: recognize EVERY registered project chat_id (incl.
    // file-less / no-activity / beyond the sidebar summary cap), matching the
    // backend registered_project_chat_ids, so chat.js::isMyThread never treats a
    // project frame as a main-thread frame on the live WS path. Prefer the
    // COMPLETE /api/state `project_chat_ids` (uncapped); fall back to the (capped)
    // projects array only if that field is absent. Sidebar visibility is a
    // SEPARATE concern (the filtered `rows` below).
    const completeChatIds = Array.isArray(projectChatIds)
        ? projectChatIds
        : all.map(p => Number(p && p.chat_id) || 0);
    state.projectChatIds = new Set(completeChatIds.map(Number).filter(Boolean));
    // Status-free sidebar (v6.33.0: project statuses removed): show projects with
    // thread activity. A project is UNREAD when its last_active_at is newer than the
    // owner's last view (server-stored project_last_viewed). Sort unread first, then
    // by recency. `has_thread_activity` is the only liveness signal now.
    const lastViewed = state.projectLastViewed || {};
    const recency = (p) => String(p.last_active_at || p.updated_at || p.created_at || '');
    const isUnread = (p) => {
        const active = Date.parse(recency(p)) || 0;
        const seen = Date.parse(lastViewed[p.id] || '') || 0;
        return active > 0 && active > seen;
    };
    const rows = all
        .filter(p => p && p.id && p.has_thread_activity !== false)
        .map(p => ({ ...p, _unread: isUnread(p) }))
        .sort((a, b) => {
            if (a._unread !== b._unread) return a._unread ? -1 : 1;  // unread to the top
            return recency(b).localeCompare(recency(a));
        });
    const json = JSON.stringify(rows.map(p => [p.id, p.name, p.chat_id, p._unread]));
    if (json === knownProjectsJson) return;
    knownProjectsJson = json;
    lastProjectRows = rows;
    paintProjectsNav();
    syncNavigationState();
}

// Mark a project as read NOW: clears its unread dot immediately and persists the
// timestamp server-side (so the dot stays cleared across reloads/devices).
function markProjectViewed(projectId) {
    if (!projectId) return;
    const now = new Date().toISOString();
    state.projectLastViewed = state.projectLastViewed || {};
    state.projectLastViewed[projectId] = now;
    if (Array.isArray(lastProjectRows)) {
        let changed = false;
        for (const r of lastProjectRows) if (r.id === projectId && r._unread) { r._unread = false; changed = true; }
        if (changed) paintProjectsNav();
    }
    apiFetch('/api/ui/preferences', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_last_viewed: { [projectId]: now } }),
    }).catch(() => {});
}

// Paint the collapsible, scrollable projects list from the cached rows.
function paintProjectsNav() {
    const rows = lastProjectRows;
    navProjectsList.textContent = '';
    navProjects.hidden = false;
    if (navProjectsCount) navProjectsCount.textContent = rows.length ? String(rows.length) : '';
    for (const project of rows) {
        const btn = document.createElement('button');
        btn.className = 'nav-row nav-project-row';
        btn.dataset.projectId = project.id;
        btn.title = project.name || project.id;
        const label = document.createElement('span');
        label.className = 'nav-row-label';
        label.textContent = project.name || project.id;
        btn.appendChild(label);
        if (project._unread) {
            const dot = document.createElement('span');
            dot.className = 'nav-unread-dot';
            dot.title = 'New activity';
            btn.appendChild(dot);
            btn.classList.add('has-unread');
        }
        if (project.id === navState.activeProjectId) btn.classList.add('active');
        btn.addEventListener('click', () => openProjectPanel(project));
        navProjectsList.appendChild(btn);
    }
}

async function refreshProjectsNav() {
    try {
        const resp = await apiFetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) return;
        const data = await resp.json();
        renderProjectsNav(data.projects || [], data.project_chat_ids);
        applyTaskBindings(data.task_bindings || {});
    } catch {}
}

// A task bound to a project (e.g. a project-chat follow-up) is ALREADY a project
// task. Its main-chat card drops the stray "turn into project" affordance (P2)
// and instead shows a calm pointer that opens the bound project's panel (F4).
// Shared with chat.js's card render via window.__ouroTaskBindings (truthy gate).
function applyTaskBindings(bindings) {
    window.__ouroTaskBindings = bindings || {};
    const entries = window.__ouroTaskBindings;
    const bound = new Set(Object.keys(entries));
    if (!bound.size) return;
    document.querySelectorAll('.chat-live-card[data-task-id]').forEach((card) => {
        const tid = card.dataset.taskId;
        // A converted card (projectCreated) already shows its own project chip.
        if (!bound.has(tid) || card.dataset.projectCreated === '1') return;
        const binding = entries[tid] || {};
        const projectId = String(binding.project_id || '');
        // Always drop the stray convert button (P2).
        card.querySelector('[data-turn-into-project]')?.closest('.chat-live-actions')?.remove();
        // With a known project, render the pointer (F4); legacy chat-id-only
        // bindings (no project_id) keep the P2-only button-removal behaviour.
        if (projectId) renderBoundProjectPointer(card, projectId, Number(binding.chat_id) || 0);
    });
}

// Turn a bound main-chat card into a pointer to its project (F4). Reuses the
// converted-card chip look; idempotent so repeated /api/state polls don't stack.
function renderBoundProjectPointer(card, projectId, chatId = 0) {
    // Prefer the full project row; fall back to the binding's chat_id so the panel
    // still opens for a project beyond the capped sidebar list (codex hardening).
    const project = (Array.isArray(lastProjectRows) && lastProjectRows.find((p) => p.id === projectId))
        || { id: projectId, name: projectId, chat_id: chatId };
    let ptr = card.querySelector('.chat-live-bound-pointer');
    if (!ptr) {
        ptr = document.createElement('button');
        ptr.type = 'button';
        ptr.className = 'chat-live-project-card-btn chat-live-bound-pointer';
        const icon = document.createElement('span');
        icon.className = 'chat-live-project-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '📁';
        const nameEl = document.createElement('span');
        nameEl.className = 'chat-live-project-name';
        const status = document.createElement('span');
        status.className = 'chat-live-project-status';
        status.textContent = 'in project ↗';
        ptr.append(icon, nameEl, status);
        ptr.addEventListener('click', () => openProjectPanel(project));
        card.appendChild(ptr);
    }
    card.dataset.projectBound = '1';
    ptr.querySelector('.chat-live-project-name').textContent = project.name || project.id;
}

window.addEventListener('ouro:project-created', async (event) => {
    const project = event?.detail?.project;
    knownProjectsJson = '';
    await refreshProjectsNav();
    if (project?.id) {
        const resolved = lastProjectRows.find((item) => item.id === project.id) || project;
        openProjectPanel(resolved);
    }
});

// A converted task card's project-identity chip asks to open the project panel.
window.addEventListener('ouro:open-project', (event) => {
    const project = event?.detail?.project;
    if (!project?.id) return;
    const resolved = lastProjectRows.find((item) => item.id === project.id) || project;
    openProjectPanel(resolved);
});

// Resizable side sections: edge drag-handles write --sidebar-width /
// --project-panel-width on :root and persist (debounced) to /api/ui/preferences.
// Disabled under the mobile drawer breakpoint. Width 0 = keep the CSS default.
// CW10 note: the DEVELOPMENT.md "no inline styles in JS" rule targets static styling
// that belongs in a stylesheet — the drag's transient `userSelect:none` was that, and
// is now the `.resizing-panels` class. Setting a custom property (`--sidebar-width`)
// for a DYNAMIC, per-frame drag value is the idiomatic CSS-variable theming API, not a
// static inline style; routing it through a managed <style> rule re-parsed each frame
// would be strictly worse, so CSS-variable mutation is the accepted pattern here.
function setupResizablePanels(prefs) {
    const root = document.documentElement;
    let persistTimer = 0;
    const persist = (patch) => {
        clearTimeout(persistTimer);
        persistTimer = setTimeout(() => {
            apiFetch('/api/ui/preferences', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
            }).catch(() => {});
        }, 400);
    };
    if (Number(prefs?.sidebar_width) > 0) root.style.setProperty('--sidebar-width', `${prefs.sidebar_width}px`);
    if (Number(prefs?.project_panel_width) > 0) root.style.setProperty('--project-panel-width', `${prefs.project_panel_width}px`);
    const isMobile = () => window.matchMedia('(max-width: 980px)').matches;
    const makeHandle = (target, cssVar, dir, prefKey, min, max) => {
        if (!target) return;
        const handle = document.createElement('div');
        handle.className = `resize-handle resize-handle-${dir}`;
        handle.title = 'Drag to resize';
        target.appendChild(handle);
        let startX = 0, startW = 0, dragging = false;
        handle.addEventListener('pointerdown', (e) => {
            if (isMobile()) return;  // mobile uses the drawer, not a resizable column
            dragging = true; startX = e.clientX; startW = target.getBoundingClientRect().width;
            try { handle.setPointerCapture(e.pointerId); } catch {}
            document.body.classList.add('resizing-panels');  // CW10: class, not inline style
            e.preventDefault();
        });
        handle.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const delta = dir === 'right' ? (e.clientX - startX) : (startX - e.clientX);
            const w = Math.max(min, Math.min(max, Math.round(startW + delta)));
            root.style.setProperty(cssVar, `${w}px`);
        });
        const end = (e) => {
            if (!dragging) return;
            dragging = false; document.body.classList.remove('resizing-panels');  // CW10
            try { handle.releasePointerCapture(e.pointerId); } catch {}
            const cur = parseInt(getComputedStyle(root).getPropertyValue(cssVar), 10) || 0;
            persist({ [prefKey]: cur });
        };
        handle.addEventListener('pointerup', end);
        handle.addEventListener('pointercancel', end);
    };
    makeHandle(document.getElementById('primary-sidebar'), '--sidebar-width', 'right', 'sidebar_width', 180, 560);
    makeHandle(document.getElementById('project-panel'), '--project-panel-width', 'left', 'project_panel_width', 320, 1100);
}

apiFetch('/api/ui/preferences', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : null))
    .then((prefs) => {
        state.projectLastViewed = (prefs && prefs.project_last_viewed) || {};
        setupResizablePanels(prefs || {});
        // Re-evaluate unread now that last-viewed is known (sidebar may have painted first).
        if (Array.isArray(lastProjectRows)) { knownProjectsJson = null; renderProjectsNav(lastProjectRows, Array.from(state.projectChatIds || [])); }
    })
    .catch(() => setupResizablePanels({}));

ws.on('open', refreshProjectsNav);
// A backend-created project (e.g. the agent's promote_chat_to_task tool) pushes
// this so the live WS fan-out learns the new project chat_id immediately, instead
// of waiting for the periodic poll and misrouting early frames into the main chat.
// Add the chat_id SYNCHRONOUSLY from the payload so fan-out is correct before the
// async /api/state refresh returns; then refresh the full nav/list.
ws.on('projects_changed', (msg) => {
    const cid = Number(msg && msg.chat_id) || 0;
    if (cid) state.projectChatIds.add(cid);
    refreshProjectsNav();
});
setInterval(refreshProjectsNav, 20000);
settingsControls = initSettings(ctx);
dashboardControls = initDashboard(ctx);
initLogs({ ...ctx, mount: document.getElementById('dashboard-panel-logs') });
initEvolution({ ...ctx, mount: document.getElementById('dashboard-panel-evolution') });
initUpdates({ ...ctx, mount: document.getElementById('dashboard-panel-updates') });
initActivity({ ...ctx, mount: document.getElementById('dashboard-panel-activity') });
initCosts({ ...ctx, mount: document.getElementById('dashboard-panel-costs') });
initSkills(ctx);
initWidgets(ctx);
initUpdateStatus(ctx);

initOnboardingOverlay();

initMatrixRain();
loadVersion();
syncNavigationState();

// Mobile soft-keyboard handling: --vvh + keyboard-open without inline styles.
(function () {
    const vvhStyle = document.createElement('style');
    vvhStyle.id = 'runtime-vvh';
    document.head.appendChild(vvhStyle);

    let wasKeyboardOpen = false;
    let keyboardTouchStartY = 0;
    let frozenBaseline = 0;

    function findScrollableKeyboardNode(target) {
        let el = target;
        while (el && el !== document.body) {
            // Class twins cover secondary chat instances (project panels);
            // the main chat keeps its historic ids.
            if (
                el.id === 'chat-messages'
                || el.id === 'chat-input'
                || el.classList?.contains('chat-messages')
                || el.classList?.contains('chat-input')
                || el.classList?.contains('chat-live-timeline')
            ) return el;
            el = el.parentElement;
        }
        return null;
    }

    function lockTouchStart(e) {
        if (e.touches && e.touches.length) keyboardTouchStartY = e.touches[0].clientY;
    }

    // Stop chat overscroll from moving the document while the keyboard is open.
    function lockBoundaryTouch(e) {
        const touch = e.touches && e.touches.length ? e.touches[0] : null;
        const scrollable = findScrollableKeyboardNode(e.target);
        if (scrollable && touch) {
            const dy = touch.clientY - keyboardTouchStartY;
            const atTop = scrollable.scrollTop <= 0;
            const atBottom = Math.ceil(scrollable.scrollTop + scrollable.clientHeight) >= scrollable.scrollHeight;
            if ((!atTop && dy > 0) || (!atBottom && dy < 0)) return;
        }
        e.preventDefault();
    }

    function captureFrozenBaseline() {
        if (window.innerWidth > 640 || wasKeyboardOpen) return;
        const candidates = [
            document.documentElement.clientHeight,
            window.innerHeight,
            window.screen.availHeight || 0,
            window.screen.height || 0,
        ];
        const best = Math.max(...candidates);
        if (best > frozenBaseline) frozenBaseline = best;
    }

    captureFrozenBaseline();

    const updateVvh = () => {
        const viewport = window.visualViewport;
        const h = viewport ? viewport.height : window.innerHeight;

        if (window.innerWidth <= 640) {
            const safeHeight = Math.max(320, Math.ceil(h || window.innerHeight || 0));
            vvhStyle.textContent = ':root{--vvh:' + safeHeight + 'px}';
            if (!wasKeyboardOpen) captureFrozenBaseline();
            const stableHeight = frozenBaseline || document.documentElement.clientHeight;
            const keyboardVisible = viewport
                ? (stableHeight - h) > Math.max(120, stableHeight * 0.25)
                : false;

            if (keyboardVisible && !wasKeyboardOpen) {
                window.scrollTo(0, 0);
                document.addEventListener('touchstart', lockTouchStart, { passive: true });
                document.addEventListener('touchmove', lockBoundaryTouch, { passive: false });
            }
            if (!keyboardVisible && wasKeyboardOpen) {
                document.removeEventListener('touchstart', lockTouchStart);
                document.removeEventListener('touchmove', lockBoundaryTouch);
            }
            document.documentElement.classList.toggle('keyboard-open', keyboardVisible);
            document.body.classList.toggle('keyboard-open', keyboardVisible);
            wasKeyboardOpen = keyboardVisible;
        } else {
            if (wasKeyboardOpen) {
                document.removeEventListener('touchstart', lockTouchStart);
                document.removeEventListener('touchmove', lockBoundaryTouch);
            }
            document.documentElement.classList.remove('keyboard-open');
            document.body.classList.remove('keyboard-open');
            wasKeyboardOpen = false;
            vvhStyle.textContent = ':root{--vvh:100dvh}';
        }
    };
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', updateVvh);
        window.visualViewport.addEventListener('scroll', updateVvh);
    }
    window.addEventListener('resize', updateVvh);
    window.addEventListener('orientationchange', () => {
        frozenBaseline = 0;
        captureFrozenBaseline();
        updateVvh();
    });
    updateVvh();
}());

// Populate the project-thread isolation set BEFORE opening the socket so the live
// fan-out never misclassifies an early project frame as main-chat traffic during
// startup (chat.js::isMyThread relies on state.projectChatIds). Connect even if
// the prefetch fails, then ws.on('open') keeps it fresh.
refreshProjectsNav().finally(() => ws.connect());
