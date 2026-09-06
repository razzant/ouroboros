/** Web UI orchestrator: shared state, navigation, page init, WS startup. */

import { createWS } from './modules/ws.js';
import { apiFetch, fetchJson } from './modules/api_client.js';
import { loadVersion, initMatrixRain } from './modules/utils.js';
import { initChat, createChatInstance } from './modules/chat.js';
import { createStateSnapshotSequencer } from './modules/chat_activity.js';
import { initFiles } from './modules/files.js';
import { apiClient } from './modules/api_client.js';
import { openNewProjectDialog, openProjectRowMenu } from './modules/project_create.js';

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
import { installAltMenuSuppression, installDesktopShellLinkInterceptor, renderProjectChip } from './modules/ui_helpers.js';

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
// Loopback-only debug hook: the Playwright leak test counts live ws listeners
// through this handle (the module-scoped `ws` is unreachable from page.evaluate).
window.__ouroWs = ws;
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
const projectPaintRequests = new Map();
let knownProjectsJson = '';
let lastProjectRows = [];
let projectPanelHideTimer = null;
let releaseMobileKeyboardForDrawer = () => {};

function setMobileDrawerOpen(open, { sync = true } = {}) {
    const nextOpen = Boolean(open);
    if (nextOpen) releaseMobileKeyboardForDrawer();
    navState.mobileDrawerOpen = nextOpen;
    if (sync) syncNavigationState();
}

// The application's one client-side route: a `#<page>` fragment, honoured once
// on load and validated against the injected sections rather than against a
// duplicated page list — the DOM is the single source of truth, so an unknown
// fragment is ignored instead of painting a blank surface. Deliberately NOT
// written back on navigation: the desktop shell and the Telegram mini app have
// no address bar to read it from.
function pageFromHash() {
    const name = String(window.location.hash || '').replace(/^#/, '').trim();
    return name && document.getElementById(`page-${name}`) ? name : '';
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

// While a Project panel paints, Main defers first hydration (bounded in chat.js).
let projectPanelOpeningSince = 0;
let mainChat;
const stateSnapshots = createStateSnapshotSequencer((data, requestedAt, generation) => {
    renderProjectsNav(data.projects || [], data.project_chat_ids);
    applyTaskBindings(data.task_bindings || {});
    hydrateOpenChatsFromState(data, requestedAt, generation);
});

const ctx = {
    ws,
    state,
    updateUnreadBadge,
    showPage,
    openSettingsTab,
    openDashboardTab,
    isProjectOpening: () => projectPanelOpeningSince > 0,
    stateSnapshots,
    setBeforePageLeave: (handler) => {
        if (typeof handler !== 'function') return () => {};
        beforePageLeaveHandlers.push(handler);
        return () => {
            const idx = beforePageLeaveHandlers.indexOf(handler);
            if (idx >= 0) beforePageLeaveHandlers.splice(idx, 1);
        };
    },
};

mainChat = initChat(ctx);
initFiles(ctx);

function hydrateOpenChatsFromState(data, snapshotRequestedAt, snapshotGeneration) {
    mainChat?.hydrateStateSnapshot?.(data, snapshotRequestedAt, snapshotGeneration);
    for (const instance of projectInstances.values()) {
        instance.hydrateStateSnapshot?.(data, snapshotRequestedAt, snapshotGeneration);
    }
}

// ---------------------------------------------------------------------------
// Multi-project navigation + right thread panel (v6.32.0). Projects come from
// /api/state; each opens as a chat instance bound to its project chat_id.
// Navigation is one state machine now: page, project, and mobile drawer are
// synchronized together so Utilities and Projects can't remain active at once.
// ---------------------------------------------------------------------------
// Single-live-panel policy (P3, owner 7A): at most ONE project chat instance is
// alive; closing or switching destroys the previous one. The exception is an
// instance holding unsendable client state (staged File attachments / an
// in-flight upload): it is hidden and marked instead, so switching to Settings
// mid-upload never drops attachments. Scroll intent survives destruction in a
// small stash keyed by project id and is re-applied after the recreated
// instance's first paint.
const projectScrollStash = new Map();

function destroyProjectInstance(pid) {
    const inst = projectInstances.get(pid);
    if (!inst) return;
    if (inst.hasPendingWork?.()) {
        inst.page.hidden = true;
        inst.page.dataset.pendingWork = '1';
        inst.cancelHistoryPaint?.();
        return;
    }
    const scroll = inst.getScrollState?.();
    if (scroll) projectScrollStash.set(pid, scroll);
    inst.destroy?.();
    projectInstances.delete(pid);
    projectPaintRequests.delete(pid);
}

function closeProjectPanel({ sync = true } = {}) {
    const activeId = navState.activeProjectId;
    navState.activeProjectId = null;
    if (activeId) destroyProjectInstance(activeId);
    // Anything left is a hidden pending-work survivor; keep it hidden.
    for (const inst of projectInstances.values()) {
        inst.page.hidden = true;
        inst.cancelHistoryPaint?.();
    }
    if (sync) syncNavigationState();
}

async function openProjectPanel(project, { closeDrawer = true } = {}) {
    if (!project?.id || String(project.lifecycle || 'active') !== 'active') return;
    if (navState.activeProjectId === project.id) {
        closeProjectPanel();
        return;
    }
    // perf2 P4.2: signal chat.js that a panel open is in flight so Main's
    // deferred first hydration yields the CPU to this build/paint.
    projectPanelOpeningSince = Date.now();
    try {
        const movedToChat = await showPage('chat', { closeProject: false, closeDrawer: false });
        if (movedToChat === false) return;
        navState.activeProjectId = project.id;
        projectPanelTitle.textContent = project.name || project.id;
        // One live panel: every OTHER project instance is destroyed (or hidden and
        // marked when it holds pending work) before the target is created/shown.
        for (const pid of [...projectInstances.keys()]) {
            if (pid !== project.id) destroyProjectInstance(pid);
        }
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
                initialScrollState: projectScrollStash.get(project.id) || null,
            });
            projectScrollStash.delete(project.id);
            projectInstances.set(project.id, inst);
        }
        // A reopened pending-work survivor is live again.
        delete inst.page.dataset.pendingWork;
        for (const [pid, other] of projectInstances) {
            other.page.hidden = pid !== project.id;
            if (pid !== project.id) other.cancelHistoryPaint?.();
        }
        if (closeDrawer) navState.mobileDrawerOpen = false;
        syncNavigationState();
        // Restore this thread's scroll instead of leaving it at the top (P7). Runs
        // after the panel is shown so the column has real geometry to scroll.
        inst.restoreScrollPosition?.();
        // ACK only the exact revision whose history was fetched and painted. chat.js
        // owns the paint receipt; an already-painted instance skips the forced
        // refetch — the server clamps the ACK, so no repaint is needed.
        await acknowledgeProjectAfterPaint(project, inst, { forcePaint: !inst.hasPaintedHistory?.() });
    } finally {
        projectPanelOpeningSince = 0;
    }
}

// A Project can receive a new visible revision while its panel remains open.
// Coalesce polling updates per Project, but never acknowledge a newer revision
// until that exact history snapshot has completed a real browser paint.
async function acknowledgeProjectAfterPaint(project, instance = null, { forcePaint = false } = {}) {
    if (!project?.id || navState.activeProjectId !== project.id) return;
    const inst = instance || projectInstances.get(project.id);
    if (!inst || inst.page.hidden) return;
    const revision = Math.max(0, Number(project.visible_revision) || 0);
    const alreadySeen = Math.max(0, Number(state.projectSeenRevision?.[project.id]) || 0);
    if (!forcePaint && revision <= alreadySeen) return;

    const current = projectPaintRequests.get(project.id);
    if (current && current.revision >= revision) return current.promise;
    inst.cancelHistoryPaint?.();
    const promise = (async () => {
        let paint = null;
        try { paint = await inst.refreshHistory?.({ revision }); } catch {}
        if (
            paint?.painted
            && Number(paint.revision) === revision
            && navState.activeProjectId === project.id
            && !inst.page.hidden
            // A destroyed instance's page reports hidden===false but is detached;
            // a late paint must never ACK a revision nobody saw (GPT#15).
            && inst.page.isConnected
        ) {
            await markProjectViewed(project.id, revision);
        }
    })().finally(() => {
        if (projectPaintRequests.get(project.id)?.promise === promise) {
            projectPaintRequests.delete(project.id);
        }
    });
    projectPaintRequests.set(project.id, { revision, promise });
    return promise;
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
    // Every active Project is visible, including a newly-created empty room. Unread
    // is a monotonic revision comparison, never a timestamp race.
    const seenRevision = state.projectSeenRevision || {};
    const recency = (p) => String(p.last_active_at || p.updated_at || p.created_at || '');
    const isUnread = (p) => Math.max(0, Number(p.visible_revision) || 0)
        > Math.max(0, Number(seenRevision[p.id]) || 0);
    const rows = all
        .filter(p => p && p.id && ['active', 'deleting'].includes(String(p.lifecycle || 'active')))
        .map(p => ({
            ...p,
            _unread: String(p.lifecycle || 'active') === 'active' && isUnread(p),
        }))
        .sort((a, b) => {
            if (a._unread !== b._unread) return a._unread ? -1 : 1;  // unread to the top
            return recency(b).localeCompare(recency(a));
        });
    if (rows.some(p => p.id === navState.activeProjectId && p.lifecycle === 'deleting')) {
        closeProjectPanel();
    }
    const json = JSON.stringify(rows.map(p => [
        p.id, p.name, p.chat_id, p.lifecycle, p.visible_revision, p._unread, p.delete_error,
    ]));
    if (json === knownProjectsJson) return;
    knownProjectsJson = json;
    lastProjectRows = rows;
    paintProjectsNav();
    syncNavigationState();
    const active = rows.find((project) => project.id === navState.activeProjectId);
    if (active?._unread && active.lifecycle === 'active') {
        acknowledgeProjectAfterPaint(active);
    }
}

// ACK exactly the revision painted. The server max-merges and clamps the cursor,
// so stale tabs cannot move it backwards or acknowledge unseen future output.
async function markProjectViewed(projectId, revision) {
    if (!projectId) return false;
    const seen = Math.max(0, Number(revision) || 0);
    try {
        await fetchJson('/api/ui/preferences', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_seen_revision: { [projectId]: seen } }),
        });
    } catch {
        // The room was painted, but the durable monotonic ACK failed. Keep it
        // unread locally so polling or the next open retries the same revision.
        return false;
    }
    state.projectSeenRevision = state.projectSeenRevision || {};
    state.projectSeenRevision[projectId] = Math.max(
        Number(state.projectSeenRevision[projectId]) || 0,
        seen,
    );
    if (Array.isArray(lastProjectRows)) {
        let changed = false;
        for (const row of lastProjectRows) {
            if (row.id !== projectId) continue;
            const unread = (Number(row.visible_revision) || 0) > state.projectSeenRevision[projectId];
            if (row._unread !== unread) { row._unread = unread; changed = true; }
        }
        if (changed) paintProjectsNav();
    }
    return true;
}

// Paint the collapsible, scrollable projects list from the cached rows.
function paintProjectsNav() {
    const rows = lastProjectRows;
    navProjectsList.textContent = '';
    navProjects.hidden = false;
    const unreadCount = rows.filter((project) => project._unread).length;
    if (navProjectsCount) {
        navProjectsCount.textContent = unreadCount ? (unreadCount > 99 ? '99+' : String(unreadCount)) : '';
        navProjectsCount.title = unreadCount ? `${unreadCount} unread project${unreadCount === 1 ? '' : 's'}` : '';
        if (unreadCount) navProjectsCount.setAttribute('aria-label', navProjectsCount.title);
        else navProjectsCount.removeAttribute('aria-label');
    }
    for (const project of rows) {
        const deleting = String(project.lifecycle || 'active') === 'deleting';
        const item = document.createElement('div');
        item.className = `nav-project-item${deleting ? ' is-deleting' : ''}`;
        const btn = document.createElement('button');
        btn.className = 'nav-row nav-project-row';
        btn.type = 'button';
        btn.dataset.projectId = project.id;
        btn.title = deleting
            ? `${project.name || project.id} — Deleting…${project.delete_error ? ` ${project.delete_error}` : ''}`
            : (project.name || project.id);
        btn.disabled = deleting;
        const label = document.createElement('span');
        label.className = 'nav-row-label';
        label.textContent = project.name || project.id;
        btn.appendChild(label);
        if (project._unread && !deleting) {
            const dot = document.createElement('span');
            dot.className = 'nav-unread-dot';
            dot.title = 'New activity';
            btn.appendChild(dot);
            btn.classList.add('has-unread');
        }
        // The action control is a sibling button, never nested interactive UI.
        let trailing;
        if (deleting) {
            trailing = document.createElement('span');
            trailing.className = 'nav-project-deleting-status';
            trailing.textContent = 'Deleting…';
            trailing.title = project.delete_error || 'Cancellation and cleanup are in progress';
        } else {
            const kebab = document.createElement('button');
            kebab.type = 'button';
            kebab.className = 'nav-project-kebab';
            kebab.textContent = '⋯';
            kebab.title = 'Project actions';
            kebab.setAttribute('aria-label', `Actions for ${project.name || project.id}`);
            kebab.addEventListener('click', (event) => {
                event.stopPropagation();
                openProjectRowMenu(project, {
                    apiClient,
                    anchorEl: kebab,
                    onChanged: (change = {}) => {
                        if (change.optimistic && change.projectId) {
                            const row = lastProjectRows.find(p => p.id === change.projectId);
                            if (row) { row.lifecycle = 'deleting'; row._unread = false; }
                            if (navState.activeProjectId === change.projectId) closeProjectPanel();
                            knownProjectsJson = '';
                            paintProjectsNav();
                            return;
                        }
                        knownProjectsJson = '';
                        refreshProjectsNav();
                    },
                });
            });
            trailing = kebab;
        }
        if (project.id === navState.activeProjectId) btn.classList.add('active');
        if (!deleting) btn.addEventListener('click', () => openProjectPanel(project));
        item.append(btn, trailing);
        navProjectsList.appendChild(item);
    }
}

document.getElementById('nav-projects-add')?.addEventListener('click', async (event) => {
    event.stopPropagation();
    const project = await openNewProjectDialog({
        apiClient,
        onCreated: () => { knownProjectsJson = ''; refreshProjectsNav(); },
    });
    if (project?.id) {
        // Fan-out learns the new thread immediately; then open its room.
        if (Number(project.chat_id)) state.projectChatIds.add(Number(project.chat_id));
        await refreshProjectsNav();
        openProjectPanel(project);
    }
});

async function refreshProjectsNav() {
    const request = stateSnapshots.begin();
    try {
        const resp = await apiFetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) return;
        const data = await resp.json();
        stateSnapshots.apply(request, data);
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
    // The pointer is a Main ROOT card affordance: a card inside a project panel
    // is already in that project (its pointer would only close the panel), and a
    // nested subagent card is not the task the binding names.
    document.querySelectorAll('#page-chat .chat-live-card[data-task-id]:not(.subagent)').forEach((card) => {
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
        ptr = renderProjectChip({
            name: project.name || project.id,
            status: 'in project ↗',
            className: 'chat-live-bound-pointer',
            // Open-or-noop: openProjectPanel toggles, and a pointer must never close
            // the panel it points at.
            onClick: () => { if (navState.activeProjectId !== project.id) openProjectPanel(project); },
        });
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
        state.projectSeenRevision = (prefs && prefs.project_seen_revision) || {};
        setupResizablePanels(prefs || {});
        // Re-evaluate unread now that revision cursors are known.
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
const hashPage = pageFromHash();
if (hashPage && hashPage !== state.activePage) showPage(hashPage);

// Mobile soft-keyboard handling: viewport shrink counts only while an editable
// owns focus. Drawer opening clears that state explicitly before navigation is
// rendered, so stale WebView geometry cannot hide an otherwise-open sidebar.
(function () {
    const vvhStyle = document.createElement('style');
    vvhStyle.id = 'runtime-vvh';
    document.head.appendChild(vvhStyle);

    const keyboardEditableSelector = [
        'textarea',
        'select',
        'input:not([type])',
        'input[type="text"]',
        'input[type="search"]',
        'input[type="email"]',
        'input[type="url"]',
        'input[type="tel"]',
        'input[type="password"]',
        'input[type="number"]',
        '[contenteditable]:not([contenteditable="false"])',
    ].join(',');

    let keyboardOpen = false;
    let keyboardTouchStartY = 0;
    let stableViewportHeight = 0;
    let focusBaselineHeight = 0;
    let focusRevision = 0;
    let baselineFrame = 0;

    function keyboardEditable(node) {
        if (!(node instanceof Element)) return null;
        const editable = node.closest(keyboardEditableSelector);
        if (!editable) return null;
        if (editable.matches('input, textarea, select') && (editable.disabled || editable.readOnly)) {
            return null;
        }
        return editable;
    }

    function focusedKeyboardEditable() {
        return keyboardEditable(document.activeElement);
    }

    function viewportHeight() {
        const candidates = [
            window.visualViewport?.height,
            window.innerHeight,
            document.documentElement.clientHeight,
        ];
        return Number(candidates.find((value) => Number.isFinite(value) && value > 0) || 0);
    }

    function findScrollableKeyboardNode(target) {
        let el = target;
        while (el && el !== document.body) {
            if (
                el.id === 'chat-messages'
                || el.id === 'chat-input'
                || el.classList?.contains('chat-messages')
                || el.classList?.contains('chat-input')
                || el.classList?.contains('chat-live-timeline')
                || el.classList?.contains('sidebar-scroll')
            ) return el;
            el = el.parentElement;
        }
        return null;
    }

    function lockTouchStart(e) {
        if (e.touches && e.touches.length) keyboardTouchStartY = e.touches[0].clientY;
    }

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

    function applyKeyboardState(visible) {
        const nextVisible = Boolean(visible);
        if (nextVisible && !keyboardOpen) {
            window.scrollTo(0, 0);
            document.addEventListener('touchstart', lockTouchStart, { passive: true });
            document.addEventListener('touchmove', lockBoundaryTouch, { passive: false });
        } else if (!nextVisible && keyboardOpen) {
            document.removeEventListener('touchstart', lockTouchStart);
            document.removeEventListener('touchmove', lockBoundaryTouch);
        }
        document.documentElement.classList.toggle('keyboard-open', nextVisible);
        document.body.classList.toggle('keyboard-open', nextVisible);
        keyboardOpen = nextVisible;
    }

    function cancelBaselineFrame() {
        if (!baselineFrame) return;
        cancelAnimationFrame(baselineFrame);
        baselineFrame = 0;
    }

    function rememberStableViewport(height) {
        cancelBaselineFrame();
        const revision = focusRevision;
        baselineFrame = requestAnimationFrame(() => {
            baselineFrame = 0;
            if (revision !== focusRevision || focusedKeyboardEditable()) return;
            stableViewportHeight = height;
        });
    }

    const updateVvh = () => {
        const h = viewportHeight();
        if (window.innerWidth <= 640) {
            vvhStyle.textContent = ':root{--vvh:' + Math.max(320, Math.ceil(h)) + 'px}';
            if (!focusedKeyboardEditable()) {
                applyKeyboardState(false);
                rememberStableViewport(h);
                return;
            }
            cancelBaselineFrame();
            if (!focusBaselineHeight) focusBaselineHeight = stableViewportHeight || h;
            const shrink = focusBaselineHeight - h;
            applyKeyboardState(shrink > Math.max(120, focusBaselineHeight * 0.25));
            return;
        }
        cancelBaselineFrame();
        applyKeyboardState(false);
        stableViewportHeight = h;
        focusBaselineHeight = 0;
        vvhStyle.textContent = ':root{--vvh:100dvh}';
    };

    document.addEventListener('focusin', (event) => {
        if (!keyboardEditable(event.target)) return;
        focusRevision += 1;
        cancelBaselineFrame();
        focusBaselineHeight = stableViewportHeight || viewportHeight();
        updateVvh();
    });
    document.addEventListener('focusout', (event) => {
        if (!keyboardEditable(event.target)) return;
        focusRevision += 1;
        const revision = focusRevision;
        cancelBaselineFrame();
        requestAnimationFrame(() => {
            if (revision !== focusRevision || focusedKeyboardEditable()) return;
            applyKeyboardState(false);
            focusBaselineHeight = 0;
        });
    });

    releaseMobileKeyboardForDrawer = () => {
        const editable = focusedKeyboardEditable();
        if (editable && typeof editable.blur === 'function') editable.blur();
        applyKeyboardState(false);
    };

    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', updateVvh);
        window.visualViewport.addEventListener('scroll', updateVvh);
    }
    window.addEventListener('resize', updateVvh);
    stableViewportHeight = viewportHeight();
    updateVvh();
}());

// Windows Alt / Layout Switch Menu-lock suppression (shared installer — the
// onboarding wizard document installs the same guard on its own iframe window).
installAltMenuSuppression();

// Desktop-shell link parity: one delegated interceptor + window.open shim that
// routes target="_blank"/download/data:/blob: intents over the pywebview
// bridge. Installs only when the bridge announces itself; inert in browsers.
installDesktopShellLinkInterceptor();

// Populate the project-thread isolation set BEFORE opening the socket so the live
// fan-out never misclassifies an early project frame as main-chat traffic during
// startup (chat.js::isMyThread relies on state.projectChatIds). Connect even if
// the prefetch fails, then ws.on('open') keeps it fresh.
refreshProjectsNav().finally(() => ws.connect());
