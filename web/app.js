/** Web UI orchestrator: shared state, navigation, page init, WS startup. */

import { createWS } from './modules/ws.js';
import { apiFetch, fetchJson } from './modules/api_client.js';
import { loadVersion, initMatrixRain } from './modules/utils.js';
import { bindScrollFade } from './modules/scroll_fade.js';
import { initChat, createChatInstance } from './modules/chat.js';
import { createStateSnapshotSequencer } from './modules/chat_activity.js';
import {
    buildProjectActivityIndex,
    reconcileProjectActivityCensus,
    summarizeProjectActivities,
} from './modules/project_activity.js';
import { initFiles } from './modules/files.js';
import { getNotifier } from './modules/notifications.js';
import { showToast } from './modules/toast.js';
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

import { fmt, markBootRead, pendingBootRead, refreshDictionary, setLanguage, storedLanguage, tr } from './modules/i18n.js';
import { initOnboardingOverlay } from './modules/onboarding_overlay.js';
import { installAltMenuSuppression, installDesktopShellLinkInterceptor } from './modules/ui_helpers.js';
import { nameProjectReference, projectReference } from './modules/project_reference.js';

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
const navProjectsActivity = document.getElementById('nav-projects-activity');
const navProjectsList = document.getElementById('nav-projects-list');
const projectInstances = new Map();
const projectPaintRequests = new Map();
// The question reveal the current showing is still making (revealProjectQuestion):
// no read decides during it. A close or another showing retires it.
const projectReveals = new Map();
let knownProjectsJson = '';
let lastProjectRows = [];
let projectActivityRows = new Map();
let projectActivityIndex = buildProjectActivityIndex();
let activitySocketDisconnected = false;
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
        if (state.activePage === 'chat') mainChat?.closeTransient?.();
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
    if (!await showPage('settings')) return false;
    if (settingsControls && typeof settingsControls.activateTab === 'function') {
        settingsControls.activateTab(tabName);
    }
    return true;
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
    data = data && typeof data === 'object' ? data : {};
    // A complete REST body cannot prove absence while the socket is in a
    // disconnect episode. Keep the previous rows and render them unknown until
    // a post-reconnect snapshot is sequenced.
    const activity = reconcileProjectActivityCensus(projectActivityRows, activitySocketDisconnected ? {} : data);
    projectActivityRows = activity.rows;
    projectActivityIndex = buildProjectActivityIndex([...projectActivityRows.values()]);
    renderProjectsNav(data.projects || lastProjectRows,
        data.project_chat_ids ?? (data.projects ? undefined : Array.from(state.projectChatIds)), projectActivityIndex);
    applyTaskBindings(data.task_bindings || {});
    hydrateOpenChatsFromState(data, requestedAt, generation);
}, undefined, markProjectActivityUnknown);

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

// Notifications: ONE notifier and ONE subscription per client. Rooms are
// destroyed as the owner navigates (closing a Project panel destroys its chat
// instance), so the subscription cannot live inside a room — it would go silent
// in exactly the case notifications exist for. app.js owns only the click
// destination, the in-app fallback surface and which chats are the owner's;
// every rule lives in modules/notifications.js.
getNotifier().attach({
    ws,
    ownerVisibleChat: (chatId) => chatId === 1 || state.projectChatIds.has(chatId),
});
getNotifier().configure({
    showToast,
    onActivate: (target) => {
        // A notification leads to its message: no document stays open over it,
        // in Main or in any room, whichever chat it names.
        mainChat?.closeTransient?.();
        for (const inst of projectInstances.values()) inst.closeTransient?.();
        const chatId = Number(target?.chatId);
        const project = Array.isArray(lastProjectRows)
            ? lastProjectRows.find((row) => Number(row?.chat_id) === chatId)
            : null;
        if (project) {
            void openProjectPanel(project, {
                openOnly: true,
                taskId: String(target?.taskId || ''),
                quizId: String(target?.quizId || ''),
            });
            return;
        }
        void showPage('chat');
    },
});

mainChat = initChat(ctx);
initFiles(ctx);

function hydrateOpenChatsFromState(data, snapshotRequestedAt) {
    mainChat?.hydrateStateSnapshot?.(data, snapshotRequestedAt);
    for (const instance of projectInstances.values()) {
        instance.hydrateStateSnapshot?.(data, snapshotRequestedAt);
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
// mid-upload never drops attachments. Every reopen lands at the newest message
// (owner decision 2026-10-05): a recreated instance opens there, a kept one moves there.

// A cancelled paint can no longer read, so its request is retired with it: a
// reopened pending-work survivor must start its own read, never join that one.
function cancelProjectPaint(pid, inst) {
    inst?.cancelHistoryPaint?.();
    projectPaintRequests.delete(pid);
}

function destroyProjectInstance(pid) {
    const inst = projectInstances.get(pid);
    if (!inst) return;
    if (inst.hasPendingWork?.()) {
        // Kept for its staged files; a reader it opened does not stay over the next view.
        inst.closeTransient?.();
        inst.page.hidden = true;
        inst.page.dataset.pendingWork = '1';
        cancelProjectPaint(pid, inst);
        return;
    }
    inst.destroy?.();
    projectInstances.delete(pid);
    projectPaintRequests.delete(pid);
}

let projectNavigationGeneration = 0;
function closeProjectPanel({ sync = true } = {}) {
    projectNavigationGeneration += 1;
    projectReveals.clear();
    const activeId = navState.activeProjectId;
    navState.activeProjectId = null;
    if (activeId) destroyProjectInstance(activeId);
    // Anything left is a hidden pending-work survivor; keep it hidden.
    for (const [pid, inst] of projectInstances) {
        inst.closeTransient?.();
        inst.page.hidden = true;
        cancelProjectPaint(pid, inst);
    }
    if (sync) syncNavigationState();
    // Leaving a Project for Main is a return to Main: its newest message (owner
    // decision 2026-10-05). Another page shows Main later through `ouro:page-shown`.
    if (activeId && state.activePage === 'chat') void mainChat?.showLatest?.();
}

async function openProjectPanel(project, { closeDrawer = true, openOnly = false, taskId = '', quizId = '' } = {}) {
    if (!project?.id || String(project.lifecycle || 'active') !== 'active') return;
    const navigation = ++projectNavigationGeneration;
    if (navState.activeProjectId === project.id) {
        if (!openOnly) closeProjectPanel();
        else if (taskId && quizId) await revealProjectQuestion(project, projectInstances.get(project.id), taskId, quizId);
        return;
    }
    projectReveals.clear();
    // perf2 P4.2: signal chat.js that a panel open is in flight so Main's
    // deferred first hydration yields the CPU to this build/paint.
    projectPanelOpeningSince = Date.now();
    try {
        const movedToChat = await showPage('chat', { closeProject: false, closeDrawer: false });
        if (movedToChat === false || navigation !== projectNavigationGeneration) return;
        // The room covers Main: a document Main had open closes rather than sit over it.
        mainChat?.closeTransient?.();
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
                // The panel's Retry reruns the open transaction (fetch, paint, ACK)
                // against the newest known revision, so a recovered read is also seen.
                onHistoryRetry: () => acknowledgeProjectAfterPaint(freshProjectRow(project), null, { forcePaint: true }),
                // Arriving at the newest messages retries a withheld acknowledgement.
                onReadingLatest: () => acknowledgeProjectAfterPaint(freshProjectRow(project)),
            });
            projectInstances.set(project.id, inst);
        }
        // A reopened pending-work survivor is live again.
        delete inst.page.dataset.pendingWork;
        for (const [pid, other] of projectInstances) {
            other.page.hidden = pid !== project.id;
            if (pid !== project.id) {
                other.closeTransient?.();
                cancelProjectPaint(pid, other);
            }
        }
        if (closeDrawer) navState.mobileDrawerOpen = false;
        syncNavigationState();
        // Reopened at the newest message; after the panel is shown, so the column
        // has real geometry. A question opened from Main is revealed below instead.
        if (!(taskId && quizId)) void inst.showLatest?.();
        // ACK only the exact revision whose history was fetched and painted. chat.js
        // owns the paint receipt; an already-painted instance skips the forced
        // refetch — the server clamps the ACK, so no repaint is needed.
        if (taskId && quizId) await revealProjectQuestion(project, inst, taskId, quizId);
        else {
            // Every showing is a navigation: a plain reopen addresses no question
            // and so voids a reveal that a hidden pending-work survivor still awaits.
            void inst.revealQuestion?.(taskId, quizId);
            await acknowledgeProjectAfterPaint(project, inst, { forcePaint: !inst.hasPaintedHistory?.() });
        }
    } finally {
        projectPanelOpeningSince = 0;
    }
}

function freshProjectRow(project) {
    return lastProjectRows.find((row) => row.id === project.id) || project;
}

// A question opened from Main is revealed BEFORE the read decision: landing on it
// is not reading the newer messages below it. The reveal is one transaction: the
// addressed question owns the viewport before either history or detail I/O (its
// chat-owned generation yields to later navigation), its paint supersedes any
// request in flight, and no acknowledgement decides until the reveal has ended,
// when this one does (DESIGN "Project unread dot") — unless a close or a later
// showing retired it meanwhile: that navigation owns the room's read instead.
async function revealProjectQuestion(project, inst, taskId, quizId) {
    const reveal = {};
    let current = false;
    projectReveals.set(project.id, reveal);
    try {
        const revealed = inst?.revealQuestion?.(taskId, quizId);
        await acknowledgeProjectAfterPaint(project, inst, { forcePaint: true, paintOnly: true });
        await revealed;
    } finally {
        current = projectReveals.get(project.id) === reveal;
        if (current) projectReveals.delete(project.id);
    }
    if (current) await acknowledgeProjectAfterPaint(freshProjectRow(project), inst);
}

// A Project can receive a new visible revision while its panel remains open.
// Coalesce polling updates per Project, but never acknowledge a newer revision
// until a history read covering it has been painted while the reader is at the
// newest messages (DESIGN "Project unread dot"); `paintOnly` paints without that decision.
async function acknowledgeProjectAfterPaint(project, instance = null, { forcePaint = false, paintOnly = false } = {}) {
    if (!project?.id || navState.activeProjectId !== project.id) return;
    const inst = instance || projectInstances.get(project.id);
    if (!inst || inst.page.hidden) return;
    const revision = Math.max(0, Number(project.visible_revision) || 0);
    const alreadySeen = Math.max(0, Number(state.projectSeenRevision?.[project.id]) || 0);
    if (!forcePaint && revision <= alreadySeen) return;

    // A question paint never joins an acknowledging request (it would inherit its
    // decision); a request may join a question paint. One that overlaps a question
    // reveal paints but decides nothing.
    const current = projectPaintRequests.get(project.id);
    if (current && current.revision >= revision && (current.paintOnly || !paintOnly)) return current.promise;
    inst.cancelHistoryPaint?.();
    const revealing = projectReveals.has(project.id);
    const promise = (async () => {
        let paint = null;
        // A failed read is drawn by the instance itself (error + Retry in its
        // history control) and resolves unpainted. A rejection here is a defect,
        // not a slow network: it is reported, and still never acknowledged.
        try { paint = await inst.refreshHistory?.({ revision }); } catch (err) {
            console.error('Project history paint failed:', err);
        }
        if (
            !paintOnly && !revealing && !projectReveals.has(project.id) && paint?.read
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
    projectPaintRequests.set(project.id, { revision, promise, paintOnly });
    return promise;
}

document.getElementById('project-panel-close')?.addEventListener('click', () => closeProjectPanel());
projectPanelBackdrop?.addEventListener('click', () => closeProjectPanel());
navProjectsToggle?.addEventListener('click', () => {
    navState.projectsExpanded = !navState.projectsExpanded;
    syncNavigationState();
});

function renderProjectsNav(projects, projectChatIds, activityIndex = projectActivityIndex) {
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
    if (rows.some(p => p._unread)) refreshSharedProjectSeen();
    const json = JSON.stringify(rows.map(p => [
        p.id, p.name, p.chat_id, p.lifecycle, p.visible_revision, p._unread, p.delete_error,
    ]));
    // Activity is a projection over the existing census and must not rebuild
    // the keyed menu: replacing rows here would steal focus from an open menu,
    // rename action or keyboard navigation. Patch marker attributes in place.
    if (json === knownProjectsJson) {
        patchProjectActivityMarkers(activityIndex);
    } else {
        knownProjectsJson = json;
        lastProjectRows = rows;
        paintProjectsNav();
        syncNavigationState();
    }
    // Every snapshot that still shows the open room unread offers the read decision
    // again, so an acknowledgement whose POST failed is retried by the next poll
    // while its reader stays at the newest message; the receipt still decides.
    const active = rows.find((project) => project.id === navState.activeProjectId);
    if (active?._unread && active.lifecycle === 'active') {
        acknowledgeProjectAfterPaint(active);
    }
}

// A transport or state-read failure cannot prove that an observed activity has
// ended. Retain its row for the next complete census, but stop motion and make
// the marker explicitly unknown rather than quietly showing stale liveness.
function markProjectActivityUnknown() {
    projectActivityRows = reconcileProjectActivityCensus(projectActivityRows).rows;
    projectActivityIndex = buildProjectActivityIndex([...projectActivityRows.values()]);
    patchProjectActivityMarkers(projectActivityIndex);
}

function setActivityMarker(marker, summary) {
    if (!marker) return;
    const label = summary.label;
    const active = Boolean(label);
    marker.hidden = !active;
    marker.dataset.state = active ? String(summary.state || 'unknown') : 'idle';
    marker.dataset.motion = summary.motion ? '1' : '0';
    marker.dataset.waiting = summary.waiting ? '1' : '0';
    marker.title = active ? label : '';
    marker.setAttribute('aria-hidden', 'true');
}

function setActivityOwnerName(owner, baseName, summary) {
    if (!owner) return;
    const label = summary.label;
    const name = label ? `${baseName} · ${label}` : baseName;
    owner.setAttribute('aria-label', name);
    owner.title = name;
}

function patchProjectActivityMarkers(activityIndex = projectActivityIndex) {
    setActivityMarker(navProjectsActivity, activityIndex.aggregate);
    const unread = navProjectsCount?.title;
    setActivityOwnerName(navProjectsToggle, unread ? `Projects · ${unread}` : 'Projects', activityIndex.aggregate);
    const buttons = new Map(
        [...(navProjectsList?.querySelectorAll('[data-project-id]') || [])]
            .map((button) => [String(button.dataset.projectId || ''), button]),
    );
    for (const row of lastProjectRows) {
        const button = buttons.get(String(row.id));
        const summary = activityIndex.byProject.get(String(row.id)) || summarizeProjectActivities();
        setActivityMarker(button?.querySelector('.nav-activity-marker'), summary);
        const name = row.name || row.id;
        const baseName = String(row.lifecycle || 'active') === 'deleting'
            ? `${name} — Deleting…${row.delete_error ? ` ${row.delete_error}` : ''}`
            : row._unread ? `${name} · Unread` : name;
        setActivityOwnerName(button, baseName, summary);
    }
}

// ACK exactly the revision painted. The server max-merges and clamps the cursor,
// so stale tabs cannot move it backwards or acknowledge unseen future output; the
// cursors it answers with, never the requested number, become this client's.
async function markProjectViewed(projectId, revision) {
    if (!projectId) return false;
    const seen = Math.max(0, Number(revision) || 0);
    let prefs;
    try {
        prefs = await fetchJson('/api/ui/preferences', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_seen_revision: { [projectId]: seen } }),
        });
    } catch {
        // The room was painted, but the durable monotonic ACK failed. Keep it
        // unread locally so polling or the next open retries the same revision.
        return false;
    }
    mergeProjectSeenRevisions(prefs?.project_seen_revision);
    return true;
}

// Shared read cursors only move forward here, so a late or stale answer can
// clear a dot another client read but never revive one.
function mergeProjectSeenRevisions(cursors) {
    state.projectSeenRevision = state.projectSeenRevision || {};
    let advanced = false;
    for (const [projectId, value] of Object.entries(cursors || {})) {
        const seen = Math.max(0, Number(value) || 0);
        if (seen <= (Number(state.projectSeenRevision[projectId]) || 0)) continue;
        state.projectSeenRevision[projectId] = seen;
        advanced = true;
    }
    if (!advanced || !Array.isArray(lastProjectRows)) return advanced;
    let changed = false;
    for (const row of lastProjectRows) {
        const unread = row._unread && (Number(row.visible_revision) || 0)
            > (Number(state.projectSeenRevision[row.id]) || 0);
        if (row._unread !== unread) { row._unread = unread; changed = true; }
    }
    if (changed) paintProjectsNav();
    return advanced;
}

// Another client may already have read what this one still shows unread. Each
// state snapshot that shows unread re-reads the shared cursors, one read at a time.
let sharedSeenRead = null;
function refreshSharedProjectSeen() {
    sharedSeenRead ||= fetchJson('/api/ui/preferences', { cache: 'no-store' })
        .then((prefs) => mergeProjectSeenRevisions(prefs?.project_seen_revision), () => false)
        .finally(() => { sharedSeenRead = null; });
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
        const activityMarker = document.createElement('span');
        activityMarker.className = 'nav-activity-marker chat-live-typing';
        for (let i = 0; i < 3; i += 1) activityMarker.appendChild(document.createElement('span'));
        btn.appendChild(activityMarker);
        if (project._unread && !deleting) {
            const dot = document.createElement('span');
            dot.className = 'nav-unread-dot';
            dot.title = 'Unread messages';
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
            kebab.title = tr('project.kebab_title', 'Project actions');
            kebab.setAttribute('aria-label', fmt('Actions for {name}', { name: project.name || project.id }));
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
    patchProjectActivityMarkers(projectActivityIndex);
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

// Callers that must observe a read taken AFTER their own change (or after a
// socket open) force one, coalesced behind an in-flight read; the boot prefetch
// and the periodic poll join whatever page-wide read is in flight.
async function refreshProjectsNav(force = true) {
    const request = await stateSnapshots.gate(force);
    if (!request) return;
    try {
        const resp = await apiFetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) {
            stateSnapshots.fail(request);
            return;
        }
        const data = await resp.json();
        stateSnapshots.apply(request, data);
    } catch {
        stateSnapshots.fail(request);
    }
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
    // The reference opens through `ouro:open-project`, whose listener below is
    // open-or-noop and resolves the freshest project row at click time.
    if (!ptr) card.appendChild(ptr = projectReference(project, { layout: 'footer' }));
    card.dataset.projectBound = '1';
    nameProjectReference(ptr, project);
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
    openProjectPanel(resolved, { openOnly: true, taskId: event.detail.task_id || '', quizId: event.detail.quiz_id || '' });
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
        mergeProjectSeenRevisions(prefs?.project_seen_revision);
        setupResizablePanels(prefs || {});
        // Re-evaluate unread now that revision cursors are known.
        if (Array.isArray(lastProjectRows)) { knownProjectsJson = null; renderProjectsNav(lastProjectRows, Array.from(state.projectChatIds || [])); }
    })
    .catch(() => setupResizablePanels({}));

// The interface language is an install-wide setting, not a UI preference: the gateway
// answers the chosen tag and the memory the overlay paints from at /api/ui/i18n. Other
// clients learn about a change from the frames below; the settings save path broadcasts
// nothing, so these are the only cross-client signals.
markBootRead(apiClient.uiI18n()
    .then((i18n) => setLanguage(i18n.language, i18n).then(() => i18n))
    .catch(() => setLanguage(storedLanguage()).then(() => null)));
ws.on('ui_language_changed', () => { refreshDictionary(); });
ws.on('ui_i18n_updated', () => { refreshDictionary(); });

ws.on('open', () => {
    activitySocketDisconnected = false;
    stateSnapshots.fail(stateSnapshots.begin());
    refreshProjectsNav(true); // the in-flight read predates the socket: one coalesced post-open read
    pendingBootRead().then(refreshDictionary); // every open, the first too: a frame sent before this subscription was missed (behind the boot read, whose older answer must not land last)
});
ws.on('close', () => {
    activitySocketDisconnected = true;
    stateSnapshots.fail(stateSnapshots.begin());
});
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
setInterval(() => refreshProjectsNav(false), 20000);
settingsControls = initSettings(ctx);
dashboardControls = initDashboard(ctx);
initLogs({ ...ctx, mount: document.getElementById('dashboard-panel-logs') });
const disposeEvolution = initEvolution({ ...ctx, mount: document.getElementById('dashboard-panel-evolution') });
initUpdates({ ...ctx, mount: document.getElementById('dashboard-panel-updates') });
initActivity({ ...ctx, mount: document.getElementById('dashboard-panel-activity') });
initCosts({ ...ctx, mount: document.getElementById('dashboard-panel-costs') });
initSkills(ctx);
initWidgets(ctx);
initUpdateStatus(ctx);

initOnboardingOverlay();

const disposeMatrixRain = initMatrixRain();
const disposeScrollFades = Array.from(document.querySelectorAll('.scroll-fade-y'), bindScrollFade);
window.addEventListener('pagehide', (event) => {
    if (event.persisted) return;
    disposeEvolution();
    disposeMatrixRain();
    disposeScrollFades.forEach((dispose) => dispose());
});
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
            const overflow = getComputedStyle(el).overflowY;
            if (['auto', 'scroll'].includes(overflow) && el.scrollHeight > el.clientHeight) return el;
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
refreshProjectsNav(false).finally(() => ws.connect());
