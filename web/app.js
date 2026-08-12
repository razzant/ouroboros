/** Web UI orchestrator: shared state, navigation, page init, WS startup. */

import { createWS } from './modules/ws.js';
import { apiFetch, fetchJson } from './modules/api_client.js';
import { loadVersion } from './modules/utils.js';
import {
    initChat,
    createChatInstance,
    forgetThreadTranscriptCache,
    headerBudgetPresentation,
} from './modules/chat.js';
import { initFiles } from './modules/files.js';
import { apiClient } from './modules/api_client.js';
import { openNewProjectDialog, openProjectRowMenu } from './modules/project_create.js';
import {
    MAIN_THREAD_ID,
    applyManualOrder,
    attachProjectReorder,
    createThreadStage,
    isMainThreadUnread,
    isThreadUnread,
    normalizeSeenRevision,
    openArchivedThreadsMenu,
    openThreadRowMenu,
    orderProjectRows,
    projectThreadRows,
    readThreadCheckout,
    rememberSeenRevision,
    renderThreadList,
    runThreadAction,
    threadActionItemsHtml,
    threadKey,
    unreadThreadCount,
} from './modules/project_threads.js';

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
import { createStatePoll } from './modules/state_poll.js';
import { showToast } from './modules/toast.js';

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
    // Nested per-thread read cursor {project: {thread: revision}}. Initialized to
    // an empty object (never undefined) so the unread arithmetic is deterministic
    // before /api/ui/preferences resolves; replaced by the normalized response.
    projectSeenRevision: {},
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
    // The open thread, as (project, thread). A project row IS its thread #0, so
    // `activeThreadId === 0` means "the project's own chat is open".
    activeProjectId: null,
    activeThreadId: null,
    projectsExpanded: true,
    mobileDrawerOpen: false,
    // Right panel = ONE slot with mutually exclusive kinds. Since T1 a project
    // thread takes over the CENTRE instead, so the panel serves registered kinds
    // (the task inspector today) only.
    panelKind: null,
};
/** The instance/stash/cursor key of the open thread, or '' when none is open. */
const activeThreadKey = () => (
    navState.activeProjectId === null
        ? ''
        : threadKey(navState.activeProjectId, navState.activeThreadId ?? MAIN_THREAD_ID)
);
// kind -> { mount(opts) => boolean|Promise<boolean>, unmount() }
const rightPanelRegistry = {};
const primarySidebar = document.getElementById('primary-sidebar');
const navDrawerBackdrop = document.getElementById('nav-drawer-backdrop');
const navProjects = document.getElementById('nav-projects');
const navProjectsToggle = document.getElementById('nav-projects-toggle');
const navProjectsCount = document.getElementById('nav-projects-count');
const navProjectsList = document.getElementById('nav-projects-list');
const navBrand = document.getElementById('nav-brand');
const navBrandStatus = document.getElementById('nav-brand-status');
const navBudget = document.getElementById('nav-budget');
const navBudgetAmount = document.getElementById('nav-budget-amount');
const navBudgetBar = document.getElementById('nav-budget-bar');
// Keyed by threadKey(projectId, threadId) — one project can hold many threads,
// and keying any of these by project id alone would make two threads of the same
// project share an instance, a scroll stash and a paint receipt.
const projectInstances = new Map();
const projectPaintRequests = new Map();
let knownProjectsJson = '';
let lastProjectRows = [];
// Owner drag-and-drop order (D3), persisted through /api/ui/preferences.
let projectOrder = [];
let threadOrder = {};
let releaseMobileKeyboardForDrawer = () => {};

function setMobileDrawerOpen(open, { sync = true } = {}) {
    const nextOpen = Boolean(open);
    if (nextOpen) releaseMobileKeyboardForDrawer();
    navState.mobileDrawerOpen = nextOpen;
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
    // Navigating anywhere else tears the open thread down (single live instance)
    // AND vacates the right panel. `openThread` passes closeProject:false because
    // it IS the navigation — and because a thread and the task inspector are no
    // longer mutually exclusive: the thread owns the centre, the inspector owns
    // the right slot, so inspecting the task a thread is discussing keeps both.
    if (options.closeProject !== false) {
        closeProjectPanel({ sync: false });
        closeRightPanel({ sync: false });
    }
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
    const openThreadKey = activeThreadKey();
    // A project ROW is "active" only when the project's OWN thread (#0) is the
    // open one; an open sibling thread highlights its own row instead.
    const activeRowProjectId = navState.activeThreadId === MAIN_THREAD_ID ? activeProjectId : null;
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
    document.querySelectorAll('.nav-project-row[data-project-id]').forEach((button) => {
        const isActive = button.dataset.projectId === activeRowProjectId;
        button.classList.toggle('active', isActive);
        if (isActive) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
    document.querySelectorAll('.nav-thread-row[data-thread-key]').forEach((button) => {
        const isActive = button.dataset.threadKey === openThreadKey;
        button.classList.toggle('active', isActive);
        if (isActive) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
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

// perf2 P4.2: non-zero while openProjectPanel is building/painting a panel —
// Main's chat instance defers its first hydration to it (bounded upper limit
// lives in chat.js), so a fast project open never competes with Main replay.
let projectPanelOpeningSince = 0;

// ---------------------------------------------------------------------------
// Single /api/state poll owner. Before this there were TWO timers —
// app.js polled every 20s for the projects nav and every chat instance polled
// every 3s for the header controls — so an open project panel multiplied the
// request rate. Now ONE app-owned fetch publishes the same snapshot to every
// consumer through `subscribeState`, and ONE self-scheduling timer sets the
// cadence: ~3s while the Chat page is visible (live budget/mode feedback),
// ~20s elsewhere, and paused entirely while the document is hidden.
//
// This consolidates TIMERS ONLY. The refresh ENTRY POINTS are unchanged: the
// startup barrier below still awaits `refreshProjectsNav()` before opening the
// socket, `projects_changed` still adds its chat_id synchronously and then
// refreshes, and create/delete still force an imperative refresh.
// ---------------------------------------------------------------------------
// The subscriber fan-out, the single-flight coalescing and the cadence decision are
// the `state_poll.js` core (node-tested there against fake timers). What stays HERE is
// the impure half: the fetch itself, plus the nav/bindings side effects only this
// module can perform. `activePage` and `hidden` are passed as GETTERS, so each arming
// decides the cadence from live state instead of a value captured at startup.

// The ONE /api/state read. A failed read resolves to an explicitly unavailable
// accounting shape so money surfaces render "Unavailable" rather than a convincing $0
// (fail closed); it never REJECTS, because a transient network blip must not be
// indistinguishable from "there is no budget".
async function readStateSnapshot() {
    try {
        const resp = await apiFetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) return { accounting: { available: false } };
        const data = await resp.json();
        renderProjectsNav(data.projects || [], data.project_chat_ids);
        applyTaskBindings(data.task_bindings || {});
        return data;
    } catch {
        return { accounting: { available: false } };
    }
}

const statePoll = createStatePoll({
    read: readStateSnapshot,
    // An open project THREAD is a chat surface, so it takes the chat cadence:
    // its unread ACK and live header state need the same ~3s freshness Main does.
    activePage: () => (state.activePage === 'thread' ? 'chat' : state.activePage),
    hidden: () => document.hidden,
    setTimer: (fn, ms) => setTimeout(fn, ms),
    clearTimer: (handle) => clearTimeout(handle),
});

const subscribeState = statePoll.subscribe;
const refreshState = () => statePoll.refresh();
const scheduleStatePoll = () => statePoll.schedule();

// Historical name kept because it is the startup barrier's contract and the
// imperative create/delete refresh call-site name.
function refreshProjectsNav() {
    return refreshState();
}

// The poll core owns its timer HANDLE (`createStatePoll` closes over it), so the
// pause is `statePoll.stop()` — the seam the module exports for exactly this.
// Consolidation moved that handle inside the core, which is why no module-scope
// timer variable may be named here: it would be a ReferenceError on every tab
// hide, not a dead line. The static pin in test_navigation_shell_static.py holds
// the old identifier out of this file entirely.
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        statePoll.stop();  // paused, not backed off: a hidden tab spends nothing
        return;
    }
    // Catch up on what was missed; the read's SETTLE re-arms the timer itself.
    refreshState();
});
// Entering/leaving Chat changes the cadence, so re-arm on navigation.
window.addEventListener('ouro:page-shown', () => scheduleStatePoll());

// Sidebar budget meter: `headerBudgetPresentation` stays the ONE budget
// formatting projection (fail-closed "Unavailable", never a fake $0); the bar
// fill is a dynamic CSS custom property, never a .style.width assignment.
function renderSidebarBudget(data) {
    const budget = headerBudgetPresentation(data);
    if (navBudgetAmount) navBudgetAmount.textContent = budget.label;
    if (navBudgetBar) {
        navBudgetBar.dataset.budgetState = budget.state;
        navBudgetBar.style.setProperty('--budget-fill', `${budget.fillPct}%`);
    }
}
subscribeState(renderSidebarBudget);
navBudget?.addEventListener('click', () => openDashboardTab('costs'));

// Brand-row liveness: the green dot mirrors the ONE shared socket's state.
function setBrandOnline(online) {
    if (navBrand) navBrand.dataset.online = online ? 'true' : 'false';
    if (navBrandStatus) navBrandStatus.textContent = online ? 'online' : 'offline';
}
ws.on('open', () => setBrandOnline(true));
ws.on('close', () => setBrandOnline(false));

const ctx = {
    ws,
    state,
    updateUnreadBadge,
    showPage,
    openSettingsTab,
    openDashboardTab,
    isProjectOpening: () => projectPanelOpeningSince > 0,
    setBeforePageLeave: (handler) => {
        if (typeof handler !== 'function') return () => {};
        beforePageLeaveHandlers.push(handler);
        return () => {
            const idx = beforePageLeaveHandlers.indexOf(handler);
            if (idx >= 0) beforePageLeaveHandlers.splice(idx, 1);
        };
    },
    // The ONE /api/state snapshot: subscribe for updates, or force a read after
    // an owner control write. No module opens its own poll timer.
    subscribeState,
    refreshState: () => refreshState(),
    // Right-panel state machine (kinds are mutually exclusive).
    registerRightPanel,
    openRightPanel: (kind, opts) => openRightPanel(kind, opts),
    closeRightPanel: (opts) => closeRightPanel(opts),
};

// The main chat controller is KEPT (it used to be discarded): other screens
// hand ordered composer parts to the chat through `setDraftParts`/`sendParts`
// instead of reaching into chat.js internals.
const chatController = initChat(ctx);

export function getChatController() {
    return chatController;
}
ctx.getChatController = getChatController;

// Empty `changes` page container. The Changes screen module replaces its
// contents; the nav row already routes here through the normal showPage path.
const changesPage = document.createElement('section');
changesPage.id = 'page-changes';
changesPage.className = 'page';
document.getElementById('content')?.appendChild(changesPage);

initFiles(ctx);

// "Is the user typing into this?" — ONE selector list and ONE disabled/readOnly
// rule, shared by the mobile soft-keyboard geometry code (which needs to know
// whether a viewport shrink belongs to a focused field) and by the ⌘L capture
// handler (which must let a focused field keep the keystroke). Returns the
// editable ELEMENT so callers can ask further questions about it — the capture
// handler checks whether it lives inside a capture dock.
const KEYBOARD_EDITABLE_SELECTOR = [
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

function isKeyboardEditable(node) {
    if (!(node instanceof Element)) return null;
    const editable = node.closest(KEYBOARD_EDITABLE_SELECTOR);
    if (!editable) return null;
    if (editable.matches('input, textarea, select') && (editable.disabled || editable.readOnly)) {
        return null;
    }
    return editable;
}

/* [anchor:phase-B] right-panel registrations */
// The Changes screen fills the `page-changes` container created above; the task
// inspector registers itself as the `inspector` right-panel kind and opens on
// `ouro:inspect-task`. Since a thread moved to the CENTRE the panel no longer
// competes with it, so keeping the action to main chat is a scoping decision
// rather than a layout constraint.
// The imports live in this region deliberately: ES module imports are hoisted, so
// keeping them here makes the whole phase-B wiring one append-only block instead
// of a second edit in the shared import header.
import { initChanges } from './modules/changes.js';
import { initTaskInspector } from './modules/task_inspector.js';

// The Changes screen owns its own dock, and the CANCELABLE `ouro:capture-selection`
// event (`[anchor:phase-C]`) is the ONE capture seam: the global handler names the
// active page, the owning page consumes it. Nothing here holds a handle into the
// module, so there is no second path into that dock that could silently diverge.
initChanges(ctx);
initTaskInspector(ctx);

/* [anchor:phase-C] global capture hotkey */
// ⌘L / Ctrl+L = "add what I'm looking at to chat context". This handler knows
// NOTHING about Files or Changes internals: it decides only WHETHER a capture is
// wanted and names the active page in one `ouro:capture-selection` event; the
// page that owns the surface listens and does the capture (files.js today, the
// Changes dock next). With no listener the event is a harmless noop.
//
// The hotkey is best-effort by decision 10 — some browsers reserve ⌘L for the
// address bar and never deliver it — so the always-visible "Add to chat" /
// "Add selection" buttons remain the guaranteed path, not a convenience.
//
// Typing must win: while an editable other than a capture dock has focus, the
// keystroke belongs to that editable. The test is the shared `isKeyboardEditable`
// above — one selector list for the whole app, no second definition to drift.
//
// Suppressing the browser default is conditional on somebody actually CONSUMING
// the capture: the event is dispatched cancelable, and only a listener that calls
// preventDefault (i.e. the page owning the surface really captured) earns the
// `event.preventDefault()` here. With no such listener ⌘L falls through to the
// browser, which is the honest outcome — silently swallowing the address-bar
// shortcut to do nothing would be worse than not handling it.
const CAPTURE_PAGES = new Set(['files', 'changes']);

document.addEventListener('keydown', (event) => {
    if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) return;
    if (String(event.key).toLowerCase() !== 'l') return;
    if (!CAPTURE_PAGES.has(state.activePage)) return;
    const editable = isKeyboardEditable(document.activeElement);
    if (editable && !editable.closest('[data-capture-dock]')) return;
    const request = new CustomEvent('ouro:capture-selection', {
        detail: { page: state.activePage },
        cancelable: true,
    });
    window.dispatchEvent(request);
    if (request.defaultPrevented) event.preventDefault();
});

// ---------------------------------------------------------------------------
// Multi-project navigation + the CENTRE thread stage (project threads, T1).
// Projects come from /api/state; each carries its canonical thread projection,
// and opening a thread mounts a chat instance bound to that thread's chat_id in
// the CENTRE area — the same place Main Chat lives. It used to open as a right
// split panel, which on a phone became a second full-screen overlay stacked on
// the content area; the centre is where a conversation belongs. The right panel
// is now the task inspector's alone.
// Navigation is one state machine: page, thread, and mobile drawer are
// synchronized together so Utilities and a thread can't remain active at once.
// ---------------------------------------------------------------------------
// Single-live-instance policy (P3, owner 7A): at most ONE project thread chat
// instance is alive; closing or switching destroys the previous one. The
// exception is an instance holding unsendable client state (staged File
// attachments / an in-flight upload): it is hidden and marked instead, so
// switching to Settings mid-upload never drops attachments. That survivor rule
// is keyed by THREAD now — two threads of one project are two rooms, and keying
// it by project would have let opening a sibling thread silently discard the
// other thread's staged attachments. Scroll intent survives destruction in a
// small stash keyed the same way and is re-applied after the recreated
// instance's first paint.
const projectScrollStash = new Map();

function destroyProjectInstance(key) {
    const inst = projectInstances.get(key);
    if (!inst) return;
    if (inst.hasPendingWork?.()) {
        inst.page.hidden = true;
        inst.page.dataset.pendingWork = '1';
        inst.cancelHistoryPaint?.();
        return;
    }
    const scroll = inst.getScrollState?.();
    if (scroll) projectScrollStash.set(key, scroll);
    // Release this thread's transcript cache with its instance. It is a paint
    // accelerator the server can rebuild, not durable state — but it is per
    // THREAD and nothing else ever removes it, so leaving it behind lets a long
    // session exhaust the sessionStorage quota, after which every write throws
    // and is swallowed, the DRAFT write included. Dropping the rebuildable copy
    // is what keeps the unrebuildable one (typed-but-unsent text) working.
    forgetThreadTranscriptCache(inst.chatId);
    inst.destroy?.();
    projectInstances.delete(key);
    projectPaintRequests.delete(key);
}

// The right panel is ONE slot for registered kinds (the task inspector today).
// `project` is no longer one of them — a thread opens in the centre — and the
// name stays reserved so a module cannot re-register the retired behaviour.
function registerRightPanel(kind, handlers) {
    const name = String(kind || '').trim();
    if (!name || name === 'project') return () => {};
    rightPanelRegistry[name] = handlers || {};
    return () => {
        if (navState.panelKind === name) closeRightPanel();
        delete rightPanelRegistry[name];
    };
}

async function openRightPanel(kind, opts = {}) {
    const entry = rightPanelRegistry[kind];
    if (!entry || typeof entry.mount !== 'function') return false;
    if (navState.panelKind && navState.panelKind !== kind) closeRightPanel({ sync: false });
    navState.panelKind = kind;
    let mounted = true;
    try {
        mounted = (await entry.mount(opts)) !== false;
    } catch {
        mounted = false;
    }
    if (!mounted) navState.panelKind = null;
    syncNavigationState();
    return mounted;
}

function closeRightPanel({ sync = true } = {}) {
    const kind = navState.panelKind;
    if (kind) {
        try { rightPanelRegistry[kind]?.unmount?.(); } catch {}
    }
    navState.panelKind = null;
    if (sync) syncNavigationState();
}

// The centre stage every thread mounts into. Built once; the header carries the
// title, the per-thread menu and the close affordance so the chat instance
// itself never has to render project chrome.
const threadStage = createThreadStage({
    content: document.getElementById('content'),
    onClose: () => showPage('chat'),
    onMenu: async (anchorEl) => {
        const project = lastProjectRows.find((row) => row.id === navState.activeProjectId);
        if (!project) return;
        const thread = projectThreadRows(project)
            .find((row) => row.id === (navState.activeThreadId ?? MAIN_THREAD_ID));
        if (!thread) return;
        if (thread.id === MAIN_THREAD_ID) {
            openProjectRowMenu(project, await projectRowMenuOptions(project, anchorEl));
        } else {
            openThreadRowMenu(project, thread, { apiClient, anchorEl, onChanged: onProjectsMutated });
        }
    },
});

// Historical name kept: it is what every close call site says. Tearing the open
// thread down IS the close, because the stage hosts exactly one.
function closeProjectPanel({ sync = true } = {}) {
    const openKey = activeThreadKey();
    navState.activeProjectId = null;
    navState.activeThreadId = null;
    if (openKey) destroyProjectInstance(openKey);
    // Anything left is a hidden pending-work survivor; keep it hidden.
    for (const inst of projectInstances.values()) {
        inst.page.hidden = true;
        inst.cancelHistoryPaint?.();
    }
    if (sync) syncNavigationState();
}

/** Open a project's own chat (thread #0) — what clicking a project row does. */
function openProjectPanel(project, options = {}) {
    const thread = project ? projectThreadRows(project)[0] : null;
    return openThread(project, thread, options);
}

async function openThread(project, thread, { closeDrawer = true } = {}) {
    if (!project?.id || String(project.lifecycle || 'active') !== 'active') return;
    if (!thread) return;
    const key = threadKey(project.id, thread.id);
    if (activeThreadKey() === key) {
        closeProjectPanel();
        showPage('chat');
        return;
    }
    // perf2 P4.2: signal chat.js that a thread open is in flight so Main's
    // deferred first hydration yields the CPU to this build/paint.
    projectPanelOpeningSince = Date.now();
    try {
        const moved = await showPage('thread', { closeProject: false, closeDrawer: false });
        if (moved === false) return;
        navState.activeProjectId = project.id;
        navState.activeThreadId = Number(thread.id) || MAIN_THREAD_ID;
        threadStage.setTitle(project, thread);
        // One live instance: every OTHER thread instance is destroyed (or hidden
        // and marked when it holds pending work) before the target is shown.
        for (const other of [...projectInstances.keys()]) {
            if (other !== key) destroyProjectInstance(other);
        }
        let inst = projectInstances.get(key);
        if (!inst) {
            inst = createChatInstance({
                ...ctx,
                chatId: Number(thread.chat_id) || Number(project.chat_id) || 1,
                projectId: project.id,
                // Per THREAD, not per project: the instance namespaces its DOM
                // ids with this prefix, and the single-live-instance policy has
                // one sanctioned exception — a hidden pending-work survivor. Two
                // threads of one project would then be two live subtrees sharing
                // every `#pchat-<pid>-*` id.
                idPrefix: `pchat-${project.id}-${thread.id}`,
                mountEl: threadStage.body,
                // Thread chrome (no global agent controls) in the CENTRE layout —
                // the two used to be one `asPanel` flag (X8).
                layout: 'centre',
                chrome: 'thread',
                title: thread.name || project.name || project.id,
                initialScrollState: projectScrollStash.get(key) || null,
            });
            projectScrollStash.delete(key);
            projectInstances.set(key, inst);
        }
        // A reopened pending-work survivor is live again.
        delete inst.page.dataset.pendingWork;
        for (const [other, instance] of projectInstances) {
            instance.page.hidden = other !== key;
            if (other !== key) instance.cancelHistoryPaint?.();
        }
        if (closeDrawer) navState.mobileDrawerOpen = false;
        syncNavigationState();
        // Restore this thread's scroll instead of leaving it at the top (P7). Runs
        // after the stage is shown so the column has real geometry to scroll.
        inst.restoreScrollPosition?.();
        // ACK only the exact revision whose history was fetched and painted. chat.js
        // owns the paint receipt; an already-painted instance skips the forced
        // refetch — the server clamps the ACK, so no repaint is needed.
        await acknowledgeProjectAfterPaint(project, thread, inst, {
            forcePaint: !inst.hasPaintedHistory?.(),
        });
    } finally {
        projectPanelOpeningSince = 0;
    }
}

// A thread can receive a new visible revision while it stays open. Coalesce
// polling updates per thread, but never acknowledge a newer revision until that
// exact history snapshot has completed a real browser paint.
async function acknowledgeProjectAfterPaint(project, thread, instance = null, { forcePaint = false } = {}) {
    if (!project?.id || !thread) return;
    const key = threadKey(project.id, thread.id);
    if (activeThreadKey() !== key) return;
    const inst = instance || projectInstances.get(key);
    if (!inst || inst.page.hidden) return;
    const revision = Math.max(0, Number(thread.visible_revision) || 0);
    if (!forcePaint && !isThreadUnread(thread, state.projectSeenRevision, project.id)) return;

    const current = projectPaintRequests.get(key);
    if (current && current.revision >= revision) return current.promise;
    inst.cancelHistoryPaint?.();
    const promise = (async () => {
        let paint = null;
        try { paint = await inst.refreshHistory?.({ revision }); } catch {}
        if (
            paint?.painted
            && Number(paint.revision) === revision
            && activeThreadKey() === key
            && !inst.page.hidden
            // A destroyed instance's page reports hidden===false but is detached;
            // a late paint must never ACK a revision nobody saw (GPT#15).
            && inst.page.isConnected
        ) {
            await markProjectViewed(project.id, thread.id, revision);
        }
    })().finally(() => {
        if (projectPaintRequests.get(key)?.promise === promise) {
            projectPaintRequests.delete(key);
        }
    });
    projectPaintRequests.set(key, { revision, promise });
    return promise;
}

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
    // Every active Project is visible, including a newly-created empty room.
    // Unread is a monotonic per-THREAD revision comparison, never a timestamp
    // race. TWO numbers, because the project row is two things at once: `_unread`
    // is the GROUP aggregate that feeds the `#nav-projects-count` pill, and
    // `_mainUnread` is thread #0's own state, which is what the row's dot shows.
    const rows = orderProjectRows(
        all.filter(p => p && p.id && ['active', 'deleting'].includes(String(p.lifecycle || 'active')))
            .map(p => ({
                ...p,
                _unread: unreadThreadCount(p, state.projectSeenRevision),
                _mainUnread: isMainThreadUnread(p, state.projectSeenRevision),
            })),
        projectOrder,
    );
    if (rows.some(p => p.id === navState.activeProjectId && p.lifecycle === 'deleting')) {
        closeProjectPanel();
        showPage('chat');
    }
    // The per-thread tuple carries `lifecycle` and `delete_error` because the
    // paint READS them: `threadRowPresentation` greys a `deleting` row, disables
    // its click and drops its unread dot, and the `Retry delete` menu row shows
    // `delete_error` as the reason it is on offer. Omitted from the fingerprint,
    // both changed invisibly — a rewritten `delete_error` never reached the menu,
    // and an `active -> deleting` transition from another tab or a resumed worker
    // left this tab painting an ordinary full-menu row (I11).
    const json = JSON.stringify(rows.map(p => [
        p.id, p.name, p.chat_id, p.lifecycle, p.visible_revision, p._unread, p._mainUnread, p.delete_error,
        projectThreadRows(p).map(t => [t.id, t.name, t.visible_revision, t.lifecycle, t.delete_error]),
    ]));
    if (json === knownProjectsJson) return;
    knownProjectsJson = json;
    lastProjectRows = rows;
    paintProjectsNav();
    syncNavigationState();
    const active = rows.find((project) => project.id === navState.activeProjectId);
    const openThreadRow = active && projectThreadRows(active)
        .find((thread) => thread.id === (navState.activeThreadId ?? MAIN_THREAD_ID));
    // A rename reaches the sidebar through `projects_changed` -> /api/state, so
    // the open stage's title has to follow it too; otherwise the header keeps
    // showing the old name until the thread is closed and reopened.
    if (openThreadRow) threadStage.setTitle(active, openThreadRow);
    else if (active && navState.activeThreadId !== null) {
        // The open thread left the projection — tombstoned, or archived with
        // nothing live in it. Nothing closed the stage, so the owner was left
        // looking at a room with no row, whose kebab now finds no thread and does
        // nothing at all (I12). The project row does this already, one branch up.
        closeProjectPanel();
        showPage('chat');
    }
    if (openThreadRow && active.lifecycle === 'active'
        && isThreadUnread(openThreadRow, state.projectSeenRevision, active.id)) {
        acknowledgeProjectAfterPaint(active, openThreadRow);
    }
}

// ACK exactly the revision painted, into that THREAD's lane of the nested cursor.
// The server max-merges and clamps against the thread's own visible_revision, so
// stale tabs cannot move it backwards or acknowledge unseen future output.
async function markProjectViewed(projectId, threadId, revision) {
    if (!projectId) return false;
    const seen = Math.max(0, Number(revision) || 0);
    const tid = String(Number(threadId) || MAIN_THREAD_ID);
    try {
        await fetchJson('/api/ui/preferences', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_seen_revision: { [projectId]: { [tid]: seen } } }),
        });
    } catch {
        // The room was painted, but the durable monotonic ACK failed. Keep it
        // unread locally so polling or the next open retries the same revision.
        return false;
    }
    state.projectSeenRevision = rememberSeenRevision(
        state.projectSeenRevision || {}, projectId, threadId, seen,
    );
    if (Array.isArray(lastProjectRows)) {
        let changed = false;
        for (const row of lastProjectRows) {
            if (row.id !== projectId) continue;
            // Both derived numbers, because an ACK into ANY lane can change the
            // pill total while an ACK into lane 0 also clears the row's dot.
            const unread = unreadThreadCount(row, state.projectSeenRevision);
            const mainUnread = isMainThreadUnread(row, state.projectSeenRevision);
            if (row._unread !== unread) { row._unread = unread; changed = true; }
            if (row._mainUnread !== mainUnread) { row._mainUnread = mainUnread; changed = true; }
        }
        if (changed) paintProjectsNav();
    }
    return true;
}

// One writer for the owner's manual sidebar order (D3). It rides the SAME
// /api/ui/preferences surface as widget_order — no second ordering mechanism.
//
// LAST WRITE WINS, deliberately: the patch replaces the WHOLE `project_order` /
// `project_thread_order` key rather than merging per row, so two tabs dragging
// at once leave the loser's arrangement overwritten by the winner's, and the
// loser's sidebar catches up on its next poll. Merging would be worse, not
// better — an order is one list, and interleaving two of them produces an
// arrangement neither owner asked for.
async function persistSidebarOrder(patch) {
    try {
        await fetchJson('/api/ui/preferences', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patch),
        });
    } catch {
        // A failed write leaves the dragged order painted for THIS session only;
        // nothing durable changed, so the next reload shows whatever order was
        // last persisted (the default order if none ever was).
    }
}

// Every registry mutation from a row/thread menu funnels here: an optimistic
// delete paints immediately, everything else re-reads authoritative truth.
function onProjectsMutated(change = {}) {
    if (change.optimistic && change.projectId) {
        const row = lastProjectRows.find(p => p.id === change.projectId);
        if (row) { row.lifecycle = 'deleting'; row._unread = 0; row._mainUnread = false; }
        if (navState.activeProjectId === change.projectId) {
            closeProjectPanel();
            showPage('chat');
        }
        knownProjectsJson = '';
        paintProjectsNav();
        return;
    }
    // A fork hands us the new thread's canonical row. Learn its chat_id
    // SYNCHRONOUSLY, before the refresh: `chat.js::isMyThread` routes an inbound
    // frame by this set, so a thread missing from it has its FIRST frame
    // misrouted to Main — and the refresh below (let alone the poll behind it) is
    // too late to prevent that. The `+ new thread` path already does this inline;
    // fork passed `change.thread` here and had it dropped on the floor.
    const newChatId = Number(change.thread?.chat_id);
    if (newChatId) state.projectChatIds.add(newChatId);
    knownProjectsJson = '';
    refreshProjectsNav();
}

// The project row IS thread #0's row, so its menu carries thread #0's own
// branch/merge/checkout rows as well as the project-level ones. Without them A7
// ("each thread can work in the project folder OR in its own checkout") held for
// every thread EXCEPT the one the project opens by default — a hole no refusal
// would ever mention, because the routes accept thread #0 perfectly well and
// nothing was asking them. Archive/delete are NOT among them: thread #0 has no
// lifecycle of its own and the server refuses it by name, which `threadActions`
// already renders as a disabled row with that reason.
//
// Plus ONE project-level row this module does not own: archived threads. The
// sidebar paints `/api/state`, whose projection hides them, so without a surface
// that can ask for them `restore` was unreachable and archiving a thread was a
// one-way trip (T3R-8/D4). Kept to a disclosure plus a list, in the existing
// row-menu vocabulary — no new screen (P7).
const THREAD_ZERO_MENU_ROWS = ['branch_off', 'merge_back', 'show_changes', 'remove_worktree'];

async function projectRowMenuOptions(project, anchorEl) {
    const zero = projectThreadRows(project)[0] || { id: MAIN_THREAD_ID, name: project.name };
    const { location, inspection, locationError } = await readThreadCheckout(project.id, zero.id);
    return {
        apiClient,
        anchorEl,
        onChanged: onProjectsMutated,
        extraItemsHtml: `${threadActionItemsHtml(zero, location, locationError, THREAD_ZERO_MENU_ROWS)}
        <button type="button" role="menuitem" data-prm="archived_threads">Archived threads…</button>`,
        onExtraSelect: async (action) => {
            if (action === 'archived_threads') {
                await openArchivedThreadsMenu(project, {
                    apiClient, anchorEl, onChanged: onProjectsMutated,
                });
                return true;
            }
            if (!THREAD_ZERO_MENU_ROWS.includes(action)) {
                return false;
            }
            await runThreadAction(action, project, zero, {
                apiClient, onChanged: onProjectsMutated, location, inspection,
            });
            return true;
        },
    };
}

// Paint the collapsible, scrollable projects list from the cached rows.
function paintProjectsNav() {
    const rows = lastProjectRows;
    navProjectsList.textContent = '';
    navProjects.hidden = false;
    // The header pill counts unread THREADS across every project — the number the
    // sidebar's dots actually add up to. Counting projects would have hidden a
    // project with three unread threads behind a "1".
    const unreadCount = rows.reduce((total, project) => total + project._unread, 0);
    if (navProjectsCount) {
        navProjectsCount.textContent = unreadCount ? (unreadCount > 99 ? '99+' : String(unreadCount)) : '';
        navProjectsCount.title = unreadCount ? `${unreadCount} unread thread${unreadCount === 1 ? '' : 's'}` : '';
        if (unreadCount) navProjectsCount.setAttribute('aria-label', navProjectsCount.title);
        else navProjectsCount.removeAttribute('aria-label');
    }
    for (const project of rows) {
        const deleting = String(project.lifecycle || 'active') === 'deleting';
        const item = document.createElement('div');
        item.className = `nav-project-item${deleting ? ' is-deleting' : ''}`;
        item.dataset.projectId = project.id;
        item.draggable = !deleting;
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
        // Thread #0's dot, NOT the group's: this row opens thread #0, so an
        // aggregate dot here would double-count a sibling (whose own row is lit
        // one line below) and could never be cleared by clicking it. The group
        // total is the `#nav-projects-count` pill above.
        if (project._mainUnread && !deleting) {
            const dot = document.createElement('span');
            dot.className = 'nav-unread-dot';
            dot.title = 'New activity';
            btn.appendChild(dot);
            btn.classList.add('has-unread');
        }
        // The action controls are SIBLING buttons, never nested interactive UI
        // inside the row button. `trailing` stays ONE slot node — it now holds
        // the thread "+" and the kebab — so the pinned two-child row markup
        // contract is unchanged.
        let trailing;
        if (deleting) {
            trailing = document.createElement('span');
            trailing.className = 'nav-project-deleting-status';
            trailing.textContent = 'Deleting…';
            trailing.title = project.delete_error || 'Cancellation and cleanup are in progress';
        } else {
            trailing = document.createElement('span');
            trailing.className = 'nav-project-actions';
            const add = document.createElement('button');
            add.type = 'button';
            add.className = 'nav-project-kebab nav-thread-add';
            add.textContent = '+';
            add.title = 'New thread in this project';
            add.setAttribute('aria-label', `New thread in ${project.name || project.id}`);
            add.addEventListener('click', async (event) => {
                event.stopPropagation();
                // A5: "+" creates a thread — an EMPTY chat sharing this project's
                // folder (A2). Its chat_id is learned synchronously so the live
                // fan-out never misroutes the new thread's first frame to Main.
                try {
                    const payload = await apiClient.projectThreadCreate(project.id);
                    const thread = payload?.thread;
                    if (Number(thread?.chat_id)) state.projectChatIds.add(Number(thread.chat_id));
                    knownProjectsJson = '';
                    await refreshProjectsNav();
                    const fresh = lastProjectRows.find((row) => row.id === project.id) || project;
                    if (thread) openThread(fresh, thread);
                } catch (e) {
                    showToast(`Could not create a thread: ${e?.body?.error || e?.message || e}`, 'error');
                }
            });
            const kebab = document.createElement('button');
            kebab.type = 'button';
            kebab.className = 'nav-project-kebab';
            kebab.textContent = '⋯';
            kebab.title = 'Project actions';
            kebab.setAttribute('aria-label', `Actions for ${project.name || project.id}`);
            kebab.addEventListener('click', async (event) => {
                event.stopPropagation();
                openProjectRowMenu(project, await projectRowMenuOptions(project, kebab));
            });
            trailing.append(add, kebab);
        }
        if (project.id === navState.activeProjectId && navState.activeThreadId === MAIN_THREAD_ID) {
            btn.classList.add('active');
        }
        if (!deleting) btn.addEventListener('click', () => openProjectPanel(project));
        item.append(btn, trailing);
        navProjectsList.appendChild(item);
        // The thread list is a SIBLING container after the project row, never a
        // child of it: the row is one button, and interactive thread rows cannot
        // live inside a button.
        const threads = deleting ? null : renderThreadList(project, {
            cursor: state.projectSeenRevision,
            manualOrder: threadOrder[project.id] || [],
            activeThreadKey: activeThreadKey(),
            onOpen: (proj, thread) => openThread(proj, thread),
            onMenu: (proj, thread, anchorEl) => openThreadRowMenu(proj, thread, {
                apiClient, anchorEl, onChanged: onProjectsMutated,
            }),
            onReorder: (pid, ids) => {
                threadOrder = { ...threadOrder, [pid]: ids };
                knownProjectsJson = '';
                paintProjectsNav();
                persistSidebarOrder({ project_thread_order: threadOrder });
            },
        });
        if (threads) navProjectsList.appendChild(threads);
    }
}

// Bound ONCE, not per paint: `#nav-projects-list` is a persistent element, so
// re-attaching here on every repaint would stack a new set of drag listeners on
// it each time /api/state changed anything (a thread list is rebuilt from
// scratch each paint, so its own listeners go with it). The rows are found by
// selector at drag time, so a rebuilt list needs no rebinding.
attachProjectReorder(navProjectsList, (ids) => {
    projectOrder = ids;
    knownProjectsJson = '';
    // The manual order is applied where the rows are BUILT (`renderProjectsNav`
    // -> `orderProjectRows`); `paintProjectsNav` paints `lastProjectRows`
    // verbatim. So the drop has to reorder the cache itself, or the row snaps
    // back to where it was until the next /api/state poll repaints it (3s on a
    // chat/thread page, 20s elsewhere). Threads need no equivalent because
    // `renderThreadList` applies `manualOrder` at paint time.
    lastProjectRows = applyManualOrder(lastProjectRows, projectOrder, (row) => String(row.id));
    paintProjectsNav();
    persistSidebarOrder({ project_order: projectOrder });
});


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

// Resizable side sections: the edge drag-handle writes --sidebar-width on :root
// and persists (debounced) to /api/ui/preferences. The project thread lost its
// resizable column when it moved to the CENTRE (the centre is sized by the shell
// grid), so `project_panel_width` is no longer read; the key stays accepted by
// the preferences contract so an older stored value is not an error.
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
}

apiFetch('/api/ui/preferences', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : null))
    .then((prefs) => {
        // The cursor is nested per thread (T1). Normalizing HERE too — not only on
        // the server — is what lets a browser that reads a flat value written by an
        // older runtime agree with it instead of showing every project as unread.
        state.projectSeenRevision = normalizeSeenRevision(prefs && prefs.project_seen_revision);
        projectOrder = Array.isArray(prefs?.project_order) ? prefs.project_order.map(String) : [];
        threadOrder = (prefs && typeof prefs.project_thread_order === 'object' && prefs.project_thread_order)
            ? prefs.project_thread_order : {};
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

loadVersion();
syncNavigationState();

// Mobile soft-keyboard handling: viewport shrink counts only while an editable
// owns focus. Drawer opening clears that state explicitly before navigation is
// rendered, so stale WebView geometry cannot hide an otherwise-open sidebar.
(function () {
    const vvhStyle = document.createElement('style');
    vvhStyle.id = 'runtime-vvh';
    document.head.appendChild(vvhStyle);

    let keyboardOpen = false;
    let keyboardTouchStartY = 0;
    let stableViewportHeight = 0;
    let focusBaselineHeight = 0;
    let focusRevision = 0;
    let baselineFrame = 0;

    // The editable test is the module-scope `isKeyboardEditable` (shared with the
    // ⌘L capture handler); this IIFE only asks the question about focus.
    function focusedKeyboardEditable() {
        return isKeyboardEditable(document.activeElement);
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
        if (!isKeyboardEditable(event.target)) return;
        focusRevision += 1;
        cancelBaselineFrame();
        focusBaselineHeight = stableViewportHeight || viewportHeight();
        updateVvh();
    });
    document.addEventListener('focusout', (event) => {
        if (!isKeyboardEditable(event.target)) return;
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

// Populate the project-thread isolation set BEFORE opening the socket so the live
// fan-out never misclassifies an early project frame as main-chat traffic during
// startup (chat.js::isMyThread relies on state.projectChatIds). Connect even if
// the prefetch fails, then ws.on('open') keeps it fresh.
refreshProjectsNav().finally(() => ws.connect());
