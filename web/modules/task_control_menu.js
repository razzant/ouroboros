// S3 (Q2/HQ1): the shared three-action task stop/hurry control.
//
// One dropdown of exactly three owner-decided actions — "Wrap up"
// (soft finalize-then-stop), "Hurry up" (typed task-local hurry control,
// NO chat message ever), "Stop now" (hard stop). Dismissing the
// menu continues the run (the dismiss affordance replaced the old separate
// "keep running" confirm). While a cancel intent is already pending, the only
// offered action is "Stop now" — the monotonic escalation of the
// SAME durable stop intent; "Hurry up" is refused then and never offered.
//
// Chat live cards and the Activity tab consume the SAME module (owner
// product-wide parity), so eligibility gates differ per surface but the
// actions, endpoint bindings, request-id retry, and refusals do not.

import { cancelTask, hurryTask, resumeTask } from './api_client.js';
import { showToast } from './toast.js';

export const ACTION_FINALIZE = 'finalize';
export const ACTION_HURRY = 'hurry';
export const ACTION_RESUME = 'resume';
export const ACTION_STOP_NOW = 'stop_now';

// Logical slots that may host multiple independent cycles (v6.82: shared so
// control eligibility and the chat card layer read the same truth).
export const REUSABLE_TASK_IDS = new Set(['bg-consciousness', 'active']);

/**
 * v6.82 (P5): may this live card offer the stop/hurry control?
 * Card shape alone cannot answer it — a subagent narration replayed without
 * its lineage would mint a root-shaped card with a live Cancel. So eligibility
 * requires the supervisor's host-attested `cancelable` progress-meta marker on
 * top of the structural gates: a ROOT (non-subagent) card, not a reusable
 * slot, not finished, not converted into a project chip. The marker is stamped
 * from the ONE ownership seam the cancel endpoint itself consults — a pooled
 * root's RUNNING row, or the in-process direct-chat turn (which has no queue
 * row but is stopped cooperatively through the same owner mailbox) — so a
 * card that shows the control is a task the endpoint will actually stop.
 */
export function cancelRunEligibility({
    groupId = '', isSubagent = false, finished = false, cancelable = false, converted = false,
} = {}) {
    return Boolean(cancelable)
        && !isSubagent
        && !finished
        && !converted
        && Boolean(String(groupId || '').trim())
        && !REUSABLE_TASK_IDS.has(String(groupId || ''));
}

// Frozen owner wording (Q2/HQ1) — exact strings, never localized/reworded here.
export const TASK_CONTROL_LABELS = Object.freeze({
    [ACTION_FINALIZE]: 'Wrap up',
    [ACTION_HURRY]: 'Hurry up',
    [ACTION_RESUME]: 'Resume',
    [ACTION_STOP_NOW]: 'Stop now',
});

/**
 * The action set for the current card state (pure — node-testable).
 * @param {{cancelPending?: boolean}} [state]
 * @returns {string[]} ordered action ids
 */
export function taskControlActions({ cancelPending = false, budgetPaused = false } = {}) {
    // A pending cancel refuses hurry (HQ1) and a second soft stop is a no-op:
    // the single offered action is the hard escalation of the same intent.
    if (cancelPending) return [ACTION_STOP_NOW];
    // A budget-paused member is not running: nothing to wrap up or hurry.
    // The host-attested pause fact gates the offer; the server re-validates
    // (replay_unsafe and sibling checks answer 409 with the reason).
    if (budgetPaused) return [ACTION_RESUME, ACTION_STOP_NOW];
    return [ACTION_FINALIZE, ACTION_HURRY, ACTION_STOP_NOW];
}

export async function resumeTaskAction(taskId) {
    const id = String(taskId || '');
    if (!id || inFlight.has(id)) return;
    inFlight.add(id);
    try {
        await resumeTask(id);
        showToast('Resuming: the task returns to the queue.', 'info');
    } catch (exc) {
        // The server names the refusal (replay_unsafe / fence missing /
        // not budget-paused): show it verbatim instead of a generic failure.
        // Handled here, never rethrown: the menu callback is fire-and-forget.
        showToast(`Resume refused: ${exc?.message || exc}`, 'error');
    } finally {
        inFlight.delete(id);
    }
}

/**
 * Map a stop action to the wire stop_policy (empty = not a stop action).
 * @param {string} action
 * @returns {string}
 */
export function stopPolicyFor(action) {
    if (action === ACTION_FINALIZE) return 'finalize_then_cancel';
    if (action === ACTION_STOP_NOW) return 'immediate';
    return '';
}

// Stable per-task hurry request id (HQ1): a retry of the SAME click reuses the
// id so the endpoint acknowledges idempotently instead of minting a second
// typed control. Page-session scoped — a reload is a new owner intent.
const hurryRequestIds = new Map();

export function hurryRequestId(taskId) {
    const id = String(taskId || '').trim();
    if (!hurryRequestIds.has(id)) {
        const uuid = (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function')
            ? globalThis.crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        hurryRequestIds.set(id, `hurry-${uuid}`);
    }
    return hurryRequestIds.get(id);
}

// One action in flight per task: the menu disables items while a request is
// awaiting, so a double click cannot race two controls.
const inFlight = new Set();

export function taskControlBusy(taskId) {
    return inFlight.has(String(taskId || '').trim());
}

/**
 * Submit the typed task-local hurry control (HQ1): body carries ONLY the
 * stable request_id — no text field exists and no chat message is created
 * anywhere on this path. Resolves with the acknowledgement (duplicate=true is
 * the idempotent success shape); a typed refusal rejects.
 * @param {string} taskId
 * @returns {Promise<import('./api_types.js').TaskHurryResponse>}
 */
export async function requestHurry(taskId) {
    const id = String(taskId || '').trim();
    inFlight.add(id);
    try {
        return await hurryTask(id, hurryRequestId(id));
    } finally {
        inFlight.delete(id);
    }
}

/**
 * The COMPLETE "Hurry up" flow both surfaces share (HQ1): submit the typed
 * control, acknowledge via LOCAL toast only — success, idempotent duplicate,
 * or a visible typed refusal (e.g. a pending cancel). Never a chat message.
 * @param {string} taskId
 * @returns {Promise<boolean>} whether the control was accepted
 */
export async function hurryTaskAction(taskId) {
    try {
        const ack = await requestHurry(taskId);
        showToast(ack?.duplicate
            ? 'Hurry up: already accepted for this task.'
            : 'Hurry up: accepted — the task will speed up at the next boundary.', 'ok');
        return true;
    } catch (exc) {
        showToast(`Hurry up: refused — ${exc?.message || exc}`, 'error');
        return false;
    }
}

/**
 * Submit a stop action ({@link ACTION_FINALIZE} or {@link ACTION_STOP_NOW}).
 * Both surfaces cancel the task AND its live subtree (v6.82 semantics kept).
 * @param {string} taskId
 * @param {string} action
 * @returns {Promise<import('./api_types.js').TaskCancelResponse>}
 */
export async function requestStop(taskId, action) {
    const id = String(taskId || '').trim();
    inFlight.add(id);
    try {
        return await cancelTask(id, { cascade: true, stopPolicy: stopPolicyFor(action) });
    } finally {
        inFlight.delete(id);
    }
}

// ---------------------------------------------------------------------------
// Dropdown DOM (shared by Chat live cards and the Activity tab)
// ---------------------------------------------------------------------------

let openMenu = null;
let openTrigger = null;
let stopMenuWatcher = null;
let focusReturnTimer = null;

const MENU_MARGIN = 8;
const MENU_GAP = 4;

function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

function positionTaskControlMenu(anchor, menu) {
    if (!anchor?.isConnected || !menu?.isConnected) {
        closeTaskControlMenu();
        return false;
    }

    const anchorRect = anchor.getBoundingClientRect();
    if (anchorRect.width <= 0 || anchorRect.height <= 0) {
        closeTaskControlMenu();
        return false;
    }

    const menuRect = menu.getBoundingClientRect();
    // scrollHeight preserves the intrinsic item height if the viewport-level
    // fallback max-height has already constrained the rendered border box.
    const borderHeight = Math.max(0, menuRect.height - menu.clientHeight);
    const naturalHeight = Math.max(menuRect.height, menu.scrollHeight + borderHeight);
    const spaceAbove = Math.max(0, anchorRect.top - MENU_GAP - MENU_MARGIN);
    const spaceBelow = Math.max(
        0,
        window.innerHeight - MENU_MARGIN - anchorRect.bottom - MENU_GAP,
    );
    const openUpwards = spaceBelow < naturalHeight && spaceAbove > spaceBelow;
    const availableHeight = openUpwards ? spaceAbove : spaceBelow;
    const effectiveHeight = Math.min(naturalHeight, availableHeight);
    const top = openUpwards
        ? Math.max(MENU_MARGIN, anchorRect.top - MENU_GAP - effectiveHeight)
        : anchorRect.bottom + MENU_GAP;
    const left = clamp(
        anchorRect.right - menuRect.width,
        MENU_MARGIN,
        window.innerWidth - menuRect.width - MENU_MARGIN,
    );

    menu.style.setProperty('--tcm-top', `${Math.round(top)}px`);
    menu.style.setProperty('--tcm-left', `${Math.round(left)}px`);
    menu.style.setProperty('--tcm-max-height', `${Math.floor(availableHeight)}px`);
    return true;
}

function watchTaskControlMenu(anchor, menu) {
    const isCurrentMenu = () => openMenu === menu && openTrigger === anchor;
    const closeForViewportChange = (event) => {
        // A height-clamped portal is its own scrollport. Scrolling it does not
        // move the trigger, so keep it open; ancestor/page scroll still closes.
        if (event.type === 'scroll' && event.target === menu) return;
        if (isCurrentMenu()) closeTaskControlMenu();
    };
    document.addEventListener('scroll', closeForViewportChange, true);
    window.addEventListener('resize', closeForViewportChange);

    const observer = new IntersectionObserver((entries) => {
        if (!isCurrentMenu()) return;
        const entry = entries.find((candidate) => candidate.target === anchor);
        if (!anchor.isConnected || (entry && !entry.isIntersecting)) {
            closeTaskControlMenu();
        }
    }, { threshold: 0 });
    observer.observe(anchor);

    return () => {
        observer.takeRecords();
        observer.disconnect();
        document.removeEventListener('scroll', closeForViewportChange, true);
        window.removeEventListener('resize', closeForViewportChange);
    };
}

function clearPendingFocusReturn() {
    if (focusReturnTimer !== null) window.clearTimeout(focusReturnTimer);
    focusReturnTimer = null;
}

function finishOutsidePointerFocusReturn(trigger) {
    clearPendingFocusReturn();
    focusReturnTimer = window.setTimeout(() => {
        focusReturnTimer = null;
        if (!openMenu && trigger?.isConnected) trigger.focus();
    }, 0);
}

export function closeTaskControlMenu() {
    clearPendingFocusReturn();
    const menu = openMenu;
    const trigger = openTrigger;
    const restoreFocus = Boolean(menu?.contains(document.activeElement));
    const stopWatching = stopMenuWatcher;
    stopMenuWatcher = null;
    stopWatching?.();
    menu?.remove();
    openMenu = null;
    trigger?.setAttribute?.('aria-expanded', 'false');
    openTrigger = null;
    document.removeEventListener('pointerdown', onOutsidePointer, true);
    document.removeEventListener('keydown', onMenuKeydown, true);
    if (restoreFocus && trigger?.isConnected) trigger.focus();
}

function onOutsidePointer(event) {
    if (!openMenu || openMenu.contains(event.target)) return;
    const trigger = openTrigger;
    const restoreFocus = openMenu.contains(document.activeElement);
    closeTaskControlMenu();
    // The pointerdown default focuses the clicked element after capture-phase
    // listeners run. Restore once more in the next task without cancelling the
    // outside element's click; a successor task menu cancels this stale timer.
    if (restoreFocus && trigger?.isConnected) finishOutsidePointerFocusReturn(trigger);
}

function onMenuKeydown(event) {
    // Dismiss = continue the run (Q2: the dismiss affordance replaced the old
    // explicit "keep running" item).
    if (event.key === 'Escape') {
        event.preventDefault();
        closeTaskControlMenu();
    }
}

/**
 * Open the three-action dropdown in a viewport-level portal anchored to `anchor`.
 * Dismissing (outside click / Escape) continues the run. Selecting an item
 * closes the menu and invokes `onAction(actionId)`.
 * @param {HTMLElement} anchor trigger element
 * @param {{cancelPending?: boolean, busy?: boolean, onAction: (action: string) => void}} opts
 */
export function openTaskControlMenu(anchor, { cancelPending = false, budgetPaused = false, busy = false, onAction } = {}) {
    closeTaskControlMenu();
    if (!anchor?.isConnected || !document.body) return null;
    // A11y: the trigger owns a popup menu; expanded tracks the open state.
    anchor.setAttribute('aria-haspopup', 'menu');
    anchor.setAttribute('aria-expanded', 'true');
    const menu = document.createElement('div');
    menu.className = 'task-control-menu';
    menu.setAttribute('role', 'menu');
    for (const action of taskControlActions({ cancelPending, budgetPaused })) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `task-control-item${action === ACTION_STOP_NOW ? ' danger' : ''}`;
        item.dataset.taskControl = action;
        item.setAttribute('role', 'menuitem');
        item.textContent = TASK_CONTROL_LABELS[action];
        if (busy) item.disabled = true;
        item.addEventListener('click', (event) => {
            event.stopPropagation();
            closeTaskControlMenu();
            onAction?.(action);
        });
        menu.appendChild(item);
    }
    document.body.appendChild(menu);
    openMenu = menu;
    openTrigger = anchor;
    stopMenuWatcher = watchTaskControlMenu(anchor, menu);
    if (!positionTaskControlMenu(anchor, menu)) return null;
    // A11y: keyboard users land on the first actionable item on open.
    menu.querySelector('button:not(:disabled)')?.focus?.();
    document.addEventListener('pointerdown', onOutsidePointer, true);
    document.addEventListener('keydown', onMenuKeydown, true);
    return menu;
}

// The trigger's constant label. The trigger always opens the dropdown; the
// pending-cancel state changes the OFFERED actions (escalation only), not the
// interaction shape — so an accidental click never hard-stops directly.
export const TASK_CONTROL_TRIGGER_LABEL = 'Stop…';
