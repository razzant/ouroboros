// The shared task control menu used by Chat live cards and the Activity tab:
// "Wrap up" (soft finalize-then-stop), "Hurry up" (typed task-local hurry
// control, never a chat message), "Pause"/"Resume" (root-only, whole tree; the
// request_id survives uncertain or latch_pending replies until a confirmed
// Pause or Resume retires it), "Stop earlier attempt" (a work's earlier
// Continue attempt that still owns live helpers) and "Stop now" (hard stop).
// Dismissing the menu continues the run. While a cancel intent is pending, the
// only offered action is "Stop now", the monotonic escalation of the same
// durable stop intent; "Hurry up" is refused then and never offered.
//
// Which actions a work offers is the server's decision (`/api/state` works,
// work_status.js lays them on this menu); endpoints, stop policy, request-id
// retry and verbatim refusals are shared.

import { cancelTask, hurryTask, pauseTask, resumeTask } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { haptic } from './haptics.js';
import { showToast } from './toast.js';
import { fmt, tr } from './i18n.js';

export const ACTION_FINALIZE = 'finalize';
export const ACTION_HURRY = 'hurry';
export const ACTION_PAUSE = 'pause';
export const ACTION_RESUME = 'resume';
export const ACTION_STOP_EARLIER = 'stop_earlier';
export const ACTION_STOP_NOW = 'stop_now';

// Logical slots that may host multiple independent cycles (v6.82: shared so
// control eligibility and the chat card layer read the same truth). A
// consciousness wake-up is NOT one of them: it is an ordinary direct turn with
// its own task id and its own durable result.
export const REUSABLE_TASK_IDS = new Set(['active']);

/**
 * v6.82 (P5): may this live card offer the stop/hurry control?
 * Card shape alone cannot answer it — a subagent narration replayed without
 * its lineage would mint a root-shaped card with a live Cancel. So eligibility
 * requires the supervisor's host-attested `cancelable` marker or current census on
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

// The direct button and the menu use the same request owner. Only a work's Resume control enables this door
// (the server offers it where the Resume door accepts now); the caller re-checks that control on click.
export function createTaskResumeButton(taskId, { canResume, onSettled = () => {} } = {}) {
    const button = document.createElement('button');
    button.type = 'button';
    // Resume is the one primary of a paused card (its Stop… beside it stays neutral).
    button.className = 'btn btn-sm btn-primary ui-hit-area';
    button.dataset.resumeRun = '1';
    button.textContent = tr('task.control.resume', 'Resume');
    button.addEventListener('click', async (event) => {
        event.stopPropagation();
        if (!canResume?.() || taskControlBusy(taskId)) return;
        button.disabled = true;
        try { await resumeTaskAction(taskId); await onSettled(); }
        finally { button.disabled = false; }
    });
    return button;
}

// Frozen owner wording (Q2/HQ1) — exact strings, never localized/reworded here.
// "Stop earlier attempt" stops what still runs of a held Continue's earlier attempt; the Continue itself stays.
export const TASK_CONTROL_LABELS = Object.freeze({
    [ACTION_FINALIZE]: 'Wrap up',
    [ACTION_HURRY]: 'Hurry up',
    [ACTION_PAUSE]: 'Pause',
    [ACTION_RESUME]: 'Resume',
    [ACTION_STOP_EARLIER]: 'Stop earlier attempt',
    [ACTION_STOP_NOW]: 'Stop now',
});

/**
 * Whether a queue row's task is the ROOT of its tree — the server's own rule
 * (`task_results.resolve_task_lineage`), so a child never offers the whole-tree
 * Pause the server would refuse (`not_a_root_task`) or aim at its parent.
 * @param {object} task queue row task
 * @param {string} id the row's task id
 */
export function isRootTaskRow(task = {}, id = '') {
    const field = (key) => String(task?.[key] ?? task?.metadata?.[key] ?? '').trim();
    const taskId = String(id || task?.id || '').trim();
    const root = field('root_task_id') || taskId;
    const role = field('delegation_role').toLowerCase();
    const original = field('original_task_id');
    if (!taskId || field('parent_task_id')) return false;
    return root === taskId ? role !== 'subagent'
        : role === 'root' && Boolean(original) && original === field('timeout_retry_from') && original !== taskId;
}

/**
 * The action set for the current card state (pure — node-testable).
 * `wholeTree` is false for a child: its Pause would not name its own tree. `resume` offers Resume in the menu:
 * only where a work's Resume control targets this task and no Resume button of its own stands beside it.
 * @param {{cancelPending?: boolean, budgetPaused?: boolean, wholeTree?: boolean, resume?: boolean}} [state]
 * @returns {string[]} ordered action ids
 */
export function taskControlActions({ cancelPending = false, budgetPaused = false, wholeTree = true, resume = false } = {}) {
    // A pending cancel refuses hurry (HQ1) and a second soft stop is a no-op:
    // the single offered action is the hard escalation of the same intent.
    if (cancelPending) return [ACTION_STOP_NOW];
    // A budget-paused member is not running: nothing to wrap up or hurry.
    // The host-attested pause fact gates the offer; the server re-validates
    // (replay_unsafe and sibling checks answer 409 with the reason).
    if (budgetPaused) return resume ? [ACTION_RESUME, ACTION_STOP_NOW] : [ACTION_STOP_NOW];
    // Pause saves the WHOLE tree until an explicit Resume: new work is fenced,
    // members' own runs get a stop, launched reviewers finish; Stop still ends it.
    return wholeTree ? [ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_NOW]
        : [ACTION_FINALIZE, ACTION_HURRY, ACTION_STOP_NOW];
}

// Retain an uncertain Pause's id for retries; acknowledgement or Resume
// retires it, so a later intentional Pause is a new action.
const pauseRequestIds = new Map();

function pauseRequestId(taskId) {
    const id = String(taskId || '').trim();
    if (!pauseRequestIds.has(id)) {
        const uuid = (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function')
            ? globalThis.crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        pauseRequestIds.set(id, `pause-${uuid}`);
    }
    return pauseRequestIds.get(id);
}

// A durable Pause whose queue latch waited on a busy launch lock (202
// `latch_pending`): the SAME id completes it. Bounded; never a new action.
const PAUSE_LATCH_RETRY_MS = [250, 500, 1000, 2000];
const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/**
 * The shared "Pause" flow: one typed request, a local toast naming the
 * truthful tree state, a typed refusal shown verbatim. Never a chat message.
 * @param {string} taskId
 * @param {{pause?: Function, toast?: Function, wait?: Function}} [deps]
 * @returns {Promise<boolean>} whether the pause was accepted
 */
export async function pauseTaskAction(taskId, { pause = pauseTask, toast = showToast, wait = sleep } = {}) {
    const id = String(taskId || '').trim();
    if (!id || inFlight.has(id)) return false;
    inFlight.add(id);
    try {
        const requestId = pauseRequestId(id);
        let ack = await pause(id, requestId);
        if (ack?.ok !== true) throw new Error('Pause acknowledgement is unknown; retry this action.');
        // From here the fence is durable: a failed completion is never a refusal.
        for (const delay of PAUSE_LATCH_RETRY_MS) {
            if (!ack.latch_pending) break;
            await wait(delay);
            try {
                const next = await pause(id, requestId);
                if (next?.ok === true) ack = next;
            } catch { /* keep the durable acknowledgement and its id */ }
        }
        if (ack.latch_pending) {
            toast('Pausing: new work is stopped. The queue part of this Pause is not confirmed yet; '
                + 'press Pause again to finish the same action.', 'info');
            return true;
        }
        pauseRequestIds.delete(id);
        if (ack.state === 'released') {
            toast('This Pause was already resumed.', 'info');
            return true;
        }
        toast(ack.state === 'paused'
            ? 'Paused: the whole task tree is saved until you resume it.'
            // Owner S1: running work stops at its last saved point (a model answer
            // still awaited is abandoned, not awaited); started reviews finish apart.
            : 'Pausing: new work is stopped; running work stops at its last saved point. '
                + 'Reviews already started finish separately.', 'ok');
        return true;
    } catch (exc) {
        toast(`Pause refused: ${exc?.message || exc}`, 'error');
        return false;
    } finally {
        inFlight.delete(id);
    }
}

// A held Continue's observed blockers in the owner's words (#1721), from the
// server's census (supervisor/continuation_admission.py). Facts only: a start
// with an unknown outcome is not called running or stoppable, and a kind with no
// sentence yet stays raw (DESIGN §4). The machine record keeps the codes.
const HELD_FACTS = {
    running_member: (b) => `task ${b.task_id} is still running`,
    dispatchable_member: (b) => `task ${b.task_id} is queued to run`,
    owned_process: (b) => `a process of task ${b.task_id} is still alive`,
    // Mirrors work_stop.model_handoff_words (local_custody_repair.receiver_reason).
    model_handoff: (b) => ({
        local_call_live: `a model answer for task ${b.task_id} is awaited by a live worker`,
        local_completion_unknown: `it is unknown whether the model answer receiver of task ${b.task_id} ended: `
            + 'its worker cannot be proven ended',
    })[b.reason] || 'a model request has no recorded result',
    // Mirrors work_stop.tool_handoff_words: the server classifies (tool_custody.tool_handoff_reason).
    tool_handoff: (b) => {
        const tool = b.tool || 'tool';
        const effects = (Array.isArray(b.effects) && b.effects.length ? b.effects.join(' and ') : 'outside effect');
        return ({
            local_call_live: `the ${tool} call of task ${b.task_id} is in a live worker: still running, `
                + 'or its finished result is not recorded yet',
            local_completion_unknown: `it is unknown whether the ${tool} call of task ${b.task_id} finished: `
                + 'its worker cannot be proven ended',
            owner_ended_effect_unknown: `the ${tool} call of task ${b.task_id} ended with its worker; `
                + `the outcome of its ${effects} is unknown`,
            returned_effect_unconfirmed: `the ${tool} call of task ${b.task_id} returned; its ${effects} `
                + `${Array.isArray(b.effects) && b.effects.length > 1 ? 'are' : 'is'} not confirmed`,
        })[b.reason] || `a ${tool} call of task ${b.task_id} is held`;
    },
    merge_operation: (b) => `a merge of task ${b.task_id} is not settled`,
    workspace_executor: (b) => `a workspace process of task ${b.task_id} is not settled`,
    delegated_run: (b) => (b.state === 'running' ? `delegated run ${b.run_id} is running`
        : b.stop_outcome === 'pending_invocation_unbound'
            ? 'a delegated start has an unknown outcome (no run was recorded)'
            : b.state === 'stop_requested' ? `delegated run ${b.run_id}: its stop is requested, not confirmed`
                : `delegated run ${b.run_id || b.invocation_id}: its outcome is unknown`),
};

/** @returns {string} the distinct observed facts, '' when the census named none */
export function heldWorkFacts(blockers, limit = 3) {
    const facts = [...new Set((Array.isArray(blockers) ? blockers : []).filter((b) => b && typeof b === 'object')
        .map((b) => (HELD_FACTS[b.kind] ? HELD_FACTS[b.kind](b)
            : /_unreadable$/.test(String(b.kind)) ? 'its custody record could not be read'
                : String(b.kind || '').startsWith('late_phase_') ? `its work after the answer is ${b.kind.slice(11)}`
                    : String(b.kind || 'unknown'))))];
    return facts.length > limit
        ? `${facts.slice(0, limit).join('; ')}; and ${facts.length - limit} more` : facts.join('; ');
}

// The damaged saved records an accepted Continue or Resume did not wait for (`membership_gaps`: the census
// tied them to no task; at most 20 named, all counted). The Pause row says the same fact in the transcript
// (ouroboros/pause_notices.py). Named, never called another task's; whether they belong here is unknown.
const GAP_WORDS = {
    continue: ['Continue did not wait for {n} saved task record that could not be read and that nothing ties to '
        + 'this task ({names}); whether it belongs here is unknown.', 'Continue did not wait for {n} saved task '
        + 'records that could not be read and that nothing ties to this task ({names}); whether they belong here is unknown.'],
    resume: ['The Pause did not wait for {n} saved task record that could not be read and that nothing ties to '
        + 'this task ({names}); whether it belongs here is unknown.', 'The Pause did not wait for {n} saved task '
        + 'records that could not be read and that nothing ties to this task ({names}); whether they belong here is unknown.'],
};

/** @returns {string} the gap an accepted `action` ('continue' | 'resume') answered with, '' without one */
export function membershipGapWords(gaps, action) {
    const count = gaps?.count;
    if (!Number.isInteger(count) || count <= 0 || !GAP_WORDS[action]) return '';
    const names = (Array.isArray(gaps.rows) ? gaps.rows : []).map((row) => String(row?.task_id || '')).filter(Boolean);
    return fmt(GAP_WORDS[action][count === 1 ? 0 : 1],
        { n: count, names: [...names, ...(count > names.length ? ['…'] : [])].join(', ') });
}

/** A warning beside the action's own toast, kept until dismissed: a Continue's gap, or one found after the Pause
 *  row, has no other on-screen home (docs/DESIGN.md: a warning whose line is its only home stays). */
export function announceMembershipGap(gaps, action, toast = showToast) {
    const words = membershipGapWords(gaps, action);
    if (words) toast(words, 'warn', { ttl: 0, dismissLabel: 'Dismiss notice' });
    return words;
}

const KEPT_PAUSED = Symbol('kept_paused');

// Every holder is a held call whose outcome is unknown (the server lists their ids): the owner may go on
// without their results. They are never run again; the task records the decision (local_custody_repair).
async function resumeWithoutUnknown(id, refusal, { resume, askOwner }) {
    const named = refusal?.body?.acceptable_unknown;
    if (!Array.isArray(named) || !named.length) throw refusal;
    const facts = heldWorkFacts(refusal.body.blockers, 5);
    const one = named.length === 1;
    const confirmed = await askOwner({
        title: 'Resume without these results?',
        body: `${facts.charAt(0).toUpperCase()}${facts.slice(1)}. If you resume, ${one ? 'this call is' : 'these calls are'} `
            + `never run again, whatever ${one ? 'it' : 'they'} changed outside stays as it is, `
            + 'and your decision is recorded on the task.',
        confirmLabel: 'Resume anyway',
    });
    return confirmed ? resume(id, { accept_unknown: named, reason: 'The owner resumed without these results.' })
        : KEPT_PAUSED;
}

export async function resumeTaskAction(taskId, { resume = resumeTask, toast = showToast, askOwner = openConfirmDialog } = {}) {
    const id = String(taskId || '').trim();
    if (!id || inFlight.has(id)) return;
    inFlight.add(id);
    try {
        const ack = await resume(id).catch((refusal) => resumeWithoutUnknown(id, refusal, { resume, askOwner }));
        if (ack === KEPT_PAUSED) return;
        if (ack?.ok !== true) throw new Error('Resume acknowledgement is unknown; retry this action.');
        pauseRequestIds.delete(id);
        // An answered root's late work (D10) continues in place; its answer stays delivered.
        toast(ack.tree_only ? 'Resumed. Ouroboros or you choose which helpers continue.'
            : ack.late_phase ? 'Resuming: the work left after the delivered answer continues.'
            : 'Resuming: the task returns to the queue.', 'info');
        announceMembershipGap(ack.membership_gaps, 'resume', toast);  // read from the Pause's current record
    } catch (exc) {
        // The server names the refusal (replay_unsafe / fence missing /
        // not budget-paused): show it verbatim instead of a generic failure,
        // and a held Continue's observed blockers as facts (#1721).
        // Handled here, never rethrown: the menu callback is fire-and-forget.
        const facts = heldWorkFacts(exc?.body?.blockers);
        toast(facts ? `Resume refused: ${facts}.${exc.body.action === 'wait_or_stop_the_previous_work'
            ? ' Wait for that work or stop it.' : ''}` : exc?.body?.message || `Resume refused: ${exc?.message || exc}`, 'error');
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
const pendingStopActions = new Map();

export async function requestStop(taskId, action) {
    const id = String(taskId || '').trim();
    const key = JSON.stringify([id, action]);
    // A lost response keeps this page-session action retryable. A confirmed
    // response retires it; a later click is a genuinely new Stop. Hardening a
    // soft Stop is a distinct action, even while the soft response is unknown.
    if (!pendingStopActions.has(key)) {
        const uuid = globalThis.crypto?.randomUUID?.()
            || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
        pendingStopActions.set(key, `stop-${uuid}`);
    }
    inFlight.add(id);
    try {
        const result = await cancelTask(id, { cascade: true, stopPolicy: stopPolicyFor(action),
            stopActionId: pendingStopActions.get(key) });
        if (result?.ok !== true) throw new Error('Stop acknowledgement is unknown; retry this action.');
        pendingStopActions.delete(key);
        return result;
    } finally {
        inFlight.delete(id);
    }
}

/**
 * One owner Stop over a work's targets, in the server's order (a held Continue before its earlier attempt): each
 * target goes through the existing cascade door (`stopOne`, a card's own path where the caller has one) and the
 * owner hears one combined result. A target that already ended (404) is settled, not a failure.
 * @returns {Promise<boolean>} whether every target accepted its Stop
 */
export async function requestStopAll(targets, action, { stopOne = requestStop, toast = showToast } = {}) {
    const ids = [...new Set((Array.isArray(targets) ? targets : []).map((id) => String(id || '').trim()).filter(Boolean))];
    const failed = [];
    for (const id of ids) {
        try { await stopOne(id, action); } catch (exc) {
            if (exc?.status !== 404) failed.push(exc?.message || String(exc));
        }
    }
    if (failed.length) {
        toast(failed.length === ids.length ? `Stop failed: ${failed[0]}`
            : `Stop failed for ${failed.length} of ${ids.length} parts of this work: ${failed[0]}`, 'error');
    }
    return !failed.length;
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
 * closes the menu and invokes `onAction(actionId)`. `actions` (a work's controls,
 * work_status.js) replaces the card-state set; `pauseTarget` names Pause's task
 * when the anchor does not; `afterAction` runs once an item's effect settled.
 * @param {HTMLElement} anchor trigger element
 * @param {{cancelPending?: boolean, budgetPaused?: boolean, wholeTree?: boolean, resume?: boolean, busy?: boolean,
 *     actions?: string[], pauseTarget?: string, canAct?: (action: string) => boolean,
 *     onAction: (action: string) => void, afterAction?: (action: string) => void}} opts
 */
export function openTaskControlMenu(anchor, {
    cancelPending = false, budgetPaused = false, wholeTree = true, resume = false, busy = false, canAct = () => true, onAction,
    actions = null, pauseTarget = '', afterAction = null,
} = {}) {
    closeTaskControlMenu();
    if (!anchor?.isConnected || !document.body) return null;
    // A11y: the trigger owns a popup menu; expanded tracks the open state.
    anchor.setAttribute('aria-haspopup', 'menu');
    anchor.setAttribute('aria-expanded', 'true');
    const menu = document.createElement('div');
    menu.className = 'task-control-menu';
    menu.setAttribute('role', 'menu');
    for (const action of actions || taskControlActions({ cancelPending, budgetPaused, wholeTree, resume })) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `task-control-item${action === ACTION_STOP_NOW ? ' danger' : ''}`;
        item.dataset.taskControl = action;
        item.setAttribute('role', 'menuitem');
        item.textContent = TASK_CONTROL_LABELS[action];
        if (busy) item.disabled = true;
        item.addEventListener('click', (event) => {
            event.stopPropagation();
            // A portal can outlive the facts that offered it. Its owner rechecks before EVERY effect,
            // including Pause, which dispatches here rather than through onAction.
            if (!canAct(action)) { closeTaskControlMenu(); return; }
            if (action === ACTION_STOP_NOW || action === ACTION_STOP_EARLIER) haptic('stop');  // a decisive press, felt
            closeTaskControlMenu();
            // Both surfaces' anchors name their task (Activity row `data-id`,
            // chat card root `data-task-id`), so Pause needs no per-surface hook.
            const effect = action === ACTION_PAUSE
                ? pauseTaskAction(pauseTarget || anchor.dataset?.id || anchor.closest?.('[data-task-id]')?.dataset?.taskId || '')
                : onAction?.(action);
            if (afterAction) Promise.resolve(effect).then(() => afterAction(action), () => afterAction(action));
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

/**
 * The shared controls of one exact task that no work row speaks for (a card the work projection does not list):
 * the same request owners, never Resume (only a work's Resume control offers it). `stop` is the caller's own stop
 * path (a card keeps its optimistic chip); the Pause item reads its target from the anchor.
 */
export function openTaskControls(anchor, taskId, { activity = null, cancelPending = false, canAct, stop }) {
    const id = String(taskId || '').trim();
    return openTaskControlMenu(anchor, {
        cancelPending,
        budgetPaused: activity?.phase === 'budget_paused',
        busy: taskControlBusy(id),
        canAct,
        onAction: (action) => (action === ACTION_HURRY ? hurryTaskAction(id) : stop(action)),
    });
}

// The trigger's constant label. The trigger always opens the dropdown; the
// pending-cancel state changes the OFFERED actions (escalation only), not the
// interaction shape — so an accidental click never hard-stops directly.
export const TASK_CONTROL_TRIGGER_LABEL = 'Stop…';
