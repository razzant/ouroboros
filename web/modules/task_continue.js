// Owner Batch4: "Continue" on a card whose root task was interrupted.
//
// The server decides what a settled card may offer (`continuation_offer` on the
// task detail: a Continue after a technical interruption, or the successor this
// task already has). The press carries ONE random action nonce kept in local
// storage per task, so a retry or a reload after a lost answer returns the SAME
// admission instead of starting a second task; a timeout is never read as "not
// continued". The card then points to its successor (a stale card never offers
// a second Continue — new work goes through the conversation).

import { continueTask, fetchTaskDetail } from './api_client.js';
import { announceMembershipGap, heldWorkFacts } from './task_control_menu.js';
import { ensureLiveActionsEl, ownLiveActionsEl } from './chat_activity.js';
import { fmt, tr, tx } from './i18n.js';
import { taskDoneIsTerminal, taskReasonPhrase } from './log_events.js';
import { showToast } from './toast.js';
import { setInlineStatus } from './ui_primitives.js';

const NONCE_KEY = 'ouro_continue_nonce:';
const inFlight = new Set();
const sessionNonces = new Map();
// Cards whose full task detail was read once for the server's offer.
const offerReads = new WeakSet();

/** One Continue action per task. Unavailable storage limits retention to this page. */
export function continueNonce(taskId, storage, retainedNonce = '') {
    const key = `${NONCE_KEY}${taskId}`;
    let nonce = retainedNonce || sessionNonces.get(key);
    try {
        storage ??= globalThis.localStorage;
        nonce ||= storage?.getItem?.(key);
    } catch { /* even accessing localStorage can throw */ }
    nonce ||= (globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`)
        .replace(/[^A-Za-z0-9_-]/g, '');
    sessionNonces.set(key, nonce);
    try { storage?.setItem?.(key, nonce); } catch { /* best effort */ }
    return nonce;
}

/** @returns {{kind: 'offer'|'retry'|'successor'|'refusal'|'none', successorId?: string, cause?: string, actionNonce?: string}} */
export function continueOfferView(detail) {
    const offer = detail?.continuation_offer;
    if (!offer || typeof offer !== 'object') return { kind: 'none' };
    if (offer.state === 'bound') return offer.action_nonce
        ? { kind: 'retry', actionNonce: String(offer.action_nonce) } : { kind: 'none' };
    if (offer.successor_task_id) return { kind: 'successor', successorId: String(offer.successor_task_id) };
    if (offer.eligible === false && offer.refusal === 'hard_limit_reached') {
        return { kind: 'refusal', cause: String(offer.cause || offer.refusal) };
    }
    return offer.eligible ? { kind: 'offer', cause: String(offer.cause || '') } : { kind: 'none' };
}

/**
 * Press Continue: one request under the persisted nonce; the answer names the
 * successor (a replay of an earlier press answers the same one).
 * @returns {Promise<string>} the successor task id, or '' when refused
 */
export async function continueTaskAction(taskId, { request = continueTask, storage, toast = showToast, actionNonce = '' } = {}) {
    const id = String(taskId || '').trim();
    if (!id || inFlight.has(id)) return '';
    inFlight.add(id);
    try {
        const ack = await request(id, continueNonce(id, storage, actionNonce));
        const successor = ack?.ok === true && typeof ack.successor_task_id === 'string'
            ? ack.successor_task_id.trim() : '';
        // A resolved body is not an acknowledgement (fetchJson resolves a
        // malformed 200 as {error}): without the server's ok + successor the
        // admission is unconfirmed, and the same nonce stays for the retry.
        if (!successor) {
            const reason = typeof ack?.error === 'string' && ack.error ? ack.error : 'no acknowledgement';
            throw Object.assign(new Error(reason), { body: {} });
        }
        // A held successor names what the census observed (#1721), never a generic wait. It starts by itself once
        // that work settles: nothing for the owner to resume; Stop earlier attempt ends it sooner.
        toast(ack?.held
            ? `Continue accepted; it starts by itself when the earlier attempt's work finishes: ${
                heldWorkFacts(ack.blockers) || 'earlier work is not proven settled'}.`
            : 'Continue accepted.', 'ok');
        announceMembershipGap(ack.membership_gaps, 'continue', toast);  // a replay answers its admission's gap too
        return successor;
    } catch (exc) {
        const body = exc?.body || {};
        if (body.reason_code === 'already_continued' && body.successor_task_id && body.state !== 'bound') {
            toast('Already continued.', 'info');
            return String(body.successor_task_id);
        }
        if (body.state === 'bound' && body.action_nonce) continueNonce(id, storage, body.action_nonce);
        // The same nonce stays: pressing again retries this very admission.
        toast(`Continue not confirmed: ${exc?.message || exc}`, 'error');
        return '';
    } finally {
        inFlight.delete(id);
    }
}

function renderSuccessor(button, successorId) {
    button.textContent = tr('task.continue.continued', 'Continued');
    button.disabled = true;
    button.dataset.continueSuccessor = successorId;
}

/** The successor a card's Continue control names: '' while it still offers Continue, null with no control. */
export function continueControlSuccessor(record) {
    const button = ownLiveActionsEl(record)?.querySelector('[data-continue-task]');
    return button ? String(button.dataset.continueSuccessor || '') : null;
}

/**
 * Keep one settled card's Continue action in step with the server's offer.
 *
 * The full task detail (`GET /api/tasks/{id}`) and a settled task's replayed
 * history rows state the offer, so a card rebuilt after a reload shows it
 * without being opened. A live event row (task_done, task_eval, …) says nothing
 * about it and never removes a shown action; when it is the terminal of a ROOT
 * card that did not complete cleanly (a best-effort completion may be a
 * technical limit), the card reads the full detail once to learn the offer.
 */
export function syncContinueAction(record, detail, { read = fetchTaskDetail, write = (fn) => fn() } = {}) {
    if (!detail || typeof detail !== 'object') return false;
    if (!('continuation_offer' in detail)) {
        const status = String(detail.task_terminal_status || detail.status || '').toLowerCase();
        const clean = ['completed', 'done'].includes(status) && detail.outcome_axes?.execution?.status !== 'best_effort';
        if (record?.root && !record.isSubagent && !offerReads.has(record) && taskDoneIsTerminal(detail) && !clean) {
            offerReads.add(record);
            Promise.resolve().then(() => read(record.groupId))
                .then((full) => {
                    const offer = full?.continuation_offer;
                    if (!offer || typeof offer !== 'object' || Array.isArray(offer)) {
                        offerReads.delete(record);  // null is the production reader's failure result
                    } else if (record.root?.isConnected) write(() => syncContinueAction(record, full, { read, write }));
                })
                .catch(() => { offerReads.delete(record); });  // a later terminal row retries
        }
        return false;
    }
    // Continue belongs to a root: the host refuses a child (not_a_root_task), so a child's card shows none, even
    // when a parent's replayed row (a subagent's rejection notice) brings the parent's offer onto it.
    const view = record?.isSubagent ? { kind: 'none' } : continueOfferView(detail);
    const actions = view.kind === 'none' ? ownLiveActionsEl(record) : ensureLiveActionsEl(record);
    const existing = actions?.querySelector('[data-continue-task]');
    const refusal = actions?.querySelector('[data-continue-refusal]');
    if (!actions || view.kind === 'none') {
        existing?.remove();
        refusal?.remove();
        return false;
    }
    if (view.kind === 'refusal') {
        existing?.remove();
        const note = refusal || document.createElement('span');
        if (!refusal) {
            note.className = 'ui-status';
            note.dataset.continueRefusal = String(record.groupId || '');
            actions.appendChild(note);
        }
        setInlineStatus(note, `${tr('task.continue.unavailable', 'Continue unavailable.')} ${taskReasonPhrase(view.cause)}`, 'neutral');
        return !refusal;
    }
    refusal?.remove();
    const button = existing || document.createElement('button');
    if (!existing) {
        button.type = 'button';
        // Continue is the one primary a settled card may offer; Continued (disabled) reads as a record.
        button.className = 'btn btn-sm btn-primary';
        button.dataset.continueTask = String(record.groupId || '');
        actions.appendChild(button);
    }
    if (view.kind === 'successor') {
        renderSuccessor(button, view.successorId);
        return !existing;
    }
    button.textContent = view.kind === 'retry' ? tr('task.continue.retry', 'Retry Continue') : tr('task.continue.label', 'Continue');
    button.disabled = false;
    button.title = view.kind === 'retry' ? tr('task.continue.retry_title', 'Retry the same unconfirmed Continue action')
        : fmt('Start a new task that continues this interrupted one ({cause})', { cause: tx(view.cause || 'technical interruption') });
    button.onclick = async (event) => {
        event.stopPropagation();
        button.disabled = true;
        const successor = await continueTaskAction(record.groupId, { actionNonce: view.actionNonce || '' });
        if (!successor) { button.disabled = false; return; }
        renderSuccessor(button, successor);
        // The card's own chat reads the predecessor's record now, so the successor joins this card (work_portal.js).
        record.root?.dispatchEvent?.(new CustomEvent('ouro:task-continued', { bubbles: true,
            detail: { taskId: String(record.groupId || ''), successorId: successor } }));
    };
    return !existing;
}
