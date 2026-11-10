// What a Swarm message says and what its root did (docs/DESIGN.md "Composer owner controls"). The quiet
// `Swarm` in an owner message's time line is the room's Subagents value when the host accepted that message: a request,
// never proof that helpers ran. One line on a Swarm root's card says only what the root's receipt records (SYN2 A.8:
// helpers started, and requested where a request has no recorded start; starts that failed or are unconfirmed; models,
// the money known and what is not final yet, subscription sessions), never minds, waves, families, peaks, percentages
// or one sum the facts do not make; an unknown amount is a word, never $0. Only a root proven to have tried no helper
// has no line; not knowing is not that proof.
import { SUBAGENTS_MARK } from './composer_subagents.js';
import { formatUsd2 } from './utils.js';

const counted = (count, one) => `${count} ${one}${count === 1 ? '' : 's'}`;
const recorded = (value) => (typeof value === 'number' && Number.isFinite(value))
    || (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Date.parse(value)));

const receiptActors = (receipt) => (Array.isArray(receipt?.actor_intervals) ? receipt.actor_intervals : [])
    .filter((actor) => actor && typeof actor === 'object');

/* The receipt's actors (ouroboros/swarm_receipt_sources.py) folded into helpers: a child task, or a delegated run of the
 * root or another supervisor; a child's own runs and starts count with that child; never the root or another supervisor,
 * never a review. `started` has a recorded start; `open` has a start nobody can confirm or deny yet (a pending start, or a
 * run whose start is not recorded) and is no started helper; `children` are the child tasks the tree holds. */
function helperFacts(receipt, rootId) {
    const actors = receiptActors(receipt);
    const supervisors = new Set([rootId, ...actors.filter((actor) => actor.kind === 'supervisor').map((actor) => String(actor.task_id || ''))]);
    const started = new Set(), children = new Set(), open = new Set();
    for (const actor of actors) {
        const task = String(actor.task_id || '');
        const child = Boolean(task) && !supervisors.has(task);
        const helper = child ? `task:${task}` : String(actor.id || '');
        if (actor.kind === 'task' && child) {
            children.add(helper);
            if (recorded(actor.start)) started.add(helper);
        } else if (actor.kind === 'delegated_run' || actor.kind === 'delegated_start') {
            (actor.kind === 'delegated_run' && recorded(actor.start) ? started : open).add(helper);
        }
    }
    for (const helper of started) open.delete(helper);
    for (const set of [started, children, open]) set.delete('');
    return { started, children, open };
}

/* A receipt that knows every actor of its tree: complete coverage and its actors' list. A refused start has no actor,
 * so the zero proof below also needs the refusals the receipt read. */
const knowsTree = (receipt) => receipt?.coverage?.tree === 'complete' && Array.isArray(receipt?.actor_intervals);

/* The record's proof that this root tried no helper: the host observed no fan-out, and a receipt that knows its whole
 * tree holds only the root and its reviews, no refused start and no session or attempt of another actor. An observation
 * alone, or an older or partial receipt, is no proof. */
function triedNoHelper(receipt, efficiency, rootId) {
    if (efficiency?.observed_started !== 0 || !knowsTree(receipt)
        || !Array.isArray(receipt.refused_starts) || receipt.refused_starts.length) return false;
    const own = (row) => row && typeof row === 'object' && String(row.task_id || '') === rootId;
    return receipt.actor_intervals.every((actor) => own(actor) && ['supervisor', 'review', 'review_custody'].includes(actor.kind))
        && (Array.isArray(receipt.cost?.attempts) ? receipt.cost.attempts : []).every((attempt) => own(attempt)
            && (attempt.kind !== 'subscription_session' || attempt.actor_id === `task:${rootId}`
                || String(attempt.actor_id || '').startsWith('review:')));
}

/* The line's words about helpers, from recorded facts; [] only with the proof above. Started helpers, then the requests
 * with no recorded start (`none started` only where a receipt knows its whole tree), then every refused start and every
 * start not yet confirmed or denied. */
function helperWords(detail, efficiency) {
    const receipt = detail.swarm_receipt;
    const rootId = String(receipt?.root_task_id || detail.task_id || '');
    const { started, children, open } = helperFacts(receipt, rootId);
    const noted = efficiency?.subagent_count;
    const requested = Math.max(Number.isInteger(noted) && noted > 0 ? noted : 0, new Set([...children, ...started]).size);
    const failed = Array.isArray(receipt?.refused_starts) ? receipt.refused_starts.filter((row) => row && typeof row === 'object').length : 0;
    const words = started.size
        ? (requested > started.size ? [`${counted(requested, 'helper')} requested`, `${started.size} started`] : [`${counted(started.size, 'helper')} started`])
        : requested ? [`${counted(requested, 'helper')} requested`, ...(knowsTree(receipt) && !open.size ? ['none started'] : [])] : [];
    words.push(...(failed ? [`${counted(failed, 'helper start')} failed`] : []),
        ...(open.size ? [`${counted(open.size, 'helper start')} unconfirmed`] : []));
    if (words.length) return words;
    return triedNoHelper(receipt, efficiency, rootId) ? [] : ['helpers unknown'];
}

function ownBubble(feed, row) {
    const cmid = String(row?.client_message_id || '');
    return cmid ? [...(feed?.querySelectorAll('.chat-bubble.user[data-client-message-id]') || [])]
        .find((node) => node.dataset.clientMessageId === cmid) || null : null;
}

/** Puts `Swarm ·` before the time of an owner message the host accepted under Swarm. The bubble is the row's own, or
 *  (the owner's own echo) the one in `feed` that carries the row's client message id. */
export function markSwarmLabel(feed, row, bubble = ownBubble(feed, row)) {
    const time = row?.role === 'user' && row.subagents_mode === 'swarm' ? bubble?.querySelector('.msg-time') : null;
    if (!time || time.querySelector('.msg-swarm')) return false;
    const doc = time.ownerDocument;
    const label = doc.createElement('span');
    label.className = 'msg-swarm';
    label.title = 'Sent while Subagents was on Swarm';
    label.innerHTML = `<svg viewBox="0 0 184 184" aria-hidden="true">${SUBAGENTS_MARK}</svg>`;
    const word = doc.createElement('span');
    word.textContent = 'Swarm';
    label.append(word);
    const dot = doc.createElement('span');
    dot.className = 'msg-swarm-sep';
    dot.setAttribute('aria-hidden', 'true');
    dot.textContent = '·';
    time.prepend(label, dot);
    return true;
}

/** A Swarm root's line from its recorded facts: null while none are in this read, '' when they prove nothing to say. */
function swarmReceiptLine(detail) {
    const metadata = detail?.metadata;
    if (metadata?.subagents_mode !== 'swarm' || detail.parent_task_id || metadata.parent_task_id) return null;
    const efficiency = detail.swarm_efficiency && typeof detail.swarm_efficiency === 'object' ? detail.swarm_efficiency : null;
    const cost = detail.swarm_receipt?.cost && typeof detail.swarm_receipt.cost === 'object' ? detail.swarm_receipt.cost : null;
    if (!efficiency && !cost) return null;
    // A start is said only from a start fact: the receipt's actors. Without one, what is known is the requests
    // (`subagent_count` counts emitted children, refused ones too).
    const helpers = helperWords(detail, efficiency);
    if (!helpers.length) return '';
    const attempts = (Array.isArray(cost?.attempts) ? cost.attempts : []).filter((attempt) => attempt && typeof attempt === 'object');
    const models = new Set(attempts.map((attempt) => String(attempt.model || '')).filter(Boolean)).size;
    // A session is its task: a resumed session is one more attempt of the same session.
    const sessions = new Set(attempts.filter((attempt) => attempt.kind === 'subscription_session')
        .map((attempt) => String(attempt.task_id || attempt.attempt_id || ''))).size;
    const known = Number(cost?.known_final_usd) > 0 ? `${formatUsd2(cost.known_final_usd)} known` : '';
    const open = Math.trunc(Number(cost?.unknown_attempts) || 0);
    return ['Swarm', ...helpers, models && counted(models, 'model'),
        [known, open > 0 ? `${open} not final` : ''].filter(Boolean).join(' + '),
        sessions && counted(sessions, 'subscription session')].filter(Boolean).join(' · ');
}

/** The one line a Swarm root's card says from its recorded facts, or '' when it has none to say. */
export function swarmReceiptText(detail) {
    return swarmReceiptLine(detail) || '';
}

/** The receipt line, first in a root card's foot, once the root's detail carries its facts; an event or a row without
 *  them changes nothing, and facts that leave nothing to say take the line away. */
export function syncSwarmReceipt(record, detail) {
    const text = record?.root && record.footEl && !record.isSubagent ? swarmReceiptLine(detail) : null;
    if (text === null || (record.swarmReceiptEl?.textContent ?? '') === text) return false;
    if (!text) {
        record.swarmReceiptEl.remove();
        record.swarmReceiptEl = null;
        return true;
    }
    if (!record.swarmReceiptEl) {
        record.swarmReceiptEl = record.root.ownerDocument.createElement('div');
        record.swarmReceiptEl.className = 'chat-live-swarm-receipt';
        record.swarmReceiptEl.setAttribute('data-i18n-skip', '');
        record.footEl.prepend(record.swarmReceiptEl);
    }
    record.swarmReceiptEl.textContent = text;
    return true;
}
