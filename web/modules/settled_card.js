// The settled task card's one seam: its Files row (result_files.js), its Continue action (task_continue.js) and its
// Details' `Model handover` row (stored details only) follow the same settled task detail; the work's review-gate
// field (review_gate_card.js) follows every detail, live ones too. It also owns the reads that reach that detail (one
// per card: the census's read and a reveal join it) and how a child's own terminal source is presented to the subagent
// lifecycle (terminalChildFrame).
// A replayed history row settles its card from the terminal facts it froze (historicalTerminalFacts).

import { syncResultFilesItem } from './result_files.js';
import { syncReviewGateItem } from './review_gate_card.js';
import { cardMetaKeys, positiveTaskTerminalFact, renderLiveCardMeta } from './chat_activity.js';
import { childCauseFacts, isTerminalTaskDetail, taskCancelPending, taskDoneIsTerminal, taskOutcomeSeverity } from './log_events.js';
import { READ_EPOCH, syncContinueAction } from './task_continue.js';
import { measure } from './layout_batch.js';
import { fetchTaskDetailStrict, withinRequestLifetime } from './api_client.js';

function handoverAttempt(detail) {
    const stamp = detail?.task_attempt ?? detail?._attempt;
    return Number.isSafeInteger(stamp) && stamp > 0 ? stamp : null;
}

/** An applied stored terminal read, not a partial terminal hint or a GET of a previous attempt. */
function hasSettledCardDetail(record) {
    return record.settledDetailAttempt === (record.handoverAttempt ?? 1);
}

const detailsOpen = record => record.episodeDisclosure?.open ?? (record.root.dataset.expanded === '1');

/** Only settled stored detail says what handover history is available. A newer attempt retires it; a late GET or
 *  cost-only frame cannot. A running snapshot of the same attempt is not evidence against a saved result. */
function syncModelHandover(record, detail, stored) {
    if (!record || (detail?.type || detail?.event) === 'task_cost_finalized') return false;
    const before = record.handoverRows;
    const attempt = handoverAttempt(detail);
    // An unnumbered legacy result cannot speak for a known second or later attempt.
    const known = record.handoverAttempt ?? 1;
    const late = (attempt ?? 1) < known;
    if (attempt !== null && attempt > known) record.handoverRows = undefined;
    if (attempt !== null && !late) record.handoverAttempt = attempt;
    if (stored && !late && isTerminalTaskDetail(detail)) {
        record.settledDetailAttempt = attempt ?? 1;
        // An omitted field adds no fact for this attempt; only a supplied array can replace the rows.
        const rows = detail.authoring_handovers;
        if (Array.isArray(rows) && JSON.stringify(rows) !== JSON.stringify(record.handoverRows)) {
            record.handoverRows = rows.length ? rows : undefined;
        }
    }
    return record.handoverRows !== before;
}

/** A durable terminal publication can follow an early answer that already closed the card. It is a read hint,
 *  not the history itself. Join the existing detail hydrator; a prior read may have seen the running record. */
function needsSettledCardDetail(record, detail) {
    if (!record || record.root.dataset.portal === '1' || detail?.task_phase === 'finalizing'
            || detail?.outcome_final === false) return false;
    const summary = detail?.system_type === 'task_summary';
    const archived = summary && detail.historical_terminal;
    const terminal = (detail?.type || detail?.event) === 'task_done'
        || ((summary || detail?.history_id) && detail.outcome_final === true) || archived;
    if (!terminal || !taskDoneIsTerminal(archived || detail)) return false;
    const known = record.handoverAttempt ?? 1;
    return !hasSettledCardDetail(record) && (handoverAttempt(detail) ?? known) >= known
        && detailsOpen(record);
}

/** The card's one GET of its task detail for its current attempt, numbered as it is sent (`record.readEpoch`) so an
 *  answer can be told from a later one. A read for the same attempt that is in flight is joined, whoever asks; `after`
 *  joins only one sent after that number. The GET is bounded by its own lifetime, not by a joiner's: a reader that gives
 *  up leaves it for the next. A read is no snapshot of its sending: the answer names the attempt the store holds now.
 *  So a read for a retired attempt still speaks when its answer names the attempt the card has now — that is the
 *  newer attempt's own evidence, and no second read is owed for it. A read sent before one that already answered, an
 *  answer naming a retired attempt (or none), and a retired read's failure resolve `undefined`: they speak for no
 *  one, so a late answer or error never replaces a newer result, offers what a retired attempt claimed, nor shows
 *  that attempt's Retry. It owes no read of its own: the frame that named the newer attempt asks again when it
 *  still needs the detail. */
function cardDetailGet(record, after = 0, signal = null) {
    const attempt = record.handoverAttempt ?? 1, shared = record.detailGet;
    if (shared?.attempt === attempt && shared.epoch > after) { shared.peers = true; return { get: shared, joined: true }; }
    const get = { attempt, epoch: record.readEpoch = (record.readEpoch ?? 0) + 1 };
    // The initiator's `signal` ends the GET only while no one else has joined it; a joiner's never does.
    const own = signal && new AbortController(), cancel = () => { if (!get.peers) own.abort(); };
    if (signal?.aborted) own.abort(); else signal?.addEventListener('abort', cancel, { once: true });
    const answered = () => (record.answeredEpoch ?? 0) > get.epoch;
    const retired = () => (record.handoverAttempt ?? 1) !== get.attempt;
    const speaksNow = detail => !retired() || (handoverAttempt(detail) ?? 1) === (record.handoverAttempt ?? 1);
    get.done = withinRequestLifetime(transport => fetchTaskDetailStrict(record.groupId, { signal: transport }), { signal: own?.signal })
        .then(detail => {
            if (answered() || !speaksNow(detail)) return undefined;
            record.answeredEpoch = get.epoch;
            if (detail && typeof detail === 'object') detail[READ_EPOCH] = get.epoch;
            return detail;
        }, error => { if (answered() || retired()) return undefined; throw error; })
        .finally(() => { signal?.removeEventListener('abort', cancel); if (record.detailGet === get) delete record.detailGet; });
    record.detailGet = get;
    return { get, joined: false };
}

/** The card's task detail (null: absent; undefined: superseded; rejects: unread). A joined read began before this
 *  request: when its answer is not `satisfied` (a running snapshot, an offer not yet settled) one authoritative read
 *  follows, and no more. `fresh`: the answer must come from a read sent now (a press just changed the store). */
export async function readCardDetail(record, { satisfied = null, fresh = false, signal = null } = {}) {
    const { get, joined } = cardDetailGet(record, fresh ? record.readEpoch ?? 0 : 0, signal);
    const detail = await get.done;
    return !joined || !detail || !satisfied || satisfied(detail) ? detail : cardDetailGet(record, get.epoch, signal).get.done;
}

/** The review hydrator's reader: a card's detail is its one read; a review revision this page has not seen needs a
 *  newer one; a task with no card is read plain. */
export const cardDetailReader = (records, plain) => (id, { signal, revision } = {}) => records.has(id)
    ? readCardDetail(records.get(id), { fresh: Boolean(revision), signal }) : plain(id, { signal });

/** One owner of the join decision: when the census's read of this card can still apply (`current()`),
 *  hydrate through it instead of racing a second concurrent GET for the same attempt — both reads apply
 *  through the same stored-detail seam. Only an answer that left `needed` unsatisfied (a running snapshot,
 *  a miss, a failure — the read's own catch settles it) earns one top-up read; the hydrator owns its
 *  error surface and Retry after that. */
function hydrateJoiningCensusRead(record, hydrate, needed) {
    const read = record.managedDetail;
    if (!read?.current()) { hydrate(record.groupId, null, needed); return; }
    void read.done.then(() => {
        const stillNeeded = () => read.current() && needed();
        if (stillNeeded()) hydrate(record.groupId, null, stillNeeded);
    });
}

export function refreshSettledCardDetail(record, detail, hydrate) {
    const needed = () => needsSettledCardDetail(record, detail);
    if (needed()) hydrateJoiningCensusRead(record, hydrate, needed);
}

// The fact one read can answer for this card: its settled terminal detail, or a cancellation the host proved
// (`cancel_state: pending`). Census-first shows the latter as Cancelling from the read it already made; a joined
// Details read must not spend a second GET on the same fact, and dropping it would leave the card silent on a stop.
const settledOrCancelling = detail => isTerminalTaskDetail(detail) || taskCancelPending(detail);

// The census may already be reading this card's detail when the terminal frame arrives. Wait for its
// application, not just the GET: only the stored terminal fact for this attempt satisfies the disclosure.
// `report('error' | 'idle')` hands the read's outcome to the Details' one status owner (the hydrator): an open card's
// failed read shows its ordinary Retry, and an applied read retires a failure an earlier read left.
export function readMissingCardDetail(record, shown, apply, report = () => {}) {
    if (record.managedDetail?.current()) return;
    const attempt = record.handoverAttempt ?? 1;
    const current = () => shown() && (record.handoverAttempt ?? 1) === attempt;
    const read = { current, done: readCardDetail(record, { satisfied: settledOrCancelling })
        .then(detail => {
            if (current() && detail !== undefined && (detail === null || (handoverAttempt(detail) ?? 1) >= attempt)) { apply(detail); report('idle'); }
        }).catch(() => {  // Failure is not absence; a later snapshot or terminal hint can read again.
            if (current() && detailsOpen(record)) report('error');
        })
        .finally(() => { if (record.managedDetail === read) delete record.managedDetail; }) };
    record.managedDetail = read;
}

// Any opening can join a GET sent while the task was running — the hydrator's own in-flight read, and
// an in-flight census read of the same attempt too: racing that with a second concurrent GET would spend
// a request on the very detail it already applies here. Follow either once only if the reader still needs
// terminal detail for that attempt. A repeat landing in current Details needs no GET. A census read that
// can no longer apply (its card left, its attempt superseded) is not joined — the opening's own GET is
// then the only effective read, not a duplicate.
export function hydrateRevealedCard(record, wasOpen, hydrate, ready) {
    if (wasOpen && record.finished && hasSettledCardDetail(record) && ready(record.groupId)) return;
    const attempt = record.handoverAttempt ?? 1;
    hydrateJoiningCensusRead(record, hydrate, () => (record.handoverAttempt ?? 1) === attempt
        && record.finished && detailsOpen(record) && !hasSettledCardDetail(record));
}

// CSS owns child visibility, including running descendants. Read it with this mutation's other layout reads.
export function hydrateOpenSettledChild(record, hydrate) {
    if (!record?.finished || record.root.dataset.expanded !== '1') return;
    const needed = () => record.root.isConnected && record.root.dataset.expanded === '1'
        && !hasSettledCardDetail(record) && record.root.getClientRects().length > 0;
    measure(needed, visible => { if (visible) hydrate(record.groupId, null, needed); });
}

export function hydrateRevealedChildren(record, records, hydrate) {
    const host = record.episodeDisclosure || record.root;
    for (const child of host.querySelectorAll('.chat-live-card.subagent[data-expanded="1"][data-finished="1"]')) {
        hydrateOpenSettledChild(records.get(child.dataset.taskId), hydrate);
    }
}

// Present a child's own terminal source through the existing subagent lifecycle path. The frame carries the
// cause facts the source actually stated and only the text it actually carries: a fact or a word the source did
// not give is never invented here, and each truncation flag travels beside the text it truncates.
export function terminalChildFrame(childId, info, evt) {
    const status = String(evt.status || '').toLowerCase();
    const severity = taskOutcomeSeverity(evt);
    const event = status === 'interrupted' ? 'interrupted'
        : severity === 'error' || status === 'failed' ? 'failed'
        : status === 'cancelled' || status === 'cancel_requested' ? 'cancelled'
        : status === 'rejected_duplicate' ? 'rejected' : severity === 'warn' ? 'completed_warn' : 'completed';
    return { ...childCauseFacts(evt),
        delegation_role: 'subagent', parent_task_id: info.parentId, subagent_task_id: childId,
        subagent_role: info.role, subagent_event: event, review_projection: evt.review_projection,
        model_execution: evt.model_execution,
        result_truncated: evt.result_truncated, trace_summary_truncated: evt.trace_summary_truncated,
        // Only the text this frame carries.
        ...Object.fromEntries(['result', 'trace_summary', 'error'].filter((key) => evt[key]).map((key) => [key, evt[key]])),
        reason_code: evt.reason_code || '',
        ...cardMetaKeys(evt) };
}

/** Paint handover evidence through the caller's viewport seam. A census row can only advance the attempt;
 *  a stored read also supplies that attempt's terminal history. */
export function noteHandoverAttempt(record, detail, write, stored = false) {
    if (syncModelHandover(record, detail, stored)) write(() => renderLiveCardMeta(record));
}

/** The finished root's closing line, shared by its card, room row and portal. */
export function closingLine(record, title = record?.titleEl?.textContent || record?.suggestedName || '') {
    if (!record?.finished) return '';
    if (record.causeHeld) return record.collapsedActivity || '';
    const own = !record.isSubagent && record.wakeSummary || '';
    return own !== title ? own : '';
}

/** @param {{write?: Function, stored?: boolean}=} options `stored`: the detail is the task's stored record (the hydrator's read)
 *  @returns {boolean} whether the card's timeline items (the Files row) or its handover row changed */
export function syncSettledItems(record, detail, options) {
    // A late offer read joins the card's one detail read, or is not needed once its settled detail is in; it writes
    // through the caller's viewport seam.
    syncContinueAction(record, detail, { ...options, read: !record || hasSettledCardDetail(record) ? null
        : () => readCardDetail(record, { satisfied: isTerminalTaskDetail }) });
    const gate = syncReviewGateItem(record, detail);
    const handover = syncModelHandover(record, detail, options?.stored);
    return syncResultFilesItem(record, detail) || gate || handover;
}

/** The task's own terminal detail or a new attempt replaces its sentence; history stamps only settle status. */
export function syncClosingSentence(record, detail) {
    const attempt = detail?.task_attempt;
    const stamped = Object.hasOwn(detail || {}, 'task_terminal_status')
        && detail?.system_type !== 'task_summary' && detail?.type !== 'task_done';
    const reset = (!stamped && (isTerminalTaskDetail(detail) || positiveTaskTerminalFact(detail)))
        || (Number.isSafeInteger(attempt) && attempt !== record?.summaryAttempt);
    const sentence = detail && 'wake_summary' in detail ? detail.wake_summary : reset ? '' : record?.wakeSummary;
    const words = record && record.wakeSummary !== sentence;
    if (record && Number.isSafeInteger(attempt)) record.summaryAttempt = attempt;
    if (words) {
        if (record.collapsedActivity === record.wakeSummary) record.collapsedActivity = '';
        record.wakeSummary = sentence;
    }
    return Boolean(words);
}

/** The terminal facts a replayed history row froze for its card (chat.js `historicalTerminals`, settled through
 *  `noteTaskDetails`): its durable terminal, the current outcome's phase, and the sentence, name, schedule and place
 *  the settled card reads. */
export function historicalTerminalFacts(row) {
    return { ...row.historical_terminal, ...(row.current_outcome && { phase: row.current_outcome.outcome_phase }),
        wake_summary: row.wake_summary, suggested_name: row.suggested_name, schedule_name: row.schedule_name,
        place_origin: row.place_origin };
}
