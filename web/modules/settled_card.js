// The settled task card's one seam: its Files row (result_files.js), its Continue action (task_continue.js) and its
// Details' `Model handover` row (stored details only) follow the same settled task detail; the work's review-gate
// field (review_gate_card.js) follows every detail, live ones too. It also owns the reads that reach that detail (one
// per card: the census's read and a reveal join it) and how a child's own terminal source is presented to the subagent
// lifecycle (terminalChildFrame).
// A replayed history row settles its card from the terminal facts it froze (historicalTerminalFacts).

import { syncResultFilesItem } from './result_files.js';
import { syncReviewGateItem } from './review_gate_card.js';
import { cardMetaKeys, isReplayEvidenceRow, noteToolHostMetrics, positiveTaskTerminalFact, renderLiveCardMeta, toolEvidenceIncomplete, toolEvidenceView } from './chat_activity.js';
import { referencesTaskOnly } from './chat_row_placement.js';
import { childCauseFacts, isTerminalTaskDetail, taskCancelPending, taskDoneIsTerminal, taskTerminalPhase } from './log_events.js';
import { isoMicros } from './utils.js';
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
export function terminalChildFrame(childId, info, evt, phase = taskTerminalPhase(evt)) {
    const status = String(evt.status || '').toLowerCase();
    const event = status === 'interrupted' ? 'interrupted'
        : phase === 'error' ? 'failed'
            : phase === 'cancelled' ? 'cancelled'
                : status === 'rejected_duplicate' ? 'rejected'
                    : phase === 'warn' ? 'completed_warn' : 'completed';
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

/** Absence is meaningful on the task's own ending, not on a status stamp or late review. */
export function ownsClosingSentence(detail) {
    if (!detail || detail.outcome_final === false || ['finalizing', 'unfinished'].includes(detail.task_phase)) return false;
    const type = detail.type || detail.event;
    return detail.system_type === 'task_summary' || (type === 'task_done' ? taskDoneIsTerminal(detail)
        : !type && !detail.system_type && !Object.hasOwn(detail, 'task_terminal_status') && isTerminalTaskDetail(detail));
}

export function syncClosingSentence(record, detail) {
    const attempt = attemptOf(detail);
    if (record?.cardAttempt != null && attempt !== record.cardAttempt) return false;
    const reset = ownsClosingSentence(detail)
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
    const frozen = row.historical_terminal;
    const current = row.current_outcome;
    return { ...frozen, phase: (attemptOf(frozen) === attemptOf(current) && current?.outcome_phase) || frozen.phase,
        ...((attemptOf(row) === attemptOf(frozen)) && Object.fromEntries(
            ['wake_summary', 'suggested_name', 'schedule_name', 'place_origin']
                .filter(key => row[key] != null).map(key => [key, row[key]]))) };
}

export function attemptOf(row) {
    const witness = row?.terminal_time?.attempt;
    // The normal supervisor task_done frame names its physical owner instead of flattening its ordinal.
    const owner = row?.execution_owner;
    const ownerAttempt = row?.task_id && owner?.task_id === row.task_id ? owner.task_attempt : undefined;
    return [row?.task_attempt, row?._attempt, witness?.task_attempt, witness?._attempt, ownerAttempt]
        .find(attempt => Number.isSafeInteger(attempt) && attempt >= 0);
}

// A numbered ending/metric owns only its presentation. The task's anchor, money, history and review receipts
// survive a new physical attempt of the same task. An unnumbered carrier cannot assert a metric for it.
function selectCardAttempt(record, attempt) {
    if (attempt == null || (record.cardAttempt != null && attempt <= record.cardAttempt)) return;
    if (record.causeHeld || record.collapsedActivity === record.wakeSummary) record.collapsedActivity = '';
    // Keep observed invocations, not the previous attempt's scalar estimates, as the running count.
    if (record.toolFold) record.toolFold.host = null;
    const journal = record.toolFold && toolEvidenceView(record.toolFold);
    Object.assign(record, { cardAttempt: attempt, attemptMetrics: {}, rounds: null, roundsRecorded: false, durationSec: null,
        toolCalls: record.portalActivity?.tool_calls ?? (journal?.calls || null), toolErrors: journal?.calls ? journal.errors : null,
        quietEffective: false, causeHeld: false, wakeSummary: '', collapsedActivityLead: '' });
}

export function applyCardEnding(record, stamp) {
    selectCardAttempt(record, stamp.attempt);
    record.replayedEnding = stamp;
}

const knownCount = value => Number.isInteger(value) && value >= 0 ? value : null;
export function cardMetricFacts(row) {
    const { tool_evidence: evidence } = row;
    const duration = row.duration_sec != null ? Number(row.duration_sec) : NaN;
    const rounds = knownCount(row.rounds);
    const [calls, errors, routing, completion] = ['tool_calls', 'tool_errors', 'routing_tool_calls', 'completion_tool_calls']
        .map(key => knownCount(row[key]));
    const journal = Boolean(evidence?.observations?.length || evidence?.legacy?.calls || toolEvidenceIncomplete(evidence?.coverage));
    if (journal || rounds !== null || calls !== null || errors > 0 || Number.isFinite(duration)) {
        return { evidence, journal, duration, rounds, calls, errors, routing, completion, counts: row.tool_call_counts };
    }
    return null;
}

/** Fold an addressed scalar only into its attempt. Invocation evidence is the task's whole journal. */
export function foldCardMetrics(record, row, facts, selectedAttempt, recorded = false) {
    const attempt = attemptOf(row);
    // The gateway marks every row of a task whose stored result is still open `unfinished`, so the phase
    // alone is not a frozen ending. Only a summary row is one: the recorded ending of an attempt that is
    // not the current execution. It never selects the card's attempt; of another attempt it owns no
    // counters at all, while a summary of the card's own attempt still states that attempt's count. Its
    // invocation evidence still folds below — the journal is the task's, not one attempt's. The open
    // attempt's own live carriers (progress, evidence, narration) keep folding exactly as before.
    const frozenEnding = row.task_phase === 'unfinished' && row.system_type === 'task_summary';
    selectCardAttempt(record, selectedAttempt ?? (frozenEnding ? null : attempt));
    const ownAttempt = attempt === record.cardAttempt;
    const owned = !frozenEnding && ownAttempt;
    // A `task_summary` row and the host's selected ending (`recorded`) state what an attempt recorded — a
    // measured zero included; any other carrier that states a count is the worker's own end-of-attempt
    // figure. Either may arrive first: the recorded count replaces the fold, and an estimate never raises
    // a count its attempt has already recorded. A frozen ending of the card's own attempt still states
    // that attempt's count; of another attempt it states nothing here.
    const stated = ownAttempt && (recorded || row.system_type === 'task_summary');
    if (!owned && !stated && !facts.journal) return null;
    const { rounds, duration, calls, errors } = facts;
    if (rounds !== null) {
        if (stated) { record.rounds = rounds; record.roundsRecorded = true; }
        else if (owned && !record.roundsRecorded) record.rounds = Math.max(record.rounds ?? 0, rounds);
    }
    if (owned) {
        if (Number.isFinite(duration)) record.durationSec = duration;
        const counts = record.attemptMetrics || record;
        if (calls !== null) counts.toolCalls = Math.max(counts.toolCalls ?? 0, calls);
        if (errors !== null) counts.toolErrors = errors;
    }
    const summary = facts.journal || (owned && (calls > 0 || errors > 0))
        ? noteToolHostMetrics(record, owned ? facts : { evidence: facts.evidence }) : null;
    if (summary) {
        if (summary.calls > 0 || record.toolCalls != null) record.toolCalls = summary.calls;
        record.toolErrors = summary.errors;
    }
    return { owned, summary };
}

/** A known attempt comes first. Two lifecycles within it are two endings, and the later statement is the task's
 * last word, whatever the earlier one knew. Of one lifecycle a revised outcome, a published ending, recorded detail and
 * a bare stamp carry progressively less authority. Time orders equivalent sources. */
function olderEnding(next, known) {
    if (!known) return false;
    if (next.attempt !== known.attempt) return next.attempt == null || next.attempt < known.attempt;
    const earlier = Number.isFinite(known.time) && !(next.time >= known.time);
    if (next.lifecycle && known.lifecycle && next.lifecycle !== known.lifecycle) return earlier;
    return next.rank === known.rank ? earlier : next.rank < known.rank;
}

// The lifecycle a terminal statement claims; an older task_done spelling reads as its settled status.
function lifecycleOf(detail) {
    const status = String(detail?.task_terminal_status || detail?.status || detail?.outcome_axes?.lifecycle?.status || '')
        .toLowerCase();
    return { done: 'completed', cancel_requested: 'cancelled' }[status] || status;
}

function hasOutcomeDetails(detail) {
    return detail && (detail.reason_code || detail.review_projection || detail.review_status?.status || detail.cancel_origin || detail.artifact_status
        || detail.artifact_bundle?.status || detail.task_completion || detail.finalization_recovery || detail.terminal_delivery
        || Object.entries(detail.outcome_axes || {}).some(([key, axis]) =>
            key !== 'lifecycle' && axis && typeof axis === 'object' && Object.keys(axis).length));
}

function endingStamp(detail, current, frozen, matchesPhase = true) {
    const facts = hasOutcomeDetails(detail);
    const published = frozen || (detail.outcome_final === true && detail.outcome_phase) || (facts && detail.type === 'task_done');
    const rank = current && (hasOutcomeDetails(current) || current.outcome_phase) ? 5
        : published ? (facts && matchesPhase ? 4 : 3) : facts ? 2 : detail.outcome_phase ? 1 : 0;
    return { rank, attempt: attemptOf(detail), lifecycle: lifecycleOf(detail),
        time: isoMicros(current?.revision || detail.ts || detail.timestamp) };
}

/** Whether a history row draws its task's completion on the card: a settled summary, or the progress or evidence
 *  carrier on which the host stamped the outcome of a summary outside the page. An answer bubble and a receipt settle
 *  the phase and draw no completion row (DESIGN "Conversation activity block": the completion note is not content). */
function drawsCompletion(row) {
    if (['finalizing', 'unfinished'].includes(row.task_phase) || row.outcome_final === false) return false;
    if (row.system_type === 'task_summary') return !isReplayEvidenceRow(row);
    return Boolean((row.is_progress || row.system_type === 'task_evidence') && row.task_terminal_status && hasOutcomeDetails(row));
}

/** Select a live start/ending before any presentation fields or counters are applied. */
export function acceptLiveFrame(row, replayActive, getCard, noteWords) {
    if (!replayActive && !row._historical && (row.type || row.event) === 'task_started' && attemptOf(row) != null) {
        const card = getCard(), attempt = attemptOf(row), ended = card.replayedEnding?.attempt;
        if (card.isSubagent) return true;
        if (attempt < card.cardAttempt || attempt === ended) return false;
        const advances = ended == null ? card.cardAttempt == null || attempt > card.cardAttempt : attempt > ended;
        if (card.finished && !advances) return false;  // a live numbered start may reopen an unnumbered ending
        selectCardAttempt(card, attempt);
        if (card.finished) {
            Object.assign(card, { finished: false, finalizingHold: false, observedOutcome: '', historicalTerminal: null });
            card.root.dataset.finished = '0';
        }
        return true;
    }
    const late = row?.current_outcome;
    if (replayActive || row.outcome_final === false || ['finalizing', 'unfinished'].includes(row.task_phase)
            || !(row.system_type === 'task_summary' || row.outcome_final === true || taskDoneIsTerminal(late || row))) return true;
    const card = getCard();
    const stamp = endingStamp(late ? { ...row, ...late } : row, late, false);
    // The caller already checked a late result's revision against the displayed outcome. A legacy review can
    // revise that outcome, but it cannot make its unnumbered time, words or counters into attempt evidence.
    if (late) stamp.attempt = attemptOf(late) ?? card.cardAttempt;
    if (card.cardAttempt != null && (stamp.attempt == null || stamp.attempt < card.cardAttempt)) return false;
    const known = card.replayedEnding;
    if (olderEnding(stamp, known)) {
        // A sparse own terminal notice at the same occurrence can clear its sentence, not replace the richer ending.
        if (!late && !row._historical && !row.wake_summary && ownsClosingSentence(row) && stamp.lifecycle
                && stamp.attempt === known.attempt && stamp.lifecycle === known.lifecycle && stamp.time === known.time) {
            noteWords?.(card, { task_attempt: stamp.attempt, wake_summary: '' });
        }
        return false;
    }
    applyCardEnding(card, stamp);
    return true;
}

/** The card keeps the selected attempt, source and time across page reads. The page owns the row contents: an ending
 *  settles only a card other rows made, with its facts (`rich`) and, where one is recorded, its completion row (`drawn`). */
export function replayedTerminalRecords(messages, cards, historicalTerminals) {
    const endings = new Map();
    const rows = messages.filter(row => row.task_id && row.role !== 'user' && !referencesTaskOnly(row)
        && (!row.system_type || row.is_progress || positiveTaskTerminalFact(row) || row.system_type === 'task_evidence'));
    const completions = new Map();  // task id → the attempts whose completion row this page holds
    for (const row of rows) {
        if (!drawsCompletion(row)) continue;
        if (!completions.has(row.task_id)) completions.set(row.task_id, []);
        completions.get(row.task_id).push(attemptOf(row));
    }
    // A receipt may follow the summary whose phase it records, even on this page.
    for (const row of rows) {
        if (!row.historical_terminal) continue;
        const id = row.task_id;
        const frozen = historicalTerminalFacts(row);
        const known = historicalTerminals.get(id);
        const revision = attemptOf(frozen) === attemptOf(row.current_outcome) && row.current_outcome?.revision;
        if (revision) frozen.revision = revision;
        if (known && known.ts === frozen.ts && known.task_attempt === frozen.task_attempt) {
            for (const key of ['wake_summary', 'suggested_name', 'schedule_name', 'place_origin']) frozen[key] ??= known[key];
            if (known.revision && (!revision || isoMicros(revision) < isoMicros(known.revision))) {
                frozen.phase = known.phase;
                frozen.revision = known.revision;
            }
        }
        const older = olderEnding({ attempt: attemptOf(frozen), time: isoMicros(frozen.ts) },
            known && { attempt: attemptOf(known), time: isoMicros(known.ts) });
        if (!older) historicalTerminals.set(id, frozen);
    }
    for (const row of rows) {
        const id = row.task_id;
        const own = row.historical_terminal;
        const rowAttempt = attemptOf(row) ?? attemptOf(row.task_outcome);
        const current = attemptOf(row.current_outcome) < rowAttempt ? null : row.current_outcome;
        const outcome = current || row.task_outcome;
        const outcomeAttempt = attemptOf(outcome);
        // A numbered publication and an unnumbered current result are different too; only two legacy unknowns share.
        const fromOtherAttempt = Boolean(current) && (rowAttempt != null || outcomeAttempt != null)
            && rowAttempt !== outcomeAttempt;
        if (['finalizing', 'unfinished'].includes(row.task_phase) ||
            (row.outcome_final === false && !(fromOtherAttempt && row.current_outcome))) continue;
        const summary = row.system_type === 'task_summary' || row.outcome_final === true;
        const stated = outcome?.task_terminal_status || outcome?.status || outcome?.outcome_axes?.lifecycle?.status
            || (typeof outcome === 'string' && outcome) || row.task_terminal_status
            || (summary && (row.status || row.outcome_axes?.lifecycle?.status));
        // A retained phase can travel on the summary itself after its task result was pruned.
        // A copy on narration alone still awaits the census and the task detail lookup.
        const hasEnding = stated || row.summary_kind === 'terminal_root_projection'
            || (own && row.summary_kind === 'terminal_result_projection') || row.outcome_final === true;
        // Another attempt's result is backed only by a receipt of its own attempt: when that attempt is unknown it
        // never falls back to this row's publication, whose status, words and clock belong to the row.
        const attempt = fromOtherAttempt ? outcomeAttempt : outcomeAttempt ?? rowAttempt ?? attemptOf(own);
        const retained = historicalTerminals.get(id);
        const frozen = retained && attemptOf(retained) === attempt ? retained : own;
        const historical = hasEnding && attempt === attemptOf(frozen) ? frozen : null;
        const status = stated || historical?.status;
        if (!status) continue;
        // A newer result piggybacked on an old row does not gain its time, story, metrics or history placement.
        const source = fromOtherAttempt ? { ...historical, task_id: id, task_attempt: outcomeAttempt, terminal_time: null }
            : { ...row, ...historical };
        const detail = { ...source, ...(typeof outcome === 'object' && outcome), status, task_terminal_status: status };
        if (current?.outcome_phase && !current.terminal_delivery) delete detail.terminal_delivery;
        if (detail.outcome_final === false || !taskDoneIsTerminal(detail)) continue;
        // A receipt can retain a delivery warning that its lifecycle alone cannot reproduce.
        // When the recorded details explain the same phase, keep them for the expanded card.
        const measured = taskTerminalPhase(detail);
        const published = current?.outcome_phase || (!hasOutcomeDetails(current) &&
            (historical?.phase || (detail.outcome_final === true && detail.outcome_phase)));
        const recorded = hasOutcomeDetails(detail);
        const phase = published || (recorded ? measured : detail.outcome_phase || measured);
        const stamp = endingStamp(detail, current, historical, phase === measured);
        const card = cards.get(id);
        const known = endings.get(id)?.stamp || card?.replayedEnding;
        if ((card?.cardAttempt != null && (stamp.attempt == null || stamp.attempt < card.cardAttempt)) || olderEnding(stamp, known)) {
            if (!endings.has(id)) endings.set(id, null);  // the already displayed ending wins this page
        } else {
            const newAttempt = stamp.attempt != null && stamp.attempt !== card?.replayedEnding?.attempt;
            // A detail that explains its phase is applied (`rich`). Its completion row is drawn where this page records
            // the completion, where a revision states it, or where the card already shows one that must not keep an
            // older ending's words (`drawn`).
            const drawn = (completions.get(id) || []).includes(stamp.attempt)
                || Boolean(current?.outcome_phase)
                || Boolean(cards.get(id)?.items?.some(item => String(item.dedupeKey || '').startsWith('task_done|')));
            const rich = phase === measured && (recorded || drawn || newAttempt);
            endings.set(id, { detail, phase, stamp, rich, drawn });
        }
    }
    for (const [id, ending] of endings) {
        const frozen = historicalTerminals.get(id);
        if (!ending || !frozen) continue;
        if (ending.stamp.attempt !== attemptOf(frozen)) {
            // Old compact receipts could omit the ordinal. An identical usable solve call can corroborate the quiet
            // flag of the selected publication; it cannot lend that publication an attempt, time, words or counters.
            const solve = ending.detail.model_execution, retained = frozen.model_execution;
            if (attemptOf(frozen) == null && ending.detail.system_type === 'task_summary'
                    && frozen.provenance === 'canonical_task_result_after_finalization'
                    && frozen.phase === ending.phase && lifecycleOf(frozen) === ending.stamp.lifecycle
                    && solve?.source === 'usable_solve_response' && retained?.source === solve.source
                    && solve.llm_call_id && solve.llm_call_id === retained.llm_call_id
                    && typeof frozen.quiet_effective === 'boolean') ending.detail.quiet_effective ??= frozen.quiet_effective;
            continue;
        }
        for (const key of ['wake_summary', 'suggested_name', 'schedule_name', 'place_origin', 'quiet_effective']) {
            if (frozen[key] != null && ending.detail[key] == null) ending.detail[key] = frozen[key];
        }
    }
    return endings;
}
