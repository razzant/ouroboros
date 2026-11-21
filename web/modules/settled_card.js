// The settled task card's one seam: its Files row (result_files.js) and its
// Continue action (task_continue.js) follow the same settled task detail; the
// work's review-gate field (review_gate_card.js) follows every detail, live ones too.
// A replayed history row settles its card from the terminal facts it froze (historicalTerminalFacts).

import { syncResultFilesItem } from './result_files.js';
import { syncReviewGateItem } from './review_gate_card.js';
import { isReplayEvidenceRow, noteToolHostMetrics, positiveTaskTerminalFact, toolEvidenceIncomplete, toolEvidenceView } from './chat_activity.js';
import { referencesTaskOnly } from './chat_row_placement.js';
import { isTerminalTaskDetail, taskDoneIsTerminal, taskTerminalPhase } from './log_events.js';
import { isoMicros } from './utils.js';
import { syncContinueAction } from './task_continue.js';

/** The finished root's closing line, shared by its card, room row and portal. */
export function closingLine(record, title = record?.titleEl?.textContent || record?.suggestedName || '') {
    if (!record?.finished) return '';
    if (record.causeHeld) return record.collapsedActivity || '';
    const own = !record.isSubagent && record.wakeSummary || '';
    return own !== title ? own : '';
}

/** @returns {boolean} whether the card's timeline items changed (the Files row) */
export function syncSettledItems(record, detail, options) {
    syncContinueAction(record, detail, options);  // a late offer read writes through the caller's viewport seam
    const gate = syncReviewGateItem(record, detail);
    return syncResultFilesItem(record, detail) || gate;
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
    Object.assign(record, { cardAttempt: attempt, attemptMetrics: {}, rounds: null, durationSec: null,
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
export function foldCardMetrics(record, row, facts, selectedAttempt) {
    const attempt = attemptOf(row);
    // The gateway marks every row of a task whose stored result is still open `unfinished`, so the phase
    // alone is not a frozen ending. Only a summary row is one: the recorded ending of an attempt that is
    // not the current execution. It neither selects the card's attempt nor owns its counters; its
    // invocation evidence still folds below — the journal is the task's, not one attempt's. The open
    // attempt's own live carriers (progress, evidence, narration) keep folding exactly as before.
    const frozenEnding = row.task_phase === 'unfinished' && row.system_type === 'task_summary';
    selectCardAttempt(record, selectedAttempt ?? (frozenEnding ? null : attempt));
    const owned = !frozenEnding && attempt === record.cardAttempt;
    if (!owned && !facts.journal) return null;
    const { rounds, duration, calls, errors } = facts;
    if (owned) {
        if (rounds !== null) record.rounds = Math.max(record.rounds ?? 0, rounds);
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
