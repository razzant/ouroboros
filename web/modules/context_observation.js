// What a room's Context capsule says its works' requests were sent in (docs/DESIGN.md "Composer owner controls"): the
// mode of the latest recorded request of each of this room's own works, read from the activity census rows `/api/state`
// already carries (`context_attempt`, the usage ledger's physical context of a work's latest attempt; `dispatched` is
// the host's commitment before the provider send, never the provider's acceptance). A pure projection: no poller, store
// or persistence. The capsule keeps one newest observation of ended works while its page lives, so their order leaves a
// dated `last`; a page that saw none says `unknown`, never the saved preference. Nothing here proves freshness: a
// census that stops arriving leaves the last fold standing, dated by its request's time.
import { fmt, translateString } from './i18n.js';
import { formatMsgTime } from './utils.js';

export const CONTEXT_MODES = Object.freeze(['nano', 'low', 'max']);
export const CONTEXT_LABELS = Object.freeze({ nano: 'Nano', low: 'Low', max: 'Max' });
export const UNKNOWN_CONTEXT = Object.freeze({ state: 'unknown', modes: [], latest: null, retired: null,
    byWork: new Map(), releasedIds: new Set(), releasedWithoutPreviousIds: new Set() });

const at = (fact) => {
    const stamp = fact?.observed_at || '';
    // Python stamps microseconds; Date.parse alone would make distinct sends in one millisecond tie.
    const fraction = stamp.match(/\.(\d+)/)?.[1] || '';
    return (Date.parse(stamp) || 0) + Number(fraction.slice(3, 6).padEnd(3, '0')) / 1000;
};
const identity = (fact) => `${fact.work}:${fact.task_attempt}:${fact.attempt_id}`;
// A work the window took from the owner's Max runs in Low, temporarily, for the window, and is named Low; the owner's
// choice stays Max beside it (`preferred_at_send`, owner 21.11.2026). Rows of 17-22 Nov 2026 named that state Max with
// chapter maps (`applied` max, `books` maps; a room's own work reads whole books in Max otherwise): they read as Low.
export const asSent = (fact) => (fact?.applied === 'max' && fact.books === 'maps' ? { ...fact, applied: 'low' } : fact);
const newest = (facts) => facts.filter(Boolean).reduce((a, b) => (!a || at(b) > at(a) ? b : a), null);
const sourceIdentity = (fact) => fact?._source_identity || identity(fact);
const previousIdentity = (fact) => identity({ ...fact.previous, work: fact.work });
const sentFact = (fact) => {
    if (fact?.ledger_state !== 'released') return fact;
    if (!fact.previous) return null;
    // Nested context_attempt projections omit work, but a previous attempt belongs
    // to this same task and therefore carries the current work identity here.
    const previous = { ...fact.previous, work: fact.work };
    return { ...previous, _source_identity: identity(previous), _released_identity: identity(fact) };
};

/** This room's root works whose fact belongs to their current attempt; a helper never feeds the room's capsule. */
export function roomContextFacts(rows, chatId) {
    if (!Array.isArray(rows) || chatId == null) return [];
    return rows.filter((row) => row && typeof row === 'object' && Number(row.chat_id ?? 1) === Number(chatId)
        && !row.parent_task_id && !row.delegation_role
        && CONTEXT_MODES.includes(row.context_attempt?.applied)
        && ['dispatched', 'settled', 'unresolved', 'released'].includes(row.context_attempt?.ledger_state)
        && (row.task_attempt == null || row.context_attempt.task_attempt === row.task_attempt))
        .map((row) => {
            const fact = asSent(row.context_attempt);
            return { ...fact, ...(fact.previous ? { previous: asSent(fact.previous) } : {}), work: String(row.activity_id || '') };
        });
}

function revisedRelease(old, fact, releasedIds, withoutPrevious) {
    const key = identity(fact);
    if (withoutPrevious.has(key)) return { ...fact, previous: undefined };
    if (old.previous && !fact.previous) {
        withoutPrevious.add(key);
        releasedIds.add(previousIdentity(old));  // it can leave only by being disproven
        return { ...fact, previous: undefined };
    }
    if (!old.previous) return old;  // no earlier send appears behind a release later on
    if (releasedIds.has(previousIdentity(fact))) return old;
    if (fact.previous.attempt_id === old.previous.attempt_id) return fact;  // the same send; its state may have moved on
    // A released attempt's previous only moves to an older attempt as each newer one is disproven, so a
    // newer previous is a stale read, and the move itself is the proof that the old previous never started.
    if (at(fact.previous) > at(old.previous)) return old;
    releasedIds.add(previousIdentity(old));
    return fact;
}

/**
 * One census read folded into the room's observation. `now`: one mode across this room's works; `mixed`: several
 * works in different modes; `last`: none now, the newest request this page saw; `unknown`: this page saw none.
 * A late read never takes a work back to an older request of the same attempt. A census the host says is incomplete
 * (`complete === false`: `active_chat_activities_complete` or `supervisor_ready` explicitly false) never makes `now`
 * or `mixed` from the subset it lists: its facts still count as seen, and the newest seen request stands as `last`,
 * marked `partial` (or `unknown`, marked, when none was ever seen). Absent flags read as complete (optional contract).
 */
export function observeContext(prior, rows, chatId, { complete = true } = {}) {
    const held = prior?.byWork || new Map();
    const rawFacts = roomContextFacts(rows, chatId);
    // Record every release before stale-attempt ordering collapses multiple rows
    // of one work to its newest fact. A late proof about an older attempt must
    // still retract a previous carried by the newer released attempt.
    const releasedIds = new Set(prior?.releasedIds || []);
    const releasedWithoutPreviousIds = new Set(prior?.releasedWithoutPreviousIds || []);
    rawFacts.filter((fact) => fact.ledger_state === 'released').forEach((fact) => releasedIds.add(identity(fact)));
    const observed = rawFacts.map((fact) => {
        const old = held.get(fact.work);
        if (old && identity(old) === identity(fact)) {
            if (old.ledger_state === 'released') {
                if (fact.ledger_state !== 'released') return old;  // a late dispatch cannot revive a disproven send
                return revisedRelease(old, fact, releasedIds, releasedWithoutPreviousIds);
            }
            if (fact.ledger_state === 'released') return fact;
        }
        return old && old.task_attempt === fact.task_attempt && at(old) > at(fact) ? old : fact;
    });
    const byWork = new Map(complete ? [] : held);
    for (const fact of observed) byWork.set(fact.work, fact);
    let retired = prior?.retired || null;
    if (retired && releasedIds.has(sourceIdentity(retired))) retired = null;
    if (complete) {
        // Only departing works update this one retained fact. An older still-active work must not erase it.
        const ended = [...held.values()].filter((f) => !byWork.has(f.work))
            .map(sentFact).filter((fact) => fact && !releasedIds.has(sourceIdentity(fact)));
        retired = newest([retired, ...ended]);
    }
    // If a resumed released attempt carried a previous send and that previous
    // send is later proven never to have started, retract the carried projection
    // too. The set is monotonic across polls, so stale frames cannot restore it.
    const facts = [...byWork.values()].map(sentFact)
        .filter((fact) => fact && !releasedIds.has(sourceIdentity(fact)));
    if (!complete) {
        const latest = newest([retired, ...facts]);
        return latest ? { state: 'last', modes: [latest.applied], latest, retired, byWork, releasedIds,
            releasedWithoutPreviousIds, partial: true }
            : { ...UNKNOWN_CONTEXT, byWork, releasedIds, releasedWithoutPreviousIds, partial: true };
    }
    if (facts.length) {
        const modes = CONTEXT_MODES.filter((mode) => facts.some((fact) => fact.applied === mode));
        return { state: modes.length > 1 ? 'mixed' : 'now', modes, latest: newest(facts), retired, byWork, releasedIds,
            releasedWithoutPreviousIds };
    }
    return retired ? { state: 'last', modes: [retired.applied], latest: retired, retired, byWork, releasedIds,
        releasedWithoutPreviousIds }
        : { ...UNKNOWN_CONTEXT, byWork, releasedIds, releasedWithoutPreviousIds };
}

// A mode's word as this install says it (the overlay translates the strip's words the same way).
const said = (mode) => translateString(CONTEXT_LABELS[mode] || '');
const listed = (modes) => (modes.length > 1
    ? fmt('{first} and {second}', { first: modes.slice(0, -1).map(said).join(', '), second: said(modes.at(-1)) }) : said(modes[0]));

// The recorded causes of a send that ran below the mode its work held at send (`causeOf` names each).
const LOWERING = ['memory', 'request', 'route', 'window', 'after_start'];

// Why the request ran below the mode its work held at send (`cause`), else that the owner's choice changed since. The
// task card's start row says its recorded reason in these same words (`task_checkpoints.js`). A start the window took
// from the owner's Max names the work's state (Low for now) apart from that choice, by its cause.
export function causeOf(fact, preferred) {
    const asked = said(fact.preferred_at_send);
    const ran = said(fact.applied);
    switch (asked && fact.cause) {
    case 'memory': return fmt('{mode} for now, not {chosen}: the whole books do not fit beside the history.', { mode: ran, chosen: asked });
    case 'request': return fmt('{mode} for now, not {chosen}: even without the history, the request leaves the whole books no room.', { mode: ran, chosen: asked });
    case 'route': return fmt('{mode} for now, not {chosen}: this model’s window cannot hold the whole books.', { mode: ran, chosen: asked });
    case 'window': return fact.applied === 'low' && fact.preferred_at_send === 'max'  // rows of 17-22 Nov: schemas tipped it
        ? fmt('{mode} for now, not {chosen}: the whole books did not fit beside the rest of the start.', { mode: ran, chosen: asked })
        : fmt('{mode} did not fit at the start.', { mode: asked });
    case 'after_start': return fmt('The work moved to {mode} after it started.', { mode: ran });
    default: break;
    }
    return fact.preferred_at_send && fact.preferred_at_send !== preferred
        ? fmt('Sent before {mode} was chosen.', { mode: said(preferred) }) : '';
}

// Low's target missed at this room's newest owner-Low start (owner 15.11.2026, 3A: where the mode is chosen, the
// screen says it before the next start): what the rest of that start's request took, from its own receipt; '' else.
export function lowMissedOf(fact) {
    const n = Number(fact?.no_memory_tokens);
    if (fact?.applied !== 'low' || fact?.profile !== 'owner_low' || !fact?.target_missed || !(n > 0)) return '';
    return fmt('Low’s 250K target was not met at the last start here: without memory it already needed about {n}; '
        + 'memory keeps its shortest tellings.', { n: `${Math.round(n / 1000)}K` });
}

// An explicitly incomplete census: what it listed is kept, but it is not the room's whole current state.
const incomplete = () => fmt('The current read is incomplete.', {});

/**
 * What the capsule shows for an observation and the room's saved mode: the one word and the lit arcs (the observed
 * mode), whether the saved choice needs its dot (only where it and the observation disagree), the head's accessible
 * name ('' leaves the head's own words: agreement looks exactly like today's capsule), and the open hint's one fact
 * line: the observation, `now` or `last`, dated by its newest recorded request, with a short cause on disagreement.
 */
export function contextShown(observation, preferred) {
    const obs = observation || UNKNOWN_CONTEXT;
    const chosen = said(CONTEXT_LABELS[preferred] ? preferred : 'max');
    if (obs.state === 'unknown') {
        return { actual: 'unknown', word: '—', lit: [], differs: true,
            label: fmt('Context: actual unknown, {chosen} chosen', { chosen }),
            fact: { word: 'Unknown', text: obs.partial ? incomplete()
                : 'No request from this room seen since this page opened.' } };
    }
    const time = formatMsgTime(obs.latest?.observed_at)?.short;
    if (obs.state === 'mixed') {
        // Each work's own latest request; the time is the newest of them, not a moment all of them shared.
        return { actual: 'mixed', word: 'Mixed', lit: obs.modes, differs: true,
            label: fmt('Context: Mixed, {modes}, {chosen} chosen', { modes: listed(obs.modes), chosen }),
            fact: { word: 'Mixed',
                text: [fmt('Latest requests of works here: {modes}.', { modes: listed(obs.modes) }),
                    time ? fmt('Newest at {time}.', { time }) : ''].filter(Boolean).join(' ') } };
    }
    const mode = obs.modes[0];
    const word = CONTEXT_LABELS[mode];  // the closed value, which the overlay translates in place
    const when = { word: fmt(obs.state === 'now' ? '{mode} now' : '{mode} last', { mode: said(mode) }),
        time: time ? fmt('Request at {time}.', { time }) : '' };
    // Agreement is today's closed capsule: its own words name it (as the overlay translated them); open, it is dated.
    // A send made under another mode keeps its own reason before the time (a Low the window took from Max, Low chosen
    // since): a temporary Low never reads as the owner's own.
    const note = obs.partial ? incomplete() : '';
    if (mode === preferred) {
        const sent = obs.latest?.preferred_at_send;
        const why = sent && sent !== mode && LOWERING.includes(obs.latest.cause) ? causeOf(obs.latest, preferred) : '';
        const text = [why, when.time, note].filter(Boolean).join(' ');
        return { actual: mode, word, lit: [mode], differs: false, label: '', fact: text ? { word: when.word, text } : null };
    }
    const why = causeOf(obs.latest, preferred);
    const sentence = [why, when.time, note].filter(Boolean).join(' ');
    return { actual: mode, word, lit: [mode], differs: true,
        label: fmt(obs.state === 'now' ? 'Context: {word} now, {chosen} chosen' : 'Context: {word} last, {chosen} chosen', { word: said(mode), chosen }),
        fact: { word: when.word, text: sentence } };
}
