// One delegated-run observation as a task-card timeline row (#1350). The typed
// `delegated_activity` progress field is produced by ouroboros/delegate_activity.py:
// an exact run-journal range (after_seq, source.read_through], its executor's
// attributed words, thinking, problems and technical events, the retained source
// ref and any range not read yet. This module only projects those facts; it never
// reads the frame's text to decide what is speech.
//
// Identity: every event of an exact record belongs to one part (a joined delta part
// names each fragment's seq and Unicode code point offset in `cuts`) or to the technical seq runs,
// or is a gap inside its range (an event not read whole): no row claims it, it stays visible
// until a later read covers it, and no words join across it.
// `reconcileDelegatedItems` walks a card's items in timeline order and presents each
// (run_id, seq) once while its identity detail fits: a later record shows
// only what no earlier row showed, and one wholly shown before is hidden. The same
// pass runs live, after a reload and after a reconnect, so they agree. Cut/run-budget
// fallbacks disclose possibly repeated text/counts and retain the source. Window rows
// (an engine without the stream) have no seq and stay as they are; a
// provisional window (a failed exact read) folds once exact records cover its range, holes included.
// An engine-keyed utterance (`speech.key` within its run and actor) is one utterance
// in any record: once its complete text exists in the card, its first visible part
// shows it and every other part folds; until then its draft pieces, seq-projected and
// never joined across unread or unshown events or by window rows, show joined at the first one,
// each keeping its preview note and source. An abandoned draft is marked; the typed final
// answer is its own disclosure. Equal words are never compared to decide identity.
//
// Presentation (DESIGN "Conversation activity block"): the executor's messages are
// readable in the row without a second Expand; thinking sits behind one labelled
// disclosure; technical events are one count line whose ordered detail and source
// downloads open with the row's own Expand; problems and unread ranges stay
// visible. The row is host progress: it never claims the card title or collapsed
// line (promote/human false). Consecutive silent rows of one run read as one row.
import { escapeHtmlAttr, escapeHtmlText as escapeHtml, renderMarkdown } from './utils.js';
import { taskSourceDownloadUrl } from './api_client.js';
import { tr } from './i18n.js';

const RECENT_TECHNICAL = 24;
// A silent stretch row lists every folded observation's source; past this many the
// next silent observation opens a new row, so each row's list stays short.
const STRETCH_MEMBERS = 40;

export function delegatedActivityOf(evt) {
    const activity = evt?.delegated_activity;
    return activity && typeof activity === 'object' && activity.run_id && Array.isArray(activity.parts)
        ? activity : null;
}

function actorLabel(actor) {
    return actor ? String(actor).split('/').join(' · ') : 'executor';
}

function partsOf(activity) {
    return (activity.parts || []).filter((part) => part && typeof part === 'object' && typeof part.text === 'string');
}

function isExact(activity) {
    return activity.source?.kind === 'run_events' && Number.isInteger(activity.after_seq)
        && Number.isInteger(activity.source.read_through);
}

function spanText(after, through) {
    return `seq ${Number(after || 0) + 1}–${through}`;
}

// -- seq interval sets: sorted, merged, inclusive [lo, hi] pairs -------------------------

function claim(list, lo, hi) {
    if (!(hi >= lo)) return list;
    const out = [];
    let next = [lo, hi];
    for (const [a, b] of list) {
        if (b + 1 < next[0]) out.push([a, b]);
        else if (next[1] + 1 < a) { out.push(next); next = [a, b]; }
        else next = [Math.min(a, next[0]), Math.max(b, next[1])];
    }
    out.push(next);
    return out.sort((x, y) => x[0] - y[0]);
}

function has(list, seq) {
    return list.some(([a, b]) => seq >= a && seq <= b);
}

function overlapCount(list, lo, hi) {
    return list.reduce((sum, [a, b]) => sum + Math.max(0, Math.min(b, hi) - Math.max(a, lo) + 1), 0);
}

function covers(list, lo, hi) {
    return hi < lo || overlapCount(list, lo, hi) === hi - lo + 1;
}

function firstFree(list, lo, hi) {
    let seq = lo;
    for (const [a, b] of list) {
        if (b < seq) continue;
        if (a > seq) break;
        seq = b + 1;
    }
    return seq <= hi ? seq : null;
}

// The producer's reasons for events inside a read range that did not reach it whole
// (ouroboros/delegate_activity.py::_holes).
const NOT_READ_WHOLE = new Set(['not_served', 'unreadable', 'not_read']);

/** Claim what an exact record read: its range less the events it names as not read whole,
 * which no row shows and nothing may be joined across. */
function claimRead(list, activity) {
    const lo = activity.after_seq + 1, hi = activity.source.read_through;
    const inside = (activity.gaps || []).filter((gap) => NOT_READ_WHOLE.has(gap.reason)
        && Number(gap.after_seq) + 1 >= lo && Number(gap.through_seq) <= hi)
        .sort((a, b) => Number(a.after_seq) - Number(b.after_seq));
    let next = lo;
    for (const gap of inside) {
        list = claim(list, next, Number(gap.after_seq));
        next = Math.max(next, Number(gap.through_seq) + 1);
    }
    return claim(list, next, hi);
}

// -- one record against what earlier rows already showed ---------------------------------

function projectPart(part, claimed) {
    if (!Number.isInteger(part.seq)) return { part, text: part.text };
    const fragments = [[part.seq, 0], ...(Array.isArray(part.cuts) ? part.cuts : [])];
    const last = Number.isInteger(part.last_seq) ? part.last_seq : part.seq;
    if (!has(claimed, part.seq)) {
        const repeats = fragments.slice(1).some(([seq]) => has(claimed, seq)) || has(claimed, last);
        return repeats ? { part, text: part.text, repeats } : { part, text: part.text };
    }
    const fresh = fragments.find(([seq]) => !has(claimed, seq));
    // Python's cuts count Unicode code points, not JavaScript's UTF-16 code units.
    if (fresh) return { part, text: Array.from(part.text).slice(fresh[1]).join(''), continued: true, start: fresh[0] };
    if (has(claimed, last)) return null;
    // The unseen fragments lie past what this record's preview or cut budget can locate.
    return part.cuts_truncated ? { part, text: part.text, repeats: true } : { part, text: '', beyondPreview: true };
}

// -- engine speech identity ----------------------------------------------------------------

function scopeOf(activity, actor, key) {
    return JSON.stringify([activity.run_id, actor || '', key]);
}

function speechScope(activity, part) {
    const key = part.kind === 'message' && !part.final ? part.speech?.key : null;
    return typeof key === 'string' && key ? scopeOf(activity, part.actor, key) : null;
}

/** What the card's records, in timeline order, say about each keyed utterance: its complete
 * part (the first exact one, else a window's; re-reads repeat it), whether the engine abandoned it, and its draft pieces
 * grouped where they join. A piece is a part's seq projection against the records before it,
 * as its row would show it, so an overlapping re-read adds only unseen fragments. A piece
 * joins only after the group's last seq, across events the records so far read whole and
 * whose pieces of the utterance a preview kept (`speech_omitted`, `leftOut`): words are never
 * joined across words not read or not shown, nor behind words that came after them. A
 * window part has no seq and joins nothing. */
function speechIndex(activities) {
    const utterances = new Map();
    const of = (scope) => utterances.get(scope)
        || utterances.set(scope, { complete: null, abandoned: false, groups: [] }).get(scope);
    const claimed = new Map();
    const unshown = new Map();
    const unproven = new Map();
    for (const activity of activities) {
        const run = activity.run_id;
        leftOut(activity, unshown, unproven);
        for (const part of partsOf(activity)) {
            const scope = speechScope(activity, part);
            if (!scope) continue;
            const utterance = of(scope);
            if (part.speech.op === 'complete') {
                // A window row's complete text is bounded: an exact record's takes its place.
                if (!utterance.complete || (isExact(activity) && !isExact(utterance.complete.activity))) {
                    utterance.complete = { part, activity };
                }
                continue;
            }
            const piece = isExact(activity) && Number.isInteger(part.seq) && projectPart(part, claimed.get(run) || []);
            if (!piece) continue;
            const group = utterance.groups.at(-1);
            const last = group ? group.at(-1).part.last_seq ?? group.at(-1).part.seq : null;
            const start = piece.start ?? part.seq;
            const between = (list) => overlapCount(list || [], last + 1, start - 1);
            if (last !== null && start > last && covers(claimRead(claimed.get(run) || [], activity), last + 1, start - 1)
                && !between(unshown.get(scope)) && !between(unproven.get(run))) {
                group.push({ ...piece, from: activity });
            } else {
                utterance.groups.push([{ ...piece, from: activity }]);
            }
        }
        for (const mark of Array.isArray(activity.speech_abandoned) ? activity.speech_abandoned : []) {
            if (typeof mark?.key === 'string' && mark.key) of(scopeOf(activity, mark.actor, mark.key)).abandoned = true;
        }
        if (isExact(activity)) claimed.set(run, claimRead(claimed.get(run) || [], activity));
    }
    return { utterances, owners: new Set(), spoke: new Set() };
}

/** Where a record read speech its preview did not keep: each keyed piece it names (`speech_omitted`) under its
 * utterance, and the whole range of a record that cannot say which (an older record, or past the list's bound). */
function leftOut(activity, unshown, unproven) {
    if (!isExact(activity)) return;
    const named = Array.isArray(activity.speech_omitted) ? activity.speech_omitted : null;
    if (Number(activity.omitted?.message || 0) && (!named || activity.speech_omitted_truncated)) {
        unproven.set(activity.run_id, claim(unproven.get(activity.run_id) || [], activity.after_seq + 1, activity.source.read_through));
    }
    for (const piece of named || []) {
        if (typeof piece?.key !== 'string' || !piece.key || !Number.isInteger(piece.seq)) continue;
        const scope = scopeOf(activity, piece.actor, piece.key);
        unshown.set(scope, claim(unshown.get(scope) || [], piece.seq, Number.isInteger(piece.last_seq) ? piece.last_seq : piece.seq));
    }
}

/** A group's pieces as one text; a cut preview's omission note ends its own paragraph. */
function joinedPieces(pieces) {
    if (pieces.length === 1) return pieces[0];
    return { part: pieces[0].part, text: pieces.map((piece) => piece.text + (piece.part.truncated ? '\n\n' : '')).join(''), pieces };
}

/** One part under its utterance, in timeline order. A keyed utterance with a complete text
 * shows it at its first part and folds every other; without one each group of its draft
 * shows joined at the group's first piece and folds the rest, the first view marked when
 * abandoned. A typed final answer folds once the same run and actor spoke before it. */
function speak(activity, part, shown, speech) {
    const voice = scopeOf(activity, part.actor, '');
    if (part.final) return shown && { ...shown, answer: { folded: speech.spoke.has(voice) } };
    const scope = speechScope(activity, part);
    const utterance = scope ? speech.utterances.get(scope) : null;
    if (utterance?.complete) {
        if (speech.owners.has(scope)) return null;
        speech.owners.add(scope);
        speech.spoke.add(voice);
        const { part: complete, activity: from } = utterance.complete;
        return { part: complete, text: complete.text, from, updated: complete.updated === true,
            sources: [...utterance.groups.flat().map((piece) => piece.from), from] };
    }
    const group = utterance?.groups.find((pieces) => pieces.some((piece) => piece.part === part));
    const view = group ? group[0].part === part && joinedPieces(group) : shown;
    // Only whitespace so far (the space between words, read alone): it joins the words around it, never a row.
    if (!view || (!view.beyondPreview && !view.text.trim())) return null;
    if (part.kind === 'message') speech.spoke.add(voice);
    if (!utterance?.abandoned || speech.owners.has(scope)) return view;
    speech.owners.add(scope);
    return { ...view, abandoned: true };
}

function projectTechnical(activity, claimed) {
    const technical = activity.technical;
    if (!technical?.count) return null;
    const recent = (technical.recent || []).filter((row) => !Number.isInteger(row.seq) || !has(claimed, row.seq));
    if (Array.isArray(technical.seqs) && isExact(activity)) {
        const fresh = technical.seqs.reduce((sum, [a, b]) => sum + (b - a + 1) - overlapCount(claimed, a, b), 0);
        const first = technical.seqs.map(([a, b]) => firstFree(claimed, a, b)).find((seq) => seq !== null);
        return fresh ? { count: fresh, of: technical.count, labels: technical.labels || [], otherKinds: technical.other_kinds || 0,
            recent, recentOmitted: technical.recent_omitted || 0, since: first ?? null, partial: fresh < technical.count } : null;
    }
    // No seq runs (window rows, or a record past its run bound): counted as reported.
    const lo = Number(activity.after_seq || 0) + 1, hi = Number(activity.source?.read_through ?? activity.through_seq);
    return { count: technical.count, of: technical.count, labels: technical.labels || [], otherKinds: technical.other_kinds || 0,
        recent, recentOmitted: technical.recent_omitted || 0, since: null, partial: false,
        inexact: isExact(activity) && overlapCount(claimed, lo, hi) > 0 };
}

function isSuperseded(activity, coverage) {
    return Boolean(activity.source?.provisional)
        && covers(coverage, Number(activity.after_seq || 0) + 1, Number(activity.through_seq));
}

/** One record against the card: `claimed`, what earlier rows showed; `observed`, every exact range read, which
 * supersedes a provisional window and answers its "not shown" lines; `coverage`, the events read whole, which alone
 * answers a hole's own line. */
function project(activity, claimed, observed, coverage, speech = speechIndex([activity])) {
    const exact = isExact(activity);
    const superseded = isSuperseded(activity, observed);
    const parts = superseded ? [] : partsOf(activity)
        .map((part) => speak(activity, part, exact ? projectPart(part, claimed) : { part, text: part.text }, speech))
        .filter(Boolean);
    const gaps = (activity.gaps || []).filter((gap) => !covers(NOT_READ_WHOLE.has(gap.reason) ? coverage : observed,
        Number(gap.after_seq) + 1, Number(gap.through_seq)));
    const technical = projectTechnical(activity, claimed);
    const omitted = Object.entries(activity.omitted || {}).filter(([, n]) => n);
    // An overlap proves only those earlier events were shown. A bounded reread
    // can omit a new tail, whose warning and source must remain visible.
    const fullyShown = exact && covers(claimed, Number(activity.after_seq) + 1, activity.source.read_through);
    const view = { activity, parts, technical: superseded ? null : technical,
        gaps: superseded ? [] : gaps, omitted, superseded, fullyShown };
    view.empty = !view.parts.length && !view.technical && !view.gaps.length && !superseded
        && !(omitted.length && !fullyShown) && !activity.source?.rows_omitted;
    view.silent = !view.empty && !view.parts.length && !view.gaps.length && !superseded && !omitted.length
        && !activity.source?.provisional && !activity.source?.rows_omitted && Boolean(view.technical);
    return view;
}

function absorb(head, view) {
    const first = head.technical;
    const stretch = head.stretch || (head.stretch = { members: [head.activity], count: first.count, since: first.since,
        labels: new Map(first.labels), otherKinds: first.otherKinds, recent: [...first.recent],
        recentOmitted: first.recentOmitted, inexact: first.partial || Boolean(first.inexact) });
    const next = view.technical;
    stretch.members.push(view.activity);
    stretch.count += next.count;
    for (const [label, n] of next.labels) stretch.labels.set(label, (stretch.labels.get(label) || 0) + n);
    stretch.otherKinds = Math.max(stretch.otherKinds, next.otherKinds);
    stretch.recentOmitted += next.recentOmitted + Math.max(0, stretch.recent.length + next.recent.length - RECENT_TECHNICAL);
    stretch.recent = [...stretch.recent, ...next.recent].slice(-RECENT_TECHNICAL);
    stretch.inexact = stretch.inexact || next.partial || Boolean(next.inexact);
}

// What decides a row's markup for one fixed record: its identity-bearing shape, not its text.
// A keyed part shows words from the records that hold its complete text or its later draft
// pieces, named here, so a same-length correction or a new piece still re-renders the row.
function signature(view) {
    const technical = view.stretch || view.technical;
    const origin = (from, part) => (from ? [from.after_seq, from.source?.read_through ?? from.through_seq, part.last_seq ?? null] : null);
    return JSON.stringify([view.hidden, view.silent, view.superseded, view.fullyShown,
        view.parts.map((shown) => [shown.part.seq ?? null, shown.text.length, Boolean(shown.continued || shown.repeats || shown.beyondPreview),
            origin(shown.from, shown.part), Boolean(shown.updated), Boolean(shown.abandoned), shown.answer ? shown.answer.folded : null,
            (shown.pieces || []).map((piece) => [origin(piece.from, piece.part), piece.text.length,
                Boolean(piece.repeats || piece.beyondPreview || piece.part.truncated)]),
            (shown.sources || []).map((source) => source.after_seq)]),
        technical && [technical.count, technical.since ?? null, technical.recent.length, Boolean(technical.partial || technical.inexact),
            [...(technical.labels || [])], view.stretch?.members.length || 1],
        view.gaps.map((gap) => [gap.after_seq, gap.through_seq])]);
}

/**
 * Project every delegated row in timeline order: each (run_id, seq) once within
 * the disclosed identity budgets, provisional windows folded once covered, and silent rows of
 * one run shown as its first row. Returns the items whose projection changed.
 */
export function reconcileDelegatedItems(record) {
    const observed = new Map();
    const coverage = new Map();
    for (const item of record?.items || []) {
        const activity = item.activity;
        if (activity && isExact(activity)) {
            observed.set(activity.run_id, claim(observed.get(activity.run_id) || [], activity.after_seq + 1, activity.source.read_through));
            coverage.set(activity.run_id, claimRead(coverage.get(activity.run_id) || [], activity));
        }
    }
    const speech = speechIndex((record?.items || []).map((item) => item.activity)
        .filter((activity) => activity && !isSuperseded(activity, observed.get(activity.run_id) || [])));
    const claimed = new Map();
    const views = [];
    let stretch = null;
    for (const item of record?.items || []) {
        const activity = item.activity;
        if (!activity) { stretch = null; continue; }
        const run = activity.run_id;
        const view = project(activity, claimed.get(run) || [], observed.get(run) || [], coverage.get(run) || [], speech);
        if (isExact(activity)) claimed.set(run, claimRead(claimed.get(run) || [], activity));
        view.hidden = view.empty;
        if (view.silent && stretch?.run === run && (stretch.view.stretch?.members.length || 1) < STRETCH_MEMBERS) {
            absorb(stretch.view, view);
            view.hidden = true;
        } else if (!view.hidden) {
            stretch = view.silent ? { run, view } : null;
        }
        views.push([item, view]);
    }
    const changed = [];
    for (const [item, view] of views) {
        const shape = signature(view);
        if (item.delegatedSignature !== shape) changed.push(item);
        item.delegatedView = view;
        item.delegatedSignature = shape;
    }
    return changed;
}

/** The projection a row renders; a row outside a reconciled card stands alone. */
export function delegatedLineView(item) {
    if (item.delegatedView?.activity === item.activity) return item.delegatedView;
    return project(item.activity, [], [], []);
}

// -- presentation --------------------------------------------------------------------------

function countsLine(technical) {
    const count = Number(technical?.count || 0);
    if (!count) return '';
    const labels = technical.labels instanceof Map ? [...technical.labels] : technical.labels || [];
    const kinds = labels.map(([label, n]) => `${label} ×${n}`);
    if (technical.otherKinds) kinds.push(`+${technical.otherKinds} other kinds`);
    const head = `${count} technical event${count === 1 ? '' : 's'}${technical.since ? ` since seq ${technical.since}` : ''}`;
    const note = technical.partial ? ` (${technical.of - count} of this observation's shown above; kinds cover all ${technical.of})`
        : technical.inexact ? ' (may include events counted above)' : '';
    return [head + note, ...kinds].join(' · ');
}

function gapLine(gap, activity) {
    const run = activity.run_id;
    if (NOT_READ_WHOLE.has(gap.reason)) {
        return `Not read whole: ${spanText(gap.after_seq, gap.through_seq)} (${gap.reason}). An utterance around it may lack words; the retained source keeps every line served.`;
    }
    if (gap.reason === 'stream_end_before_fence') {
        return `Not shown: ${spanText(gap.after_seq, gap.through_seq)} (stream ended before the observed cursor); coverage unresolved on run ${run}.`;
    }
    const span = spanText(gap.after_seq, gap.through_seq);
    if (gap.final) {
        return `Not shown: ${span} (${gap.reason}). The wait returned before reading them; they remain on run ${run}'s journal, and another wait on the run reads them.`;
    }
    return `Not shown yet: ${span} (${gap.reason}); read at the run's next advance or its end.`;
}

function whoOf(view) {
    const activity = view.activity;
    const speakers = view.parts.map(({ part }) => part.actor).filter(Boolean);
    const workers = (view.technical?.recent || activity.technical?.recent || []).map((row) => row.actor).filter(Boolean);
    const actors = [...new Set(speakers.length ? speakers : workers.length ? workers
        : [activity.latest_message?.actor].filter(Boolean))];
    return actors.length ? actors.map(actorLabel).join(', ') : 'executor';
}

function sourceLink(activity, label, title = '') {
    const ref = activity.source?.ref;
    const url = taskSourceDownloadUrl(activity.task_id, ref);
    if (!url) return '';
    const name = String(ref.path).split('/').at(-1);
    return `<a class="chat-delegated-source" href="${escapeHtmlAttr(url)}" download="${escapeHtmlAttr(name)}"${title ? ` title="${escapeHtmlAttr(title)}"` : ''}>${escapeHtml(label)}</a>`;
}

function sourceFacts(activity) {
    const source = activity.source || {};
    return `${source.events || 0} events · ${Number(source.ref?.size || 0)} bytes (JSONL)`;
}

function sourceHtml(activity, { compact = false } = {}) {
    const source = activity.source || {};
    const range = spanText(activity.after_seq, source.read_through ?? activity.through_seq);
    if (source.kind === 'timeline_window') {
        return escapeHtml(`Bounded timeline window of run ${activity.run_id} (${source.reason || 'exact event stream unavailable'}); exact originals stay on the run's journal.`);
    }
    const link = compact ? sourceLink(activity, range, sourceFacts(activity))
        : sourceLink(activity, `Full source · ${range} · ${sourceFacts(activity)}`);
    if (link) return link;
    return escapeHtml(`${compact ? range : `Source: run ${activity.run_id} ${range}`}${source.retained === false ? ' (retention failed; originals stay on the run\'s journal)' : ''}`);
}

/** Headline of one projected row. */
export function delegatedHeadline(view) {
    const run = String(view.activity.run_id);
    if (view.superseded) return `${whoOf(view)} · window view (superseded) · ${run}`;
    return view.silent ? `${whoOf(view)} · technical activity · ${run}` : `${whoOf(view)} · ${run}`;
}

/** The row body: messages first, then problems, one Thinking disclosure, counts, gaps. */
export function delegatedActivityBodyHtml(view, { expanded = false } = {}) {
    const activity = view.activity;
    // Speakers are labelled only when words or thoughts come from more than one; a
    // problem names its actor in its own line.
    const named = new Set(view.parts.filter(({ part }) => part.kind !== 'problem').map(({ part }) => part.actor)).size > 1;
    const actor = ({ part }) => (named ? `<span class="chat-delegated-actor">${escapeHtml(actorLabel(part.actor))}</span>` : '');
    const full = sourceLink(activity, 'open the full source');
    // A truncated preview already ends in the shared omission note (length included);
    // the row adds where the whole text is, one click away: each joined draft piece names its
    // own record. A complete text replacing its streamed draft says so when they differ; the
    // row's source holds the draft.
    const notes = (shown) => {
        const draft = sourceLink(activity, 'the full source');
        return [...(shown.pieces || [shown]).map((piece, index) => {
            const out = [];
            if (piece.continued && !index) out.push('Continues an utterance whose beginning is shown above.');
            if (piece.repeats) out.push('May repeat text shown above.');
            if (piece.beyondPreview) out.push(`Continues beyond the preview shown above (up to seq ${piece.part.last_seq}).`);
            if (shown.abandoned && !index) out.push('Abandoned: the executor never completed these words.');
            const whole = piece.part.truncated || piece.beyondPreview;
            const from = piece.from || shown.from;
            const own = from && from !== activity ? sourceLink(from, 'open the full source') : full;
            const where = whole ? (own ? `Read it whole: ${own}.` : escapeHtml("The run's journal holds it whole.")) : '';
            return out.length || whole ? `<div class="chat-delegated-note">${[escapeHtml(out.join(' ')), where].filter(Boolean).join(' ')}</div>` : '';
        }), shown.updated ? `<div class="chat-delegated-note">Updated. ${draft ? `The streamed draft is in ${draft}.` : escapeHtml("The streamed draft stays on the run's journal.")}</div>` : '',
        ].join('');
    };
    const text = (shown) => (shown.text ? renderMarkdown(shown.text, { inlineHeadingBreaks: true }) : '');
    const html = [];
    if (view.superseded) {
        html.push(`<div class="chat-delegated-note">${escapeHtml(`The exact read of ${spanText(activity.after_seq, activity.through_seq)} failed here (${activity.source.reason}); a later row shows those exact events.`)}</div>`);
    }
    for (const shown of view.parts) {
        const message = `<div class="chat-delegated-message">${actor(shown)}${text(shown)}${notes(shown)}</div>`;
        if (shown.answer) {
            // The run's typed final answer: its own disclosure in the Thinking look.
            html.push(`<details class="chat-delegated-answer"${shown.answer.folded ? '' : ' open'}><summary>${escapeHtml(tr('task.delegated.answer', 'Answer'))}</summary>${message}</details>`);
        } else if (shown.part.kind === 'message') {
            html.push(message);
        } else if (shown.part.kind === 'problem') {
            html.push(`<div class="chat-delegated-problem">${escapeHtml(tr('task.delegated.problem', 'Problem'))}: ${escapeHtml([actorLabel(shown.part.actor), shown.part.label].filter(Boolean).join(' · '))}: ${escapeHtml(shown.text)}${notes(shown)}</div>`);
        }
    }
    const thinking = view.parts.filter((shown) => shown.part.kind === 'thinking');
    if (thinking.length) {
        html.push(`<details class="chat-delegated-thinking"><summary>Thinking${thinking.length > 1 ? ` (${thinking.length})` : ''}</summary>${thinking.map((shown) => `<div class="chat-delegated-thought">${actor(shown)}${text(shown)}${notes(shown)}</div>`).join('')}</details>`);
    }
    const technical = view.stretch || view.technical;
    const counts = countsLine(technical);
    if (counts) html.push(`<div class="chat-delegated-technical">${escapeHtml(counts)}</div>`);
    if (view.omitted.length && !view.fullyShown) {
        const omitted = view.omitted.map(([kind, n]) => `${n} ${kind}`).join(', ');
        html.push(`<div class="chat-delegated-gap">Not in this row: ${escapeHtml(omitted)}${full ? ` — ${full}` : " (on the run's journal)"}.</div>`);
    }
    if (activity.source?.provisional && !view.superseded) {
        html.push(`<div class="chat-delegated-gap">${escapeHtml(`Bounded timeline window shown meanwhile: the exact read failed (${activity.source.reason}).`)}</div>`);
    }
    for (const gap of view.gaps) {
        html.push(`<div class="chat-delegated-gap${gap.final ? ' chat-delegated-final' : ''}">${escapeHtml(gapLine(gap, activity))}</div>`);
    }
    const rowsOmitted = Number(activity.source?.rows_omitted || 0);
    if (rowsOmitted) html.push(`<div class="chat-delegated-gap">${escapeHtml(`+${rowsOmitted} earlier timeline rows not shown (bounded window; the run's own timeline has them).`)}</div>`);
    if (expanded) {
        const recent = technical?.recent || [];
        if (recent.length) {
            html.push(`<ol class="chat-delegated-events">${recent.map((row) => `<li><span class="chat-delegated-seq">#${escapeHtml(String(row.seq ?? '?'))}</span> ${escapeHtml(row.label || '')}${row.detail ? ` — ${escapeHtml(row.detail)}` : ''}</li>`).join('')}</ol>`);
        }
        const members = view.stretch?.members || [activity];
        if (technical?.recentOmitted) {
            html.push(`<div class="chat-delegated-note">${escapeHtml(`${Number(technical.recentOmitted)} earlier technical events are listed in the full source${members.length > 1 ? 's' : ''}.`)}</div>`);
        }
        // Words shown here from later records (joined draft pieces; a complete text and the draft it
        // replaced) keep those records' sources one click away, though their own rows folded.
        const sources = [...new Set([...members, ...view.parts.flatMap((shown) => [...(shown.sources || []),
            ...(shown.pieces || []).map((piece) => piece.from)])])];
        html.push(`<div class="chat-delegated-note chat-delegated-sources">${sources.length > 1
            ? `Full sources: ${sources.map((member) => sourceHtml(member, { compact: true })).join(' · ')}`
            : sourceHtml(activity)}</div>`);
    }
    return html.join('');
}

/** The summarizer's view of one observation frame, or null for any other frame. */
export function delegatedActivityView(evt) {
    const activity = delegatedActivityOf(evt);
    const type = evt?.type || evt?.event || '';
    if (!activity || !(evt.is_progress || type === 'send_message')) return null;
    const view = project(activity, [], [], []);
    const messages = view.parts.filter(({ part }) => part.kind === 'message').map(({ text }) => text);
    const counts = countsLine(view.technical);
    const gaps = view.gaps.map((gap) => gapLine(gap, activity));
    const run = String(activity.run_id);
    return {
        phase: 'working',
        headline: delegatedHeadline(view),
        body: messages[0] || view.parts.find(({ part }) => part.kind === 'problem')?.text || counts || gaps[0] || '',
        fullBody: [...messages, counts, ...gaps].filter(Boolean).join('\n\n'),
        activityPreview: '',
        visible: true,
        promote: false,
        human: false,
        dedupeKey: `delegated:${run}:${activity.after_seq}-${activity.source?.read_through ?? activity.through_seq}`,
        activity,
    };
}

/** Live arrival of one observation frame: its own item; the same frame re-delivered is skipped. */
export function appendDelegatedItem(record, summary, { ts, rawTs, syntheticKey, headline }) {
    if (record.items.some((item) => item.dedupeKey === syntheticKey && item.sourceTs === rawTs)) {
        return { timelineUpdate: 'duplicate-skip', patchIndex: -1 };
    }
    const item = {
        phase: summary.phase || 'working', headline: headline || 'Update',
        fullHeadline: summary.fullHeadline || headline || 'Update',
        body: summary.body || '', fullBody: summary.fullBody || summary.body || '',
        fullRef: '', truncated: false, receipt: false, ts: ts || '', sourceTs: rawTs, count: 1,
        dedupeKey: syntheticKey, lineKey: `line-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        activity: summary.activity,
    };
    record.items.push(item);
    const others = reconcileDelegatedItems(record).filter((changed) => changed !== item);
    return { timelineUpdate: others.length ? 'render' : 'append', patchIndex: -1 };
}
