// Reviews A (owner 25.10, quiz answer A): the closed card's Reviews door opens flat rows named for what was
// reviewed; a row opens to one compact line per reviewer with Read into the document reader, and one Record facts
// disclosure keeps the exact record. Domain adapters own state and tone; this leaf only presents saved facts in
// the owner's words (review_record_card.js holds the one vocabulary), with tokens left to Record facts.
import { escapeHtmlAttr as escape, formatMsgTime } from './utils.js';
import { taskReviewSourceUrl, taskSkillReviewSourceUrl } from './api_client.js';
import {
    actorAnswered, landingWords, panelPoolLines, panelPoolRecorded, reviewStateWords, seatAdviser, seatCauseText, seatCostText, seatLimitText,
    seatOwed, seatStateView,
} from './review_record_card.js';
import { statusMarkHtml } from './status_mark.js';

const text = value => String(value ?? '').trim();
const list = value => (Array.isArray(value) ? value : []).filter(item => item && typeof item === 'object');
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const shortSha = value => (/^[0-9a-f]{12,}$/i.test(text(value)) ? text(value).slice(0, 7) : text(value));
const SURFACE_LABELS = {
    commit_gate: 'Code review', change: 'Code review', preflight: 'Early look', result: 'Result review', system: 'System review',
    plan: 'Plan', skill: 'Skill review', review_record: 'Review records', task_acceptance: 'Result review',
};
const GAP_WORDS = {
    missing: 'record missing', unreadable: 'record unreadable', projection_unreadable: 'record could not be shown',
    index_unavailable: 'review index unreadable',
};
// The adapters' tones on the shared status sentence (`.ui-status[data-tone]`); live work draws the moving mark.
const STATUS_TONES = { done: 'ok', ok: 'ok', warn: 'warn', error: 'error' };
const CHEVRON = '<svg class="review-chevron" width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true" focusable="false">'
    + '<path d="M5 7.5 10 12.5 15 7.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></path></svg>';
const CHEVRON_RIGHT = '<svg class="review-chevron-right" width="12" height="12" viewBox="0 0 20 20" fill="none" aria-hidden="true" focusable="false">'
    + '<path d="M7.5 5 12.5 10 7.5 15" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"></path></svg>';

/** One state: a status sentence (its dot and word) for a fact; live work is the moving mark beside its word. */
export function reviewStateMarkup(word, tone) {
    if (tone === 'working') {
        return `<span class="review-state" data-tone="working">${statusMarkHtml('', { shape: 'moving', hue: 'ink' })}`
            + `<span class="review-state-word">${escape(word)}</span></span>`;
    }
    return `<span class="review-state ui-status" data-tone="${STATUS_TONES[tone] || 'neutral'}">${escape(word)}</span>`;
}

const subjectOf = (panel) => {
    const subject = panel?.subject || {};
    return text(panel?.goal) || (subject.base && subject.head ? `${shortSha(subject.base)} → ${shortSha(subject.head)}` : '')
        || (text(subject.kind) === 'worktree' ? 'working changes' : '');
};

export function reviewViews(groups) {
    return groups.flatMap(group => (group.surface === 'review_record' && group.attempts.length ? group.attempts.map((attempt) => {
        // The record's relation to the code that landed (owner 29.10, A): its head names an earlier version, its
        // opened row says the rest. An unreadable record says nothing about any commit.
        const landed = attempt.gap ? landingWords(null) : landingWords(attempt.panel, group.adoptions);
        return {
            ...group, id: `${group.id}:${attempt.id}`, attempts: [attempt], verdict: attempt.verdict, tone: attempt.tone,
            progress: attempt.progress || '', state: attempt.state,
            label: attempt.gap ? 'Review record unavailable'
                : `${SURFACE_LABELS[attempt.panel?.surface] || 'Review'}${landed.earlier ? ' · earlier version' : ''}`,
            subject: attempt.gap ? '' : subjectOf(attempt.panel), landingLine: landed.line, authorReason: landed.reason,
        };
    }) : [group]));
}

const ledgerTally = (panel) => {
    const counted = list(panel.actors).filter(actor => !seatAdviser(actor));
    const answered = counted.filter(actorAnswered).length;
    const owed = counted.filter(actor => seatOwed(actor, panel)).length;
    const unavailable = counted.length - answered - owed;
    const counts = text(panel.surface) === 'preflight' || !counted.length ? 'answered' : 'counted answered';
    return [`${answered} of ${counted.length} ${counts}`, unavailable > 0 ? `${unavailable} unavailable` : '',
        owed ? `${owed} awaiting` : ''].filter(Boolean).join(' · ');
};

function headState(view) {
    const attempt = view.attempts.length === 1 ? view.attempts[0] : null;
    if (attempt?.gap) return { word: 'could not be read', tone: 'warn' };
    const phrase = view.progress || view.verdict
        || (view.lifecycleOnly ? (['queued', 'running'].includes(view.state) ? view.state : 'review verdict unavailable') : view.state);
    const lifecycle = ['failed', 'error', 'interrupted', 'cancelled', 'timeout'].includes(text(view.lifecycleStatus).toLowerCase())
        ? ` · lifecycle ${text(view.lifecycleStatus).toLowerCase()}` : '';
    return { word: `${reviewStateWords(phrase)}${lifecycle}`, tone: view.tone };
}

function headMeta(view) {
    const attempts = view.attempts;
    const latest = attempts.at(-1);
    const when = formatMsgTime(text(latest?.timestamp))?.short || '';
    if (latest?.gap) return [GAP_WORDS[text(latest.gapReason)] || 'record unreadable', when].filter(Boolean).join(' · ');
    if (latest?.panel) return [when, view.progress ? '' : ledgerTally(latest.panel)].filter(Boolean).join(' · ');
    const asks = attempts.filter(attempt => attempt.participants);
    if (view.surface === 'plan' && asks.length) {
        const seats = asks.reduce((total, attempt) => total + list(attempt.participants).length, 0);
        return [when, plural(asks.length, 'ask'), plural(seats, 'participant'), 'nobody votes'].filter(Boolean).join(' · ');
    }
    return [when, view.countIsAuthoritative ? plural(view.attemptCount, 'attempt') : `${attempts.length} shown`].filter(Boolean).join(' · ');
}

function readButtons(sources, { owner, recordId, seatId, name, domain }) {
    const retained = list(sources).filter(source => source.status === 'retained' && text(source.sha256));
    const word = source => (source.role === 'failure' ? 'failure note' : 'answer');
    return retained.map((source) => {
        const twins = retained.filter(other => word(other) === word(source)).length > 1;
        const part = text(source.part).split(',').filter(Boolean).join(', ');
        const what = `${word(source)}${twins && part ? ` (${part})` : ''}`;
        const url = taskReviewSourceUrl(owner, recordId, seatId, { ...source, domain });
        return `<button type="button" class="review-read ui-hit-area" data-review-read-url="${escape(url)}" `
            + `data-review-read-name="${escape(`${name} · ${what}`)}" data-review-read-sha="${escape(source.sha256)}" `
            + `data-review-read-kind="${source.role === 'failure' ? 'text' : 'markdown'}" `
            + `aria-label="${escape(`Read ${name} ${what}`)}">${escape(retained.length === 1 ? 'Read' : `Read ${what}`)}${CHEVRON_RIGHT}</button>`;
    }).join('');
}

// `wrap` gives a line two lines before it is cut; a `whole` line is never cut, at any width.
function line({ sev = '', sevTone = 'neutral', by = '', words = '', wrap = false, whole = false, quiet = false }) {
    if (!words) return '';
    return `<p class="review-line${wrap ? ' review-line-wrap' : ''}${whole ? ' review-line-whole' : ''}${quiet ? ' review-line-quiet' : ''}">`
        + `${sev ? `<span class="review-sev" data-tone="${escape(sevTone)}">${escape(sev)}</span>` : ''}`
        + `${by ? `<span class="review-line-by">${escape(by)}: </span>` : ''}<span class="review-line-text">${escape(words)}</span></p>`;
}

function seatRow({ key, identity = '', name, note = '', role = '', state, said = '', meta = '', read = '' }) {
    const foot = said || meta || read;
    return `<div class="review-seat" data-review-opinion="${escape(key)}"><div class="review-seat-head">${identity}`
        + `<span class="review-seat-name">${escape(name)}</span>${note ? `<span class="review-seat-note">${escape(note)}</span>` : ''}`
        + `${role ? `<span class="review-seat-role">${escape(role)}</span>` : ''}${reviewStateMarkup(state.word, state.tone)}</div>`
        + `${foot ? `<div class="review-seat-line">${said}${meta ? `<span class="review-seat-meta">${escape(meta)}</span>` : ''}${read}</div>` : ''}</div>`;
}

// One reviewer's line: the critical finding first (or that it is only in the full answer), else the first
// note, else why there is no answer, in the engine's own words when it reported them.
function ledgerSaid(actor, state) {
    const findings = list(actor.findings);
    const critical = findings.find(finding => text(finding.severity).toLowerCase() === 'critical');
    const criticalTotal = Object.values(actor.answers && typeof actor.answers === 'object' ? actor.answers : {})
        .reduce((total, answer) => total + (Number(answer?.critical) || 0), 0);
    const findingWords = finding => [text(finding.item || finding.title || finding.summary), text(finding.reason)].filter(Boolean).join(': ');
    if (actorAnswered(actor) && criticalTotal && !critical) {
        return line({ sev: 'Critical', sevTone: 'error', wrap: true,
            words: `not among the ${plural(findings.length, 'finding')} the card keeps; it is in the full answer (${plural(state.findings, 'finding')}).` });
    }
    const first = critical || findings[0];
    if (first) return line({ sev: critical ? 'Critical' : 'Note', sevTone: critical ? 'error' : 'neutral', words: findingWords(first) || 'Finding without a title.' });
    if (actorAnswered(actor)) {
        return line({ quiet: true, words: text(actor.semantic_verdict).toUpperCase() === 'PASS' ? 'No finding reported.' : 'The answer is saved; it states no finding.' });
    }
    if (state.owed) return '';
    const cause = seatCauseText(actor);
    return cause ? line({ by: cause.by, words: cause.text, wrap: true }) : line({ quiet: true, words: 'No cause was recorded.' });
}

function ledgerRows(view, attempt, format) {
    const panel = attempt.panel;
    const live = attempt.live === true && text(panel.record_state) === 'pending';
    return list(panel.actors).map((actor) => {
        const state = seatStateView(actor, panel, { live });
        const observed = text(actor.observed_model || actor.model);
        const requested = text(actor.requested_model);
        const name = observed || requested || text(actor.slot_id) || 'Reviewer';
        const sources = list(actor.sources);
        const answerSaved = sources.some(source => source.role === 'response' && source.status === 'retained' && text(source.sha256));
        const meta = [state.tone === 'error' && state.findings > 1 ? plural(state.findings, 'finding') : '', seatLimitText(actor),
            actorAnswered(actor) && !answerSaved ? 'full text not saved' : ''].filter(Boolean).join(' · ');
        return seatRow({
            key: `${attempt.id}:${text(actor.slot_id)}`, identity: format.executions?.(actor, { showModel: false }) || '',
            name, note: observed ? '' : (requested ? 'requested · not reported' : 'model not reported'),
            role: text(panel.surface) === 'preflight' ? 'informational' : (seatAdviser(actor) ? 'adviser, not counted' : 'counted'),
            state, said: ledgerSaid(actor, state), meta,
            read: readButtons(sources, { owner: view.presentationOwnerTaskId, recordId: text(panel.record_id), seatId: text(actor.slot_id), name, domain: 'ledger' }),
        });
    }).join('') || '<p class="review-line review-line-quiet">No reviewer answer is saved in this record.</p>';
}

// A Plan participant's standing comes from the host's fold of objections (`objections` on the ask); nobody votes.
function participantState(participant, objections, live) {
    const status = text(participant.status);
    if (status === 'pending') return { word: 'awaiting', tone: live ? 'working' : 'neutral' };
    if (status === 'not_dispatched') return { word: 'not sent', tone: 'warn' };
    if (status !== 'answered') return { word: 'unavailable', tone: 'warn' };
    if (objections === null) return { word: 'answered · objections not projected', tone: 'neutral' };
    const mine = objections.filter(objection => objection.seat === text(participant.seat_id));
    const holding = mine.filter(objection => objection.holds).length;
    const withdrawn = Array.isArray(participant.reply?.withdrawn) ? participant.reply.withdrawn.length : 0;
    const tail = withdrawn ? ` · ${withdrawn} withdrawn` : '';
    if (holding) return { word: `${plural(holding, 'objection')} may hold${tail}`, tone: 'warn' };
    if (mine.length) return { word: `${plural(mine.length, 'objection')}, never holds${tail}`, tone: 'neutral' };
    return { word: `answered, no objection${tail}`, tone: 'ok' };
}

function planRows(view, attempt, format, live) {
    const objections = Array.isArray(attempt.objections) ? attempt.objections : null;
    const participants = list(attempt.participants);
    const answered = participants.filter(participant => participant.status === 'answered').length;
    const head = `<p class="review-ask-line">${escape([attempt.label, attempt.summary ? `«${attempt.summary}»` : '',
        `${answered} of ${participants.length} answered`].filter(Boolean).join(' · '))}</p>`;
    return head + participants.map((participant) => {
        const seatId = text(participant.seat_id);
        const state = participantState(participant, objections, live);
        const mine = (objections || []).filter(objection => objection.seat === seatId);
        const standing = mine.find(objection => objection.holds) || mine[0];
        const reply = participant.reply || {};
        const cause = text(participant.error || participant.parse_error || participant.not_dispatched_reason);
        const said = standing
            ? line({ sev: standing.holds ? 'May hold' : 'Never holds', sevTone: standing.holds ? 'warn' : 'neutral', words: standing.text })
            : (text(reply.prose) ? line({ words: text(reply.prose) }) : (cause ? line({ words: cause, wrap: true }) : ''));
        const ref = participant.reply_ref?.ref || participant.reply_ref;
        const name = text(participant.model) || seatId || 'Participant';
        return seatRow({
            key: `${attempt.id}:${seatId}`, identity: format.executions?.(participant, { showModel: false }) || '', name,
            role: participant.owner_marked ? 'counts' : 'added — never holds', state, said,
            read: readButtons(text(ref?.sha256) ? [{ role: 'response', part: '', status: 'retained', sha256: text(ref.sha256) }] : [],
                { owner: view.presentationOwnerTaskId, recordId: text(attempt.id).replace(/^ask:/, ''), seatId, name, domain: 'plan' }),
        });
    }).join('');
}

// Plan waves, task-acceptance panels and skill attempts stored before the ledger keep one row per attempt;
// their normalized saved detail is labelled as such, never as an original reviewer transcript.
function attemptRow(view, attempt, format) {
    const ref = attempt.detailRef;
    let read = '';
    let meta = '';
    if (ref?.surface === 'skill' && text(ref.skill) && text(ref.recordId || ref.jobId)) {
        // The task that ran the review owns its saved text; the server also accepts the presenting owner.
        const owner = text(attempt.initiatorTaskId) || text(view.initiatorTaskId) || view.presentationOwnerTaskId;
        const url = taskSkillReviewSourceUrl(owner, text(ref.recordId || ref.jobId), ref.skill);
        read = `<button type="button" class="review-read ui-hit-area" data-review-read-url="${escape(url)}" data-review-read-name="${escape(`${ref.skill} · saved skill review`)}" `
            + `aria-label="${escape(`Read the saved skill review of ${ref.skill}`)}">Read saved review${CHEVRON_RIGHT}</button>`;
    } else if (ref?.surface === 'task_acceptance') {
        if (text(ref.url)) {
            read = `<button type="button" class="review-read ui-hit-area" data-review-read-url="${escape(ref.url)}" data-review-read-name="Saved applied review" `
                + `aria-label="Read the saved applied review">Read saved applied review${CHEVRON_RIGHT}</button>`;
        } else if (!text(attempt.id).startsWith('acceptance-incident:')) meta = 'full applied review unavailable';
    }
    const initiator = attempt.initiatorTaskId && attempt.initiatorTaskId !== view.presentationOwnerTaskId && attempt.initiatorTaskId !== view.initiatorTaskId
        ? `Initiated by task ${attempt.initiatorTaskId}` : '';
    const attemptState = { word: reviewStateWords(attempt.progress || attempt.verdict || (attempt.lifecycleOnly
        ? (['queued', 'running'].includes(attempt.state) ? attempt.state : 'review verdict unavailable') : attempt.state)), tone: attempt.tone };
    return seatRow({
        key: `${view.id}:${attempt.id}`, identity: format.executions?.(attempt) || '', name: attempt.label,
        note: [formatMsgTime(text(attempt.timestamp))?.short || '', attempt.superseded ? 'superseded' : '', attempt.replayed ? 'replay' : '',
            attempt.revised ? 'revised snapshot' : ''].filter(Boolean).join(' · '),
        state: attemptState, said: line({ words: text(attempt.summary), wrap: true }) + (initiator ? line({ quiet: true, words: initiator }) : ''),
        meta, read,
    });
}

// Record facts: every saved line as a labelled list (label → value), the record's tokens and identifiers kept.
function factRows(block) {
    const rows = [];
    for (const raw of String(block || '').split('\n')) {
        if (!raw.trim()) continue;
        if (/^\s/.test(raw) && rows.length) { rows.at(-1)[1] += `\n${raw.trim()}`; continue; }
        const value = raw.trim();
        const cut = value.indexOf(': ');
        if (value === 'Cost unavailable') rows.push(['Cost', 'unavailable']);
        else if (cut > 0 && cut <= 60) rows.push([value.slice(0, cut), value.slice(cut + 2)]);
        else rows.push(['', value]);
    }
    return rows;
}

const describe = value => (value && typeof value === 'object'
    ? Object.entries(value).filter(([, item]) => item !== '' && item !== null && item !== undefined)
        .map(([key, item]) => `${key}=${typeof item === 'object' ? JSON.stringify(item) : String(item)}`).join(' · ')
    : text(value));

function ledgerFacts(attempt) {
    const panel = attempt.panel;
    const rows = [['Record', `${text(panel.record_id)}${panel.revision != null && panel.revision !== '' ? ` · revision ${panel.revision}` : ''}`],
        ['Wave', text(panel.review_wave_id)], ['Reviewed', [text(panel.goal), describe(panel.subject)].filter(Boolean).join(' · ')],
        ['Saved', formatMsgTime(text(panel.ts))?.full || text(panel.ts)], ['Verdict', [text(panel.aggregate_signal), text(panel.reason)].filter(Boolean).join(' · ')]];
    // What landed under this record, each with the decision that governed it and its whole reason (the opened
    // row's reason may be cut); older landings past the panel's bound stay in the record it names.
    for (const landing of list(panel.commit_landings)) {
        rows.push(['Committed', describe({ commit: landing.commit_sha, tree: landing.tree_sha, parent: landing.parent_sha,
            reviewed_subject: landing.reviewed_subject, author_subject: landing.author_subject,
            author_disposition: landing.author_disposition, recorded_at: landing.recorded_at })], ['Author reason', text(landing.author_reason)]);
    }
    const omitted = Number(panel.commit_landings_omitted) || 0;
    if (omitted > 0) rows.push(['Earlier landings', `${omitted} not shown · ${text(panel.source_ref?.path) || 'in the record'}`]);
    const decided = panel.author_decision && typeof panel.author_decision === 'object' ? panel.author_decision : null;
    if (decided) {
        rows.push(['Author decision', describe({ disposition: decided.disposition, reviewer_signal: decided.reviewer_signal,
            enforcement: decided.enforcement, subject: decided.subject_hash, recorded_at: decided.recorded_at })],
        ['Author reason', text(decided.rationale)]);
    }
    rows.push(...factRows(attempt.detailText));
    // A pool-era record prints each seat's cost in its reviewer line; an earlier one says it here.
    const pooled = text(panel.surface) !== 'preflight' && panelPoolRecorded(panel.panel_facts);
    for (const actor of list(panel.actors)) {
        const seat = text(actor.slot_id) || '?';
        rows.push([`Reviewer ${seat} model`, `requested ${text(actor.requested_model) || 'not recorded'} · observed ${text(actor.observed_model) || 'not reported'}`]);
        if (!pooled && actor.cost && typeof actor.cost === 'object') rows.push([`Reviewer ${seat} cost`, seatCostText(actor.cost)]);
        const failure = describe(Object.fromEntries(['failure_code', 'http_status', 'reset_at', 'operation_state', 'reported_cause']
            .map(key => [key, actor[key]]).filter(([, value]) => value !== undefined)));
        if (failure) rows.push([`Reviewer ${seat} failure`, failure]);
        if (actor.failure_details && typeof actor.failure_details === 'object') rows.push([`Reviewer ${seat} failure details`, describe(actor.failure_details)]);
        for (const source of list(actor.sources)) {
            rows.push([`Reviewer ${seat} saved ${source.role === 'failure' ? 'failure note' : 'answer'}`,
                describe({ part: source.part, status: source.status, sha256: source.sha256, size: source.size, chars: source.chars })]);
        }
    }
    return rows;
}

function participantFacts(attempt) {
    return list(attempt.participants).flatMap((participant) => {
        const seat = text(participant.seat_id) || '?';
        const ref = participant.reply_ref?.ref || participant.reply_ref;
        const executions = [...list(participant.executions), ...(participant.execution ? [participant.execution] : [])];
        const rows = [[`Participant ${seat}`, describe({ status: participant.status, model: participant.model, reply_sha256: ref?.sha256 })]];
        for (const objection of Array.isArray(participant.reply?.objections) ? participant.reply.objections : []) {
            rows.push([`Objection ${seat}/${text(objection?.id) || '?'}`, typeof objection === 'object' ? text(objection.text) : text(objection)]);
        }
        for (const withdrawn of Array.isArray(participant.reply?.withdrawn) ? participant.reply.withdrawn : []) {
            rows.push([`Withdrawn by ${seat}`, typeof withdrawn === 'object' ? describe(withdrawn) : text(withdrawn)]);
        }
        if (executions.length) rows.push([`Participant ${seat} executed`, executions.map(describe).join(' | ')]);
        return rows;
    });
}

function recordFacts(view) {
    const rows = [];
    for (const attempt of view.attempts) {
        if (attempt.panel) rows.push(...ledgerFacts(attempt));
        else rows.push(...factRows(attempt.detailText || attempt.summary));
        if (attempt.participants) rows.push(...participantFacts(attempt));
    }
    if (view.authorDecisionText) rows.push(...factRows(`Author decision: ${view.authorDecisionText}`));
    const seen = new Set();
    const shown = rows.filter(([label, value]) => {
        const key = `${label}\u0000${value}`;
        if (!text(value) || seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    const items = shown.map(([label, value]) => (label
        ? `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>` : `<dd class="review-facts-note">${escape(value)}</dd>`)).join('');
    return `<details class="review-facts" data-review-facts="${escape(view.id)}"><summary class="review-facts-summary">${CHEVRON_RIGHT}Record facts</summary>`
        + `<div class="review-facts-body" data-review-attempt-detail="${escape(view.id)}:facts" tabindex="0" role="group" aria-label="Record facts">`
        + `<dl class="review-facts-list">${items || '<dd class="review-facts-note">No saved detail.</dd>'}</dl></div></details>`;
}

function panelLine(view) {
    const panel = view.attempts.length === 1 ? view.attempts[0].panel : null;
    if (panel && text(panel.surface) !== 'preflight' && panelPoolRecorded(panel.panel_facts)) return panelPoolLines(panel)[0].replace(/^Panel: /, '');
    if (view.surface === 'plan' && view.attempts.some(attempt => attempt.participants)) {
        return 'Nobody votes on a plan: an objection stands until the participant who raised it withdraws it.';
    }
    return '';
}

function checkBody(view, format) {
    const attempt = view.attempts.length === 1 ? view.attempts[0] : null;
    if (attempt?.gap) {
        return `<div class="review-seats">${line({ words: attempt.summary, wrap: true })}</div><div class="review-check-foot">${recordFacts(view)}</div>`;
    }
    const live = text(view.state) === 'running';
    const rows = view.attempts.map((item) => {
        const key = `${view.id}:${item.id}`;
        let body;
        if (item.panel) body = ledgerRows(view, item, format);
        else if (item.participants) body = planRows(view, item, format, live);
        else if (view.surface === 'plan' && text(item.id).startsWith('revision:')) return '';
        else body = attemptRow(view, item, format);
        return `<div class="review-attempt" data-review-attempt="${escape(key)}">${body}</div>`;
    }).join('');
    const notes = [
        view.warning ? `<p class="review-warning ui-status" data-tone="warn">${escape(view.warning)}</p>` : '',
        view.initiatorTaskId && view.initiatorTaskId !== view.presentationOwnerTaskId ? line({ quiet: true, words: `Initiated by task ${view.initiatorTaskId}` }) : '',
        // The decision's own first clause (`Author finish: partial`) stays in view; its hashes and signals are facts.
        view.authorDecisionText ? `<p class="review-line review-line-quiet" data-review-author-decision><span class="review-line-text">${escape(view.authorDecisionText.split(' · ')[0])}</span></p>` : '',
    ].join('');
    const lead = panelLine(view);
    // The record's relation to the landed code leads the seats in body ink and is never cut; then the author's
    // own reason, quiet and attributed, at most two lines, Record facts keeping the whole reason.
    const landed = view.landingLine ? `<div class="review-landing" data-review-landing>${line({ words: view.landingLine, whole: true })}`
        + `${line({ by: 'Reason', words: view.authorReason, wrap: true, quiet: true })}</div>` : '';
    return `${notes}<div class="review-seats">${landed}${rows}</div><div class="review-check-foot">${lead ? `<span class="review-check-line">${escape(lead)}</span>` : ''}${recordFacts(view)}</div>`;
}

export function renderOpinionReviews(groups, disclosure, format = {}) {
    const views = reviewViews(groups);
    const active = views.some(view => ['queued', 'running'].includes(view.state));
    const errors = disclosure.hydrateStatus === 'error';
    const loading = disclosure.hydrateStatus === 'loading';
    if (!views.length && !errors && !disclosure.hadHydrateError) return '';
    const open = disclosure.sectionExpanded === true;
    const checks = views.map((view) => {
        const expanded = disclosure.expandedGroups?.has(view.id) === true;
        const state = headState(view);
        const meta = headMeta(view);
        const subject = [view.label, view.subject].filter(Boolean).join(' · ');
        return `<article class="chat-review-group review-check" data-review-group="${escape(view.id)}" data-open="${expanded ? '1' : '0'}">`
            + `<button type="button" class="review-check-head" data-review-group-toggle="${escape(view.id)}" aria-expanded="${expanded ? 'true' : 'false'}">`
            + `<span class="review-check-subject">${escape(subject)}</span>${reviewStateMarkup(state.word, state.tone)}${CHEVRON}`
            + `${meta ? `<span class="review-check-meta">${escape(meta)}</span>` : ''}</button>`
            + `<div class="chat-review-attempts review-check-body"${expanded ? '' : ' hidden'}>${checkBody(view, format)}</div></article>`;
    }).join('');
    const history = disclosure.historyTask && disclosure.historyMore
        ? `<div class="review-history"><span class="review-history-text">${disclosure.historyCertain === false
            ? 'Earlier reviews may be outside this view' : 'Earlier reviews outside this view'}</span>`
            + `<button type="button" class="review-read ui-hit-area" data-review-history${disclosure.historyLoading ? ' disabled' : ''}>${disclosure.historyLoading ? 'Loading…' : 'Read earlier reviews'}</button>`
            + `${disclosure.historyError ? '<span class="review-history-error" role="alert">Earlier reviews could not be read.</span>' : ''}</div>` : '';
    const status = loading
        ? '<div class="review-hydrate" data-review-hydrate-status role="status"><span class="review-hydrate-text">Loading review details…</span></div>'
        : errors ? '<div class="review-hydrate" data-review-hydrate-status role="alert"><span class="review-hydrate-text ui-status" data-tone="warn">Review details failed to refresh — shown data may be incomplete.</span>'
            + '<button type="button" class="review-read ui-hit-area" data-review-hydrate-retry>Retry read</button></div>' : '';
    return `<section class="chat-live-reviews" data-review-section data-expanded="${open ? '1' : '0'}"${errors || (loading && disclosure.hadHydrateError) ? ' data-review-hydrate-visible="1"' : ''}>`
        + `${status}<div class="chat-review-groups review-checks"${open ? '' : ' hidden'}>${checks}${history}`
        + `${active ? '<span class="sr-only">Review in progress</span>' : ''}</div></section>`;
}
