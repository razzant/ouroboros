// Review record -> card text: the panel and reviewer lines a task card's Reviews
// attempt and a Logs review event print. Pure; the owning surface places the text.
import { sinceLocalTime } from './utils.js';

const text = (value) => String(value ?? '').trim();
const count = (value) => {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.trunc(number) : 0;
};

// A reviewer answer that has not arrived is a gap, not a verdict: the host
// types a planned wait as `pending_dispatch` and an exceptional one otherwise.
export const actorAwaiting = (actor) => text(actor?.operation_state) === 'pending_dispatch' && text(actor?.transport_status) !== 'success';
export const actorUnresolved = (actor) => ['in_flight', 'custody_lost'].includes(text(actor?.operation_state));
// A ledger seat whose answer is still owed (its planned wait, an open operation, or the ledger's own
// `pending` status) is awaiting, never "not reached"; a seat with a readable answer was reached.
export const actorOwed = (actor) => actorAwaiting(actor) || actorUnresolved(actor) || text(actor?.transport_status) === 'pending';
export const actorAnswered = (actor) => ['valid', 'partial'].includes(text(actor?.parse_status));
// An adviser is heard, never counted: a pool-era record says so in `additional`, an earlier one in its role.
export const seatAdviser = (actor) => actor?.additional === true || /\badditional\b/.test(text(actor?.actor_role));
// A settled record's lost seat is terminal (its outcome unknown, never re-sent): not reached, never awaited.
export const seatOwed = (actor, panel) => actorOwed(actor)
    && !(text(panel?.record_state) === 'settled' && text(actor?.operation_state) === 'custody_lost');

// The owner's words for the review vocabulary (DESIGN §4 "Reviews inside task cards"): a card row says these,
// while Record facts and Logs keep the recorded token. A token outside the table is shown as recorded.
// An earlier-schema plan wave's aggregate (GREEN, REVISE_PLAN, REVIEW_REQUIRED) reads in the same table.
const VERDICT_WORDS = Object.freeze({
    PASS: 'passed', GREEN: 'passed', FAIL: 'rejected', QUORUM_FAILED: 'no verdict', NOT_PERFORMED: 'no verdict', DEGRADED: 'no verdict',
    NOT_DISPATCHED: 'not sent', UNKNOWN: 'verdict not recorded', rail_degraded: 'released at the task deadline',
    cycles_exhausted: 'review rounds used up', revise_plan: 'plan revision asked', review_required: 'review required',
    terminal: 'no verdict', running: 'in progress', unavailable: 'verdict unavailable',
});
const SEAT_VERDICT_WORDS = Object.freeze({ PASS: 'passed it', FAIL: 'rejected it', DEGRADED: 'inconclusive', UNKNOWN: 'outcome unknown' });

/** A row's state phrase in words: its leading verdict or lifecycle token says what it means, the rest stays. */
export function reviewStateWords(phrase) {
    const value = text(phrase);
    const [lead, ...rest] = value.split(' ');
    const word = VERDICT_WORDS[lead] || VERDICT_WORDS[lead.toUpperCase()] || VERDICT_WORDS[lead.toLowerCase()];
    return word ? [word, ...rest].join(' ') : value;
}

const plainCount = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

/** One seat's state in words, tone and wait: from its own answer, else whether it is owed, else how it ended. */
export function seatStateView(actor, panel, { live = false } = {}) {
    const verdict = text(actor?.semantic_verdict).toUpperCase();
    const findings = (Array.isArray(actor?.findings) ? actor.findings.length : 0) + count(actor?.findings_omitted);
    if (actorAnswered(actor) && (verdict === 'PASS' || verdict === 'FAIL')) {
        return { word: verdict === 'PASS' && findings ? `passed it · ${plainCount(findings, 'note')}` : SEAT_VERDICT_WORDS[verdict],
            tone: verdict === 'PASS' ? 'ok' : 'error', findings };
    }
    if (seatOwed(actor, panel)) {
        return { word: `awaiting${actorAwaiting(actor) || actorUnresolved(actor) ? sinceLocalTime(actor?.awaiting_since) : ''}`,
            tone: live ? 'working' : 'neutral', findings, owed: true };
    }
    if (text(actor?.operation_state) === 'custody_lost') return { word: 'outcome unknown', tone: 'neutral', findings };
    if ([actor?.transport_status, actor?.operation_state, verdict.toLowerCase()].map(text).includes('not_dispatched')) {
        return { word: 'not sent', tone: 'warn', findings };
    }
    if (SEAT_VERDICT_WORDS[verdict]) return { word: SEAT_VERDICT_WORDS[verdict], tone: verdict === 'UNKNOWN' ? 'neutral' : 'warn', findings };
    if (actorAnswered(actor)) return { word: 'answered without a readable verdict', tone: 'warn', findings };
    return { word: 'unavailable', tone: 'warn', findings };
}

/** The sentence a failed seat's cause reads in, the engine's own words first; a bare failure code stays in facts. */
export function seatCauseText(actor) {
    const reported = text(actor?.reported_cause);
    if (reported) return { by: 'Engine report', text: reported };
    const reason = text(actor?.reason).split('\n')[0];
    return reason && reason !== text(actor?.failure_code) ? { by: '', text: reason } : null;
}

/** A refusal's typed measurement when the engine recorded one (`limit`, `measured`, `units`), else ''. */
export function seatLimitText(actor) {
    const details = actor?.failure_details && typeof actor.failure_details === 'object' ? actor.failure_details : {};
    const number = (value) => (value !== null && value !== '' && Number.isFinite(Number(value)) ? Number(value) : null);
    const [limit, measured] = [number(details.limit), number(details.measured)];
    if (limit === null) return '';
    const units = text(details.units || details.unit).replaceAll('_', ' ');
    const amount = (value) => `${value.toLocaleString('en-US')}${units ? ` ${units}` : ''}`;
    return measured === null ? `limit ${amount(limit)}` : `${amount(measured)}, limit ${amount(limit)}`;
}

function compactCoverage(coverage) {
    if (!coverage || typeof coverage !== 'object') return '';
    return Object.entries(coverage)
        .filter(([, value]) => value !== '' && value !== null && value !== undefined)
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(', ');
}

/** A record written before the review pool named the whole pool `configured`. */
export function panelComposition(facts) {
    const value = text(facts?.composition);
    return value === 'configured' ? 'full_pool' : value;
}

/** Who sat on the panel and why (the record's `panel` facts); '' when it carries none. */
export function panelFactsText(facts) {
    if (!facts || typeof facts !== 'object') return '';
    const composition = panelComposition(facts);
    const parts = [
        composition === 'full_pool' ? 'whole review pool' : composition,
        text(facts.chosen_by) ? `chosen by ${text(facts.chosen_by)}` : '',
        ...[['seats', 'seats'], ['distinct_models', 'distinct models'], ['distinct_engines', 'distinct engines']]
            .filter(([key]) => facts[key] !== undefined && facts[key] !== null)
            .map(([key, label]) => `${label}=${String(facts[key])}`),
        // `seats` is the quorum's denominator; a critic added beside the pool is counted apart.
        count(facts.additional_seats) ? `additional seats=${count(facts.additional_seats)}` : '',
        count(facts.observed_unknown_seats) ? `model not observed on ${count(facts.observed_unknown_seats)} seat(s)` : '',
        facts.single_model_panel ? 'one model on every seat (repeated runs, not independent reviewers)' : '',
    ].filter(Boolean);
    const reason = text(facts.reason);
    if (reason) parts.push(`reason: ${reason}`);
    else if (facts.reason_missing === true) parts.push('reason missing');
    return parts.join(' · ');
}

// The owner's words for the pool facts (IMPL_SPEC §5: the Reviewer tick means whose agreement
// counts); a word outside these prints as recorded, never guessed.
const STRICTNESS_WORDS = { blocking: 'Blocking', advisory: 'Advisory', cyber_pro: 'Cyber Pro' };
const CHOOSER_WORDS = { ouroboros: 'Ouroboros', host: 'the host', owner: 'you' };
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const usdText = (value) => `$${value < 0.01 ? value.toFixed(4) : value.toFixed(2)}`;

/** Whether the record carries the review pool's facts; an older record prints exactly as before. */
export function panelPoolRecorded(facts) {
    return Boolean(facts && typeof facts === 'object' && (Array.isArray(facts.pool) || text(facts.strictness)));
}

/** One seat's cost: the measured amount, `subscription`, or `cost unknown` (a gap is never $0). */
export function seatCostText(cost) {
    const usd = Number(cost?.usd);
    if (cost?.kind === 'subscription') return 'subscription';
    return cost?.kind === 'usd' && cost.usd !== null && Number.isFinite(usd) ? usdText(usd) : 'cost unknown';
}

/** The wave total over its priced seats, naming the seats that total cannot include. */
export function waveTotalText(cost) {
    const seats = cost?.seats && typeof cost.seats === 'object' ? cost.seats : {};
    const [priced, subscription, unknown] = [count(seats.usd), count(seats.subscription), count(seats.unknown)];
    const total = Number(cost?.usd);
    const money = priced && cost.usd !== null && Number.isFinite(total)
        ? `${usdText(total)}${unknown ? ' for the priced seats' : ''}` : (unknown ? 'cost unknown' : '');
    const parts = [money, subscription ? `${plural(subscription, 'seat')} on a subscription` : '',
        priced && unknown ? `${plural(unknown, 'seat')} cost unknown` : ''].filter(Boolean);
    return parts.length ? `Wave total: ${parts.join(' · ')}` : '';
}

/** Who sat on a pool-era panel and why, in plain words: strictness, who chose, k of n ticked
 * reviewers, advisers (heard, never counted), the reason or the widening, the seats still
 * awaited, those never reached and why, the wave total and the models. Facts every seat
 * shares are said once here. */
export function panelPoolLines(panel) {
    const facts = panel?.panel_facts && typeof panel.panel_facts === 'object' ? panel.panel_facts : {};
    const ticked = Array.isArray(facts.pool) ? facts.pool.length : null;
    const [counted, advisers] = [count(facts.seats), count(facts.additional_seats)];
    const [strictness, chooser, reason] = [text(facts.strictness), text(facts.chosen_by), text(facts.reason)];
    const lines = [`Panel: ${[
        STRICTNESS_WORDS[strictness] || strictness,
        chooser ? `chosen by ${CHOOSER_WORDS[chooser] || chooser}` : '',
        ticked ? `${counted} of ${plural(ticked, 'ticked reviewer')}`
            : `${ticked === 0 ? 'no reviewer ticked · ' : ''}${plural(counted, 'counted seat')}`,
        advisers ? `${plural(advisers, 'adviser')}, not counted` : '',
    ].filter(Boolean).join(' · ')}`];
    const widened = (Array.isArray(facts.widened_from) ? facts.widened_from : []).map(text).filter(Boolean);
    const why = widened.length
        ? `widened to every ticked reviewer (Blocking) from the requested ${widened.join(', ')}${reason ? ` · reason given: ${reason}` : ''}`
        : reason || (facts.reason_missing === true ? 'no reason given' : '');
    if (why) lines.push(`Why these reviewers: ${why}`);
    const actors = (Array.isArray(panel?.actors) ? panel.actors : []).filter((actor) => actor && typeof actor === 'object');
    const owedHere = (actor) => seatOwed(actor, panel);
    const owed = actors.filter(owedHere).map((actor) => `${text(actor.slot_id) || '?'}${actor.additional ? ' (adviser)' : ''}`);
    if (owed.length) lines.push(`Awaiting an answer: ${owed.join(', ')}`);
    // The record's list is read against each seat's own status, which a late answer moves on.
    const reached = new Set(actors.filter((actor) => owedHere(actor) || actorAnswered(actor)).map((actor) => text(actor.slot_id)));
    for (const seat of Array.isArray(facts.not_reached) ? facts.not_reached : []) {
        const seatId = text(seat?.seat_id);
        if (seatId && !reached.has(seatId)) lines.push(`Not reached: ${seatId} (${text(seat.cause) || 'cause not recorded'})`);
    }
    const total = waveTotalText(panel?.cost);
    if (total) lines.push(total);
    const models = [
        facts.distinct_models != null ? plural(count(facts.distinct_models), 'distinct model') : '',
        facts.distinct_engines != null ? plural(count(facts.distinct_engines), 'distinct engine') : '',
        count(facts.observed_unknown_seats) ? `model not observed on ${plural(count(facts.observed_unknown_seats), 'seat')}` : '',
        facts.single_model_panel === true ? 'one model on every seat (repeated runs, not independent reviewers)' : '',
    ].filter(Boolean);
    if (models.length) lines.push(`Panel models: ${models.join(' · ')}`);
    return lines;
}

/** One reviewer's answer per brief part, in the order the seat was asked. */
export function seatAnswersText(seat) {
    const answers = seat?.answers && typeof seat.answers === 'object' ? seat.answers : null;
    if (!answers) return '';
    const order = Array.isArray(seat.parts) && seat.parts.length ? seat.parts : Object.keys(answers);
    return order.map((part) => {
        const answer = answers[part];
        if (!answer || typeof answer !== 'object') return `${part}=not recorded`;
        const counts = [`findings ${count(answer.findings)}`, `critical ${count(answer.critical)}`,
            text(answer.coverage) && text(answer.coverage) !== 'n/a' ? `coverage ${text(answer.coverage)}` : ''].filter(Boolean);
        return `${part}=${[text(answer.status) || 'unknown', text(answer.verdict)].filter(Boolean).join(' ')} (${counts.join(', ')})`;
    }).join(' · ');
}

// How THIS ledger record relates to the code that landed under it (owner 29.10, A: one sentence inside the
// opened row). The gate types each landed commit against the record's reviewed tree (`commit_landings`); a
// different tree says only that this review does not cover the code, never that nobody reviewed it. `Adopted`
// needs the task's recorded adoption of that commit, exactly or within its adopted range; `Committed` says only
// that it landed (an adoption the task could not record leaves it so), and a range whose list was cut says it
// cannot tell. Each landing speaks with the decision that governed it, never a later one on the same record.
// No landing (an older record, a failed commit) says nothing.
export function landingWords(panel, adoptions) {
    const none = { line: '', reason: '', earlier: false };
    const landings = (Array.isArray(panel?.commit_landings) ? panel.commit_landings : []).filter((row) => text(row?.commit_sha));
    if (!landings.length) return none;
    const settled = (Array.isArray(adoptions) ? adoptions : []).filter((row) => text(row?.outcome) === 'adopted' && text(row?.commit_sha));
    const range = (row) => (Array.isArray(row.commits) ? row.commits.map(text) : []);
    const adoptedBy = (sha) => settled.find((row) => text(row.commit_sha) === sha) || settled.find((row) => range(row).includes(sha));
    const landing = [...landings].reverse().find((row) => adoptedBy(text(row.commit_sha))) || landings.at(-1);
    const sha = text(landing.commit_sha).slice(0, 12);
    const adoption = adoptedBy(text(landing.commit_sha));
    const head = text(adoption?.commit_sha).slice(0, 12);
    const within = Boolean(adoption) && text(adoption.commit_sha) !== text(landing.commit_sha);
    const open = adoption ? null : settled.find((row) => row.commits_complete !== true);  // its list may lack this commit
    const lead = !adoption ? `Committed ${sha}` : within ? `Adopted ${sha} as part of ${head}` : `Adopted ${sha}`;
    const own = text(landing.author_subject) ? ", by Ouroboros's own decision" : '';
    const verdict = text(panel.aggregate_signal).toUpperCase();
    const relation = text(landing.reviewed_subject);
    const covered = !adoption ? `committed ${sha}` : within ? `${sha}, adopted as part of ${head}` : `adopted ${sha}`;
    const line = relation === 'different' ? `This review was for earlier code; it does not cover ${covered}.`
        : relation !== 'same' ? `${lead}; whether this review saw these bytes is not recorded.`
            : verdict === 'PASS' ? `${lead}: the exact bytes this review passed.`
                : verdict === 'FAIL' ? `${lead} despite this rejection${own}.`
                    : `${lead}${within ? ',' : ''} with no verdict from this review${own}.`;
    const unsure = open ? ` Whether the adoption of ${text(open.commit_sha).slice(0, 12)} included it is not recorded.` : '';
    return { line: line + unsure, reason: text(landing.author_reason), earlier: relation === 'different' };
}

export function authorDoctrineLines(decision) {
    if (!decision || typeof decision !== 'object') return [];
    const fact = decision.review_replaced_by_subagents;
    const lines = [];
    if (fact && typeof fact === 'object') {
        const reports = Array.isArray(fact.reports) ? fact.reports : [];
        const bound = reports.filter((row) => row && row.status === 'bound').length;
        const gaps = reports.filter((row) => row && row.status === 'gap').length;
        lines.push(`Check replaced by subagents: ${String(fact.reason || '')}`);
        lines.push('Subagent reports are author evidence, not review seats, and do not create PASS.');
        if (fact.selection_rationale) lines.push(`Selection: ${String(fact.selection_rationale)}`);
        if (fact.subject_hash) lines.push(`Checked subject: ${String(fact.subject_hash)}`);
        lines.push(`Subagent reports: ${bound} bound, ${gaps} evidence gap(s).`);
        for (const report of reports) {
            if (report?.actor_identity === 'mismatch') {
                lines.push(`Actor identity mismatch: claimed ${report.ref?.subagent_id || 'unknown'}; recorded ${report.actor?.subagent_id || 'unknown'}.`);
            } else if (report?.actor_identity === 'unknown') {
                lines.push('Report actor identity: unknown (no retained host identity).');
            }
        }
    }
    const snapshot = decision.evidence_at_decision;
    if (snapshot && typeof snapshot === 'object' && snapshot.feedback_at_decision) {
        lines.push(`Evidence at decision: ${String(snapshot.feedback_at_decision)}`);
    }
    if (decision.landing && decision.landing.landed === true) {
        lines.push('Landing: this commit, recorded separately from the evidence at decision.');
    }
    return lines;
}

export function formatReviewProjection(projection) {
    const panels = Array.isArray(projection?.panels) ? projection.panels : [];
    const lines = [];
    panels.forEach((panel, panelIndex) => {
        if (!panel || typeof panel !== 'object') return;
        const quorum = panel.quorum && typeof panel.quorum === 'object' ? panel.quorum : {};
        const panelId = String(panel.panel_id || `panel-${panelIndex + 1}`);
        const awaiting = (Array.isArray(panel.actors) ? panel.actors : []).filter(actorAwaiting).length;
        const signal = String(panel.aggregate_signal || 'UNKNOWN');
        // While a slot is awaited the aggregate is not final; DEGRADED is only the host's placeholder.
        const verdictText = !awaiting ? signal : (signal === 'DEGRADED' ? `none (${awaiting} awaiting; held as DEGRADED)` : `${signal} (${awaiting} awaiting)`);
        lines.push(
            `Review panel ${panelId}: ${String(panel.surface || 'review')} · authority=${String(panel.authority || 'unspecified')} · verdict=${verdictText} · transport=${String(panel.transport_status || 'unknown')} · parse=${String(panel.parse_status || 'unknown')} · quorum=${String(quorum.contributed ?? 0)}/${String(quorum.configured ?? 0)} (required ${String(quorum.required ?? 0)}) · enforcement=${String(panel.enforcement_impact || 'unknown')}${panel.single_reviewer_no_diversity ? ' · single-reviewer (no diversity)' : ''}${panel.dialogue && panel.dialogue.status ? ` · dialogue=${String(panel.dialogue.status)}` : ''}${panel.superseded ? ' · superseded' : ''}`,
        );
        if (panel.reason) lines.push(`Panel reason: ${String(panel.reason)}`);
        // A preflight seats one named row, ticked or not, and owes no reason: never a pool panel.
        const pooled = text(panel.surface) !== 'preflight' && panelPoolRecorded(panel.panel_facts);
        if (pooled) lines.push(...panelPoolLines(panel));
        const facts = pooled ? '' : panelFactsText(panel.panel_facts);
        if (facts) lines.push(`Panel composition: ${facts}`);
        const coverage = compactCoverage(panel.coverage);
        if (coverage) lines.push(`Panel coverage: ${coverage}`);
        const binding = [
            panel.candidate_hash ? `candidate_hash=${String(panel.candidate_hash)}` : '',
            panel.evidence_revision ? `evidence_revision=${String(panel.evidence_revision)}` : '',
            panel.fence_hash ? `fence_hash=${String(panel.fence_hash)}` : '',
            panel.binding_hash ? `binding_hash=${String(panel.binding_hash)}` : '',
        ].filter(Boolean);
        if (binding.length) lines.push(`Panel binding: ${binding.join(' · ')}`);
        (Array.isArray(panel.actors) ? panel.actors : []).forEach((actor) => {
            if (!actor || typeof actor !== 'object') return;
            const slotId = String(actor.slot_id || '?');
            // A pool-era seat leads with the two facts that differ seat by seat: counted or not, and its cost.
            const seat = pooled ? `${actor.additional ? 'adviser, not counted' : 'counted'} · ${seatCostText(actor.cost)} · ` : '';
            // A model nobody reported keeps the requested one beside it, labelled as a request.
            const model = text(actor.model) || (text(actor.requested_model) ? `not reported · requested=${text(actor.requested_model)}` : 'unknown');
            lines.push(
                `Reviewer ${slotId}: ${seat}role=${String(actor.actor_role || 'reviewer')} · provider=${String(actor.provider || 'unknown')} · model=${model} · transport=${actorAwaiting(actor) ? 'awaiting' : String(actor.transport_status || 'unknown')} · parse=${actorAwaiting(actor) ? 'awaiting' : String(actor.parse_status || 'unknown')} · verdict=${String(actor.semantic_verdict || 'none')}${actor.outcome_tier ? ` · outcome_tier=${String(actor.outcome_tier)}` : ''}${actor.dialogue_status ? ` · dialogue=${String(actor.dialogue_status)}` : ''} · quorum=${actor.quorum_contribution ? 'contributes' : 'abstains'} · enforcement=${String(actor.enforcement_impact || 'unknown')}${actorAwaiting(actor) || actorUnresolved(actor) ? sinceLocalTime(actor.awaiting_since) : ''}`,
            );
            const actorCoverage = compactCoverage(actor.coverage);
            if (actorCoverage) lines.push(`Reviewer ${slotId} coverage: ${actorCoverage}`);
            const answers = seatAnswersText(actor);
            if (answers) lines.push(`Reviewer ${slotId} answers: ${answers}`);
            if (actor.reason) lines.push(`Reviewer ${slotId} reason: ${String(actor.reason)}`);
            if (Array.isArray(actor.findings)) {
                for (const finding of actor.findings) {
                    if (!finding || typeof finding !== 'object') continue;
                    const label = [text(finding.severity), text(finding.verdict)]
                        .filter(Boolean).join(' ') || 'finding';
                    const title = text(finding.item) || text(finding.summary) || '(no item)';
                    const summaryText = text(finding.summary);
                    const body = [
                        `[${label}]${text(finding.id) ? ` ${text(finding.id)}` : ''} ${title}`,
                        summaryText && summaryText !== title ? `summary: ${summaryText}` : '',
                        text(finding.reason) ? `reason: ${text(finding.reason)}` : '',
                        text(finding.evidence) ? `evidence: ${text(finding.evidence)}` : '',
                        text(finding.recommendation) ? `fix: ${text(finding.recommendation)}` : '',
                    ].filter(Boolean).join(' — ');
                    lines.push(`Reviewer ${slotId} finding: ${body}`);
                }
                const omitted = count(actor.findings_omitted);
                if (omitted) lines.push(`Reviewer ${slotId} findings omitted: ${omitted}`);
            }
            // P1: name the durable full copy unconditionally — bounded rows,
            // per-string truncation markers and pre-findings-era projections
            // all resolve through the same observability call.
            const callId = text(actor.response_ref?.call_id);
            if (callId) {
                lines.push(`Reviewer ${slotId} full response: observability call ${callId}`);
            }
        });
        lines.push(...authorDoctrineLines(panel.author_decision));
    });
    // A record that could not be read stays a line of its own; its readable neighbours are listed above.
    for (const gap of Array.isArray(projection?.gaps) ? projection.gaps : []) {
        if (gap && typeof gap === 'object') lines.push(`Review record unavailable (${text(gap.review_gap) || 'unreadable'})${text(gap.record_id) ? `: ${text(gap.record_id)}` : ''}`);
    }
    return lines.join('\n');
}
