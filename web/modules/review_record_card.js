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
        const facts = panelFactsText(panel.panel_facts);
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
            lines.push(
                `Reviewer ${slotId}: role=${String(actor.actor_role || 'reviewer')} · provider=${String(actor.provider || 'unknown')} · model=${String(actor.model || 'unknown')} · transport=${actorAwaiting(actor) ? 'awaiting' : String(actor.transport_status || 'unknown')} · parse=${actorAwaiting(actor) ? 'awaiting' : String(actor.parse_status || 'unknown')} · verdict=${String(actor.semantic_verdict || 'none')}${actor.outcome_tier ? ` · outcome_tier=${String(actor.outcome_tier)}` : ''}${actor.dialogue_status ? ` · dialogue=${String(actor.dialogue_status)}` : ''} · quorum=${actor.quorum_contribution ? 'contributes' : 'abstains'} · enforcement=${String(actor.enforcement_impact || 'unknown')}${actorAwaiting(actor) || actorUnresolved(actor) ? sinceLocalTime(actor.awaiting_since) : ''}`,
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
    });
    return lines.join('\n');
}
