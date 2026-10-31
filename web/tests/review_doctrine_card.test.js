import test from 'node:test';
import assert from 'node:assert/strict';

import { authorDoctrineLines, formatReviewProjection } from '../modules/review_record_card.js';
import { planReviewGroupFromTaskDetail, renderReviewsSection } from '../modules/review_presentation.js';

const panel = {
    panel_id: 'p1', surface: 'commit_gate', authority: 'review_ledger', aggregate_signal: 'FAIL',
    transport_status: 'success', parse_status: 'valid',
    quorum: { required: 1, contributed: 1, configured: 1 }, enforcement_impact: 'advisory', actors: [],
};

test('a card without the fact does not say the check was replaced', () => {
    const text = formatReviewProjection({ panels: [panel] });
    assert.equal(authorDoctrineLines(undefined).length, 0);
    assert.equal(authorDoctrineLines({ disposition: 'accepted', rationale: 'ordinary' }).length, 0);
    assert.doesNotMatch(text, /replaced|Evidence at decision|Landing:/);
});

test('the card shows a claimed helper check, its gaps, and a separate landing', () => {
    const decision = {
        disposition: 'accepted',
        rationale: 'I checked.',
        review_replaced_by_subagents: {
            reason: 'The panel returned nothing.',
            selection_rationale: 'This helper reads the repository.',
            subject_hash: 'abc',
            reports: [{ status: 'bound' }, { status: 'gap', gap: 'report_unreadable' }],
        },
        evidence_at_decision: { feedback_at_decision: 'none_received' },
        landing: { landed: true },
    };
    const text = formatReviewProjection({ panels: [{ ...panel, author_decision: decision }] });
    assert.match(text, /Check replaced by subagents: The panel returned nothing\./);
    assert.match(text, /not review seats, and do not create PASS/);
    assert.match(text, /Subagent reports: 1 bound, 1 evidence gap/);
    assert.match(text, /Evidence at decision: none_received/);
    assert.match(text, /Landing: this commit, recorded separately/);
    assert.doesNotMatch(text, /verdict=PASS/);
});

test('plan history shows the fact only on the decision that claimed it', () => {
    const fingerprint = 'a'.repeat(64);
    const group = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'closed' },
            waves: [{
                request_fingerprint: fingerprint,
                aggregate: 'REVIEW_REQUIRED',
                closed: true,
                author_disposition: {
                    disposition: 'partial',
                    rationale: 'Fixed the defect.',
                    reviewer_signal: 'REVIEW_REQUIRED',
                    subject_hash: fingerprint,
                    review_replaced_by_subagents: {
                        reason: 'I used a helper.',
                        reports: [],
                    },
                },
            }],
        },
    });
    const html = renderReviewsSection([group], {
        sectionExpanded: true,
        expandedGroups: new Set([group.id]),
        expandedAttempts: new Set([`${group.id}:${group.attempts[0].id}`]),
    });
    assert.match(html, /<dt>Author finish<\/dt><dd>partial · /);
    assert.match(html, /Check replaced by subagents: I used a helper\./);
    assert.match(html, /do not create PASS/);
});

test('report identity shows a conflicting claim beside the recorded actor and keeps unknowns', () => {
    const render = (reports) => authorDoctrineLines({ review_replaced_by_subagents: { reports } }).join('\n');
    const mismatch = render([{ status: 'bound', actor_identity: 'mismatch',
        ref: { subagent_id: 'claimed' }, actor: { subagent_id: 'actual' } }]);
    assert.match(mismatch, /Actor identity mismatch: claimed claimed; recorded actual/);
    assert.match(render([{ status: 'gap', actor_identity: 'unknown', ref: { subagent_id: 'claimed' } }]),
        /Report actor identity: unknown/);
    assert.doesNotMatch(render([{ status: 'bound', actor_identity: 'recorded' }]), /mismatch|identity: unknown/);
});
