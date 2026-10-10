import test from 'node:test';
import assert from 'node:assert/strict';

import {
    actorAwaiting,
    actorUnresolved,
    formatReviewProjection,
    panelComposition,
    panelFactsText,
    seatAnswersText,
} from '../modules/review_record_card.js';
import { formatReviewProjection as presentationFormatter } from '../modules/review_presentation.js';

const POOL_FACTS = {
    seats: 3, distinct_models: 3, observed_unknown_seats: 0, distinct_engines: 3, single_model_panel: false,
    composition: 'full_pool', reason: '', reason_missing: false, chosen_by: 'owner',
};

function panel(fields = {}) {
    return {
        panel_id: 'p1', surface: 'plan_review', authority: 'blocking', aggregate_signal: 'PASS',
        transport_status: 'success', parse_status: 'valid',
        quorum: { required: 2, contributed: 3, configured: 3 }, enforcement_impact: 'none',
        actors: [], ...fields,
    };
}

test('the record card is the one formatter: the presentation module re-exports it', () => {
    assert.equal(presentationFormatter, formatReviewProjection);
});

test('a PR-1 record calling the whole pool `configured` reads as the whole pool', () => {
    assert.equal(panelComposition({ composition: 'configured' }), 'full_pool');
    assert.equal(panelComposition({ composition: 'full_pool' }), 'full_pool');
    assert.equal(panelComposition({ composition: 'composed' }), 'composed');
    assert.equal(panelComposition(null), '');
    assert.equal(panelFactsText({ ...POOL_FACTS, composition: 'configured' }), panelFactsText(POOL_FACTS));
});

test('panel facts name the pool, who chose it and how many distinct models sat on it', () => {
    assert.equal(panelFactsText(POOL_FACTS),
        'whole review pool · chosen by owner · seats=3 · distinct models=3 · distinct engines=3');
    assert.equal(panelFactsText(null), '');
    assert.equal(panelFactsText({}), '');
});

test('w3: seats is the assigned count and an added critic is said apart, never folded in', () => {
    const withCritic = { ...POOL_FACTS, seats: 3, additional_seats: 1 };
    assert.equal(panelFactsText(withCritic),
        'whole review pool · chosen by owner · seats=3 · distinct models=3 · distinct engines=3 · additional seats=1');
    assert.doesNotMatch(panelFactsText({ ...POOL_FACTS, additional_seats: 0 }), /additional/);
    assert.equal(panelFactsText({ ...POOL_FACTS, additional_seats: 0 }), panelFactsText(POOL_FACTS));
    // The block review_change writes for one assigned seat beside one added critic (NEW-W3):
    // the models are everyone's, the seat count is the assigned seat alone.
    const oneAndCritic = { ...POOL_FACTS, seats: 1, additional_seats: 1, distinct_models: 2, distinct_engines: 2 };
    assert.equal(panelFactsText(oneAndCritic),
        'whole review pool · chosen by owner · seats=1 · distinct models=2 · distinct engines=2 · additional seats=1');
    assert.doesNotMatch(panelFactsText(oneAndCritic), /repeated runs/);
});

test('three runs of one model are said to be repeats, not independent reviewers', () => {
    const text = panelFactsText({ ...POOL_FACTS, distinct_models: 1, distinct_engines: 1, single_model_panel: true });
    assert.match(text, /distinct models=1/);
    assert.match(text, /one model on every seat \(repeated runs, not independent reviewers\)/);
    assert.doesNotMatch(panelFactsText(POOL_FACTS), /repeated runs/);
});

test('a composed panel shows its reason, or says the reason is missing', () => {
    const composed = { ...POOL_FACTS, composition: 'composed', chosen_by: 'author', seats: 1 };
    assert.match(panelFactsText({ ...composed, reason: 'Touches the gateway only.' }),
        /^composed · chosen by author · seats=1 .* · reason: Touches the gateway only\.$/);
    assert.match(panelFactsText({ ...composed, reason_missing: true }), / · reason missing$/);
    assert.doesNotMatch(panelFactsText(POOL_FACTS), /reason/);
});

test('an unobserved seat model is counted, never folded into the distinct count', () => {
    assert.match(panelFactsText({ ...POOL_FACTS, observed_unknown_seats: 2 }), /model not observed on 2 seat\(s\)/);
});

test('answers follow the seat parts; coverage is shown only where it means something', () => {
    const seat = {
        parts: ['change', 'coupling'],
        answers: {
            coupling: { status: 'answered', verdict: 'FAIL', findings: 1, critical: 1, coverage: 'full' },
            change: { status: 'answered', verdict: 'PASS', findings: 0, critical: 0, coverage: 'n/a' },
        },
    };
    assert.equal(seatAnswersText(seat),
        'change=answered PASS (findings 0, critical 0) · coupling=answered FAIL (findings 1, critical 1, coverage full)');
    assert.equal(seatAnswersText({ parts: ['change', 'coupling'], answers: { change: seat.answers.change } }),
        'change=answered PASS (findings 0, critical 0) · coupling=not recorded');
    assert.equal(seatAnswersText({ answers: { change: { status: 'unanswered', verdict: '' } } }),
        'change=unanswered (findings 0, critical 0)');
    assert.equal(seatAnswersText({}), '');
});

test('the card prints composition and per-part answers only when the record carries them', () => {
    const actor = {
        slot_id: 'review-1', provider: 'openai', model: 'openai/gpt-5.6-sol', transport_status: 'success',
        parse_status: 'valid', semantic_verdict: 'PASS', quorum_contribution: true, enforcement_impact: 'none',
        parts: ['change'], answers: { change: { status: 'answered', verdict: 'PASS', findings: 0, critical: 0, coverage: 'n/a' } },
    };
    const text = formatReviewProjection({ panels: [panel({ panel_facts: POOL_FACTS, actors: [actor] })] });
    assert.match(text, /^Panel composition: whole review pool · chosen by owner · seats=3/m);
    assert.match(text, /^Reviewer review-1 answers: change=answered PASS \(findings 0, critical 0\)$/m);

    const bare = formatReviewProjection({ panels: [panel({ actors: [{ ...actor, parts: undefined, answers: undefined }] })] });
    assert.doesNotMatch(bare, /Panel composition|answers:/);
    assert.match(bare, /^Review panel p1: plan_review · authority=blocking · verdict=PASS/m);
});

test('an awaited or unresolved reviewer is a gap, not a verdict', () => {
    assert.equal(actorAwaiting({ operation_state: 'pending_dispatch' }), true);
    assert.equal(actorAwaiting({ operation_state: 'pending_dispatch', transport_status: 'success' }), false);
    assert.equal(actorUnresolved({ operation_state: 'in_flight' }), true);
    assert.equal(actorUnresolved({ operation_state: 'custody_lost' }), true);
    assert.equal(actorUnresolved({ operation_state: 'settled' }), false);
    const text = formatReviewProjection({ panels: [panel({ aggregate_signal: 'DEGRADED',
        actors: [{ slot_id: 'review-2', operation_state: 'pending_dispatch' }] })] });
    assert.match(text, /verdict=none \(1 awaiting; held as DEGRADED\)/);
    assert.match(text, /Reviewer review-2: .* transport=awaiting · parse=awaiting/);
});
