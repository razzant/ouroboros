import test from 'node:test';
import assert from 'node:assert/strict';

import {
    actorAwaiting,
    actorOwed,
    actorUnresolved,
    formatReviewProjection,
    panelComposition,
    panelFactsText,
    panelPoolLines,
    panelPoolRecorded,
    seatAnswersText,
    seatCostText,
    waveTotalText,
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

// The review pool's facts (IMPL_SPEC §4.6) in the shape review_projection.ledger_record_panel
// gives a record that carries them: the panel block keeps its own keys, each actor gains
// `additional` and `cost`, the panel its wave `cost`.
const ADVISORY = {
    ...POOL_FACTS, composition: 'composed', chosen_by: 'ouroboros', seats: 2, additional_seats: 1,
    strictness: 'advisory', pool: ['sol', 'opus', 'grok'], reason: 'Touches the gateway only.',
};

function seat(slotId, fields = {}) {
    return {
        slot_id: slotId, provider: 'openai', model: `model-${slotId}`, transport_status: 'success', parse_status: 'valid',
        semantic_verdict: 'PASS', quorum_contribution: true, enforcement_impact: 'advisory', additional: false,
        cost: { kind: 'usd', usd: 0.12 }, ...fields,
    };
}

const poolPanel = (facts, fields = {}) => panel({
    panel_facts: facts, cost: { usd: 0.12, seats: { usd: 1, subscription: 1, unknown: 1 } },
    actors: [seat('sol'), seat('opus', { cost: { kind: 'subscription', usd: null } }),
        seat('critic', { additional: true, quorum_contribution: false, cost: { kind: 'unknown', usd: null } })],
    ...fields,
});

test('an older record without the pool facts prints exactly as before, even beside a cost', () => {
    const old = { ...POOL_FACTS, composition: 'composed', chosen_by: 'author', seats: 1, additional_seats: 1, reason: 'Touches the gateway only.' };
    const actor = {
        slot_id: 'review-1', provider: 'openai', model: 'openai/gpt-5.6-sol', transport_status: 'success',
        parse_status: 'valid', semantic_verdict: 'PASS', quorum_contribution: true, enforcement_impact: 'none',
        parts: ['change'], answers: { change: { status: 'answered', verdict: 'PASS', findings: 0, critical: 0, coverage: 'n/a' } },
    };
    // The text the card printed before the pool facts existed (frozen from the base module).
    const before = [
        'Review panel p1: plan_review · authority=blocking · verdict=PASS · transport=success · parse=valid · quorum=3/3 (required 2) · enforcement=none',
        'Panel composition: composed · chosen by author · seats=1 · distinct models=3 · distinct engines=3 · additional seats=1 · reason: Touches the gateway only.',
        'Reviewer review-1: role=reviewer · provider=openai · model=openai/gpt-5.6-sol · transport=success · parse=valid · verdict=PASS · quorum=contributes · enforcement=none',
        'Reviewer review-1 answers: change=answered PASS (findings 0, critical 0)',
    ].join('\n');
    assert.equal(panelPoolRecorded(old), false);
    assert.equal(formatReviewProjection({ panels: [panel({ panel_facts: old, actors: [actor] })] }), before);
    // Only the record's own pool facts switch the plain lines on; a cost alone does not.
    const costed = formatReviewProjection({ panels: [panel({ panel_facts: old, cost: { usd: 1, seats: { usd: 1 } },
        actors: [{ ...actor, cost: { kind: 'usd', usd: 1 }, additional: false }] })] });
    assert.equal(costed, before);
    assert.doesNotMatch(costed, /^Panel: |Wave total|ticked|counted/m);
});

test('a pool-era panel says strictness, who chose, k of n ticked reviewers and the advisers in words', () => {
    assert.equal(panelPoolRecorded(ADVISORY), true);
    assert.equal(panelPoolRecorded({ ...POOL_FACTS, strictness: 'blocking' }), true);
    assert.equal(panelPoolRecorded({ ...POOL_FACTS, pool: [] }), true);
    const lines = formatReviewProjection({ panels: [poolPanel(ADVISORY)] }).split('\n');
    assert.ok(lines.includes('Panel: Advisory · chosen by Ouroboros · 2 of 3 ticked reviewers · 1 adviser, not counted'), lines.join('\n'));
    assert.ok(lines.includes('Why these reviewers: Touches the gateway only.'));
    // The technical composition line gives way to the words; its model facts stay, in words too.
    assert.ok(!lines.some((line) => line.startsWith('Panel composition:')));
    assert.ok(lines.includes('Panel models: 3 distinct models · 3 distinct engines'));
    assert.ok(!lines.some((line) => /chosen by ouroboros|seats=|reason: /.test(line)));
});

test('who chose reads Ouroboros, the host or you, and strictness reads as the owner names it', () => {
    const head = (facts) => panelPoolLines({ panel_facts: facts })[0];
    assert.match(head({ ...ADVISORY, chosen_by: 'host' }), /^Panel: Advisory · chosen by the host · /);
    assert.match(head({ ...ADVISORY, chosen_by: 'owner', strictness: 'blocking' }), /^Panel: Blocking · chosen by you · /);
    assert.match(head({ ...ADVISORY, strictness: 'cyber_pro' }), /^Panel: Cyber Pro · chosen by Ouroboros · /);
    // A word outside the known ones prints as the record wrote it, never a guess.
    assert.match(head({ ...ADVISORY, chosen_by: 'author', strictness: 'lenient' }), /^Panel: lenient · chosen by author · /);
    assert.equal(head({ ...ADVISORY, chosen_by: '', strictness: '', additional_seats: 0 }), 'Panel: 2 of 3 ticked reviewers');
});

test('Blocking widens a named subset and says so; a narrowed panel says its reason or that none was given', () => {
    const why = (facts) => panelPoolLines({ panel_facts: facts }).find((line) => line.startsWith('Why these reviewers: ')) || '';
    const blocking = { ...ADVISORY, strictness: 'blocking', chosen_by: 'owner', seats: 3, additional_seats: 0, reason: '' };
    assert.equal(why({ ...blocking, widened_from: ['sol'] }),
        'Why these reviewers: widened to every ticked reviewer (Blocking) from the requested sol');
    assert.equal(why({ ...blocking, widened_from: ['sol', 'opus'], reason: 'gateway only' }),
        'Why these reviewers: widened to every ticked reviewer (Blocking) from the requested sol, opus · reason given: gateway only');
    assert.equal(why({ ...ADVISORY, reason: '', reason_missing: true }), 'Why these reviewers: no reason given');
    // The whole pool by the owner's ticks owes no reason and prints none.
    assert.equal(why(blocking), '');
    assert.equal(why({ ...blocking, widened_from: [] }), '');
});

test('each seat leads with counted or adviser and its own cost: an amount, subscription or cost unknown', () => {
    const lines = formatReviewProjection({ panels: [poolPanel(ADVISORY)] }).split('\n');
    assert.ok(lines.some((line) => line.startsWith('Reviewer sol: counted · $0.12 · role=')), lines.join('\n'));
    assert.ok(lines.some((line) => line.startsWith('Reviewer opus: counted · subscription · role=')));
    assert.ok(lines.some((line) => line.startsWith('Reviewer critic: adviser, not counted · cost unknown · role=')));
    assert.equal(seatCostText({ kind: 'usd', usd: 0.0042 }), '$0.0042');
    assert.equal(seatCostText({ kind: 'usd', usd: 1.5 }), '$1.50');
    // A gap is never a receipt: a missing amount is unknown, not $0.
    assert.equal(seatCostText({ kind: 'usd', usd: null }), 'cost unknown');
    assert.equal(seatCostText(undefined), 'cost unknown');
    assert.equal(seatCostText({ kind: 'subscription', usd: 3 }), 'subscription');
});

test('the seats never reached are listed with their cause; a missing cause is said', () => {
    const lines = panelPoolLines({ panel_facts: { ...ADVISORY, not_reached: [
        { seat_id: 'grok', cause: 'quota window spent, resets 18:40' }, { seat_id: 'critic', cause: '' }, { seat_id: '' }] } });
    assert.ok(lines.includes('Not reached: grok (quota window spent, resets 18:40)'));
    assert.ok(lines.includes('Not reached: critic (cause not recorded)'));
    assert.equal(lines.filter((line) => line.startsWith('Not reached: ')).length, 2);
    assert.ok(!panelPoolLines({ panel_facts: ADVISORY }).some((line) => line.startsWith('Not reached')));
});

test('a settled record never awaits its lost seat: it is not reached; an open record still awaits it', () => {
    const lost = { slot_id: 'grok', operation_state: 'custody_lost', transport_status: 'error' };
    const lines = (state) => panelPoolLines({ record_state: state, actors: [lost], panel_facts: {
        ...ADVISORY, not_reached: [{ seat_id: 'grok', cause: 'custody_lost' }] } });
    assert.ok(lines('settled').includes('Not reached: grok (custody_lost)'));
    assert.ok(!lines('settled').some((line) => line.startsWith('Awaiting an answer')));
    assert.ok(lines('pending').includes('Awaiting an answer: grok'));
    assert.ok(!lines('pending').some((line) => line.startsWith('Not reached')));
});

test('the wave total counts the priced seats and names the seats it cannot include', () => {
    const total = (usd, seats) => waveTotalText({ usd, seats });
    assert.equal(total(0.42, { usd: 2, subscription: 0, unknown: 0 }), 'Wave total: $0.42');
    assert.equal(total(0.42, { usd: 2, subscription: 1, unknown: 0 }), 'Wave total: $0.42 · 1 seat on a subscription');
    assert.equal(total(0.42, { usd: 1, subscription: 0, unknown: 2 }), 'Wave total: $0.42 for the priced seats · 2 seats cost unknown');
    assert.equal(total(null, { usd: 0, subscription: 1, unknown: 1 }), 'Wave total: cost unknown · 1 seat on a subscription');
    assert.equal(total(null, { usd: 0, subscription: 3, unknown: 0 }), 'Wave total: 3 seats on a subscription');
    assert.equal(total(0, { usd: 1, subscription: 0, unknown: 0 }), 'Wave total: $0.0000');
    assert.equal(waveTotalText(undefined), '');
    assert.ok(formatReviewProjection({ panels: [poolPanel(ADVISORY)] }).split('\n')
        .includes('Wave total: $0.12 for the priced seats · 1 seat on a subscription · 1 seat cost unknown'));
});

test('an empty pool says no reviewer was ticked instead of a fraction of none', () => {
    assert.equal(panelPoolLines({ panel_facts: { ...ADVISORY, pool: [], seats: 1, additional_seats: 0, chosen_by: 'owner', reason: '' } })[0],
        'Panel: Advisory · chosen by you · no reviewer ticked · 1 counted seat');
    assert.equal(panelPoolLines({ panel_facts: { ...ADVISORY, pool: undefined, additional_seats: 0 } })[0],
        'Panel: Advisory · chosen by Ouroboros · 2 counted seats');
});

test('panel facts are said once whatever the seat count; each seat line carries only its own facts', () => {
    // docs/DESIGN.md §6 and AGENTS "fixture geometry": the panel line is the same for 2 seats and 18,
    // and no seat line repeats a fact every seat shares.
    for (const size of [2, 18]) {
        const actors = Array.from({ length: size }, (_, index) => seat(`s${index + 1}`));
        const facts = { ...ADVISORY, seats: size, additional_seats: 0, pool: actors.map((actor) => actor.slot_id) };
        const lines = formatReviewProjection({ panels: [poolPanel(facts, { actors })] }).split('\n');
        assert.deepEqual(lines.filter((line) => line.startsWith('Panel: ')), [`Panel: Advisory · chosen by Ouroboros · ${size} of ${size} ticked reviewers`]);
        const seatLines = lines.filter((line) => /^Reviewer s\d+: /.test(line));
        assert.equal(seatLines.length, size);
        assert.ok(seatLines.every((line) => !/Advisory|chosen by|ticked|Wave total/.test(line)), seatLines[0]);
    }
});

test('a seat still owed an answer reads as awaiting, never as not reached; the record list follows each seat', () => {
    // The ledger's open seats: a planned wait, an operation in flight, custody lost, the plain `pending` status
    // a plan ask records, and an adviser in flight. The record's list was written while the wave was open.
    const open = (fields) => ({ transport_status: 'pending', parse_status: 'none', semantic_verdict: '', ...fields });
    const actors = [
        seat('a'),
        seat('b', open({ operation_state: 'in_flight' })),
        seat('c', open({ operation_state: 'pending_dispatch' })),
        seat('d', open({ operation_state: 'custody_lost' })),
        seat('e', open({ operation_state: 'pending' })),
        seat('critic', open({ operation_state: 'in_flight', additional: true })),
        seat('x', { transport_status: 'error', parse_status: 'none', semantic_verdict: '', operation_state: 'error', cost: { kind: 'unknown', usd: null } }),
    ];
    assert.deepEqual(actors.map(actorOwed), [false, true, true, true, true, true, false]);
    const facts = { ...ADVISORY, seats: 6, additional_seats: 1, not_reached: [{ seat_id: 'a', cause: 'in_flight' },
        { seat_id: 'b', cause: 'in_flight' }, { seat_id: 'c', cause: 'pending_dispatch' }, { seat_id: 'x', cause: 'quota window spent' }] };
    const lines = panelPoolLines({ panel_facts: facts, actors });
    assert.ok(lines.includes('Awaiting an answer: b, c, d, e, critic (adviser)'), lines.join('\n'));
    // A seat that answered late was reached; only the seat that failed is said not reached.
    assert.deepEqual(lines.filter((line) => line.startsWith('Not reached: ')), ['Not reached: x (quota window spent)']);
    // Nothing owed: no awaiting line, and a failed seat stays not reached.
    const settled = panelPoolLines({ panel_facts: { ...facts, not_reached: [{ seat_id: 'x', cause: 'quota window spent' }] },
        actors: [actors[0], actors[6]] });
    assert.ok(!settled.some((line) => line.startsWith('Awaiting')), settled.join('\n'));
    assert.ok(settled.includes('Not reached: x (quota window spent)'));
    // A seat the list names but the card has no row for keeps the record's word.
    assert.ok(panelPoolLines({ panel_facts: facts, actors: [] }).includes('Not reached: b (in_flight)'));
});

test('a preflight record reads its pre-pool line, whatever pool facts its record carries', () => {
    // One named row, ticked or not, owing no reason (the preflight facts after fix round 1).
    const preflight = { seats: 1, additional_seats: 0, distinct_models: 1, observed_unknown_seats: 0, distinct_engines: 1,
        single_model_panel: false, composition: 'composed', chosen_by: 'ouroboros', reason: '', reason_missing: false };
    const scout = { ...seat('scout'), cost: undefined, additional: undefined };
    const card = (surface, facts) => formatReviewProjection({ panels: [panel({ surface, panel_facts: facts, actors: [scout] })] });
    const plain = card('preflight', preflight);
    assert.match(plain, /^Panel composition: composed · chosen by ouroboros · seats=1 · distinct models=1 · distinct engines=1$/m);
    assert.doesNotMatch(plain, /^Panel: |ticked|Why these reviewers|Wave total|Awaiting|Reviewer scout: counted/m);
    // A preflight written while the composer still stamped pool facts on it reads the same line.
    const stamped = { ...preflight, strictness: 'advisory', pool: ['a', 'b', 'c'] };
    const old = card('preflight', stamped);
    assert.match(old, /^Panel composition: composed · chosen by ouroboros · seats=1/m);
    assert.doesNotMatch(old, /^Panel: |ticked reviewer|Why these reviewers|Wave total|Reviewer scout: counted/m);
    // The same facts on a change wave are a pool panel.
    assert.match(card('change', stamped), /^Panel: Advisory · chosen by Ouroboros · 1 of 3 ticked reviewers$/m);
});
