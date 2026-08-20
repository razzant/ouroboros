import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    costDashboardPresentation,
    headerBudgetPresentation,
    mergeStickyCostMeta,
    taskCostMeta,
    taskCostProjection,
    withTaskCostMeta,
} from '../modules/costs.js';
import { summarizeLogEvent } from '../modules/log_events.js';
import {
    accountedUpperBound,
    accountedUpperBoundWithChildren,
    formatUsd4,
} from '../modules/utils.js';

// The nullable-money contract now belongs to costs.js, which owns the single
// optionalFiniteNumber helper: a null ledger reading must never render as $0.
function assertNullableCostPresentation() {
    assert.deepEqual(headerBudgetPresentation(), {
        state: 'loading', label: 'Loading…', fillPct: 0,
    });
    assert.deepEqual(headerBudgetPresentation({ accounting: { available: false } }), {
        state: 'unavailable', label: 'Unavailable', fillPct: 0,
    });
    assert.equal(
        headerBudgetPresentation({ accounting: { available: true }, spent_usd: null, budget_limit: 10 }).state,
        'unavailable',
    );
    assert.equal(costDashboardPresentation({ accounting: { available: true, accounted_usd: null } }).state,
        'unavailable');
}

test('header starts loading and fails closed when ledger money is unavailable', () => {
    assertNullableCostPresentation();
});

test('header accepts the legacy numeric state shape without fabricating null as zero', () => {
    assert.deepEqual(headerBudgetPresentation({ spent_usd: 0, budget_limit: 10 }), {
        state: 'available', label: '$0 / $10', fillPct: 0,
    });
});

test('task cards distinguish unavailable, pending zero, and final zero', () => {
    assert.deepEqual(taskCostMeta({
        cost_usd: null,
        cost_accounting_status: 'unavailable',
        cost_final: false,
    }), ['cost unavailable']);

    assert.deepEqual(taskCostMeta({
        cost_usd: 0,
        cost_accounting_status: 'available',
        cost_final: false,
        reserved_usd: 1.25,
        unresolved_upper_bound_usd: 0.5,
    }), ['cost=$0.00 (pending)', 'reserved=$1.25', 'unresolved≤$0.50']);

    assert.deepEqual(taskCostMeta({
        cost_usd: 0,
        cost_accounting_status: 'available',
        cost_final: true,
    }), ['cost=$0.00']);
});

test('a bare per-round cost_usd delta is NOT task cost (v6.82 P1)', () => {
    // llm_round_finished carries only cost_usd — no task-scope accounting
    // evidence — so it must render nothing and produce no sticky projection.
    assert.deepEqual(taskCostMeta({ cost_usd: 0.03 }), []);
    assert.equal(taskCostProjection({ cost_usd: 0.03 }, '2026-07-29T00:00:00Z'), null);
    // Task-scope frames (subagent progress_meta shape) still qualify.
    const projection = taskCostProjection({
        cost_usd: 0.12,
        cost_accounting_status: 'available',
        cost_final: false,
    }, '2026-07-29T00:00:00Z');
    assert.deepEqual(projection.meta, ['cost=$0.12 (pending)']);
    assert.equal(projection.final, false);
    assert.equal(projection.ts, Date.parse('2026-07-29T00:00:00Z'));
});

test('sticky card cost survives costless frames and obeys finality/timestamp precedence', () => {
    const pendingEarly = taskCostProjection({
        cost_usd: 0.5, cost_accounting_status: 'available', cost_final: false,
    }, '2026-07-29T00:00:00Z');
    const pendingLate = taskCostProjection({
        cost_usd: 0.8, cost_accounting_status: 'available', cost_final: false,
    }, '2026-07-29T00:05:00Z');
    const finalMid = taskCostProjection({
        cost_usd: 0.9, cost_accounting_status: 'available', cost_final: true,
    }, '2026-07-29T00:02:00Z');

    // A frame without cost evidence never touches the stored projection.
    assert.equal(mergeStickyCostMeta(pendingEarly, null), pendingEarly);
    assert.equal(mergeStickyCostMeta(null, pendingEarly), pendingEarly);
    // Newer pending replaces older pending; an older replay can NOT overwrite.
    assert.equal(mergeStickyCostMeta(pendingEarly, pendingLate), pendingLate);
    assert.equal(mergeStickyCostMeta(pendingLate, pendingEarly), pendingLate);
    // Final outranks pending regardless of timestamp direction.
    assert.equal(mergeStickyCostMeta(pendingLate, finalMid), finalMid);
    assert.equal(mergeStickyCostMeta(finalMid, pendingLate), finalMid);
    // An older final can never overwrite a newer final.
    const finalLate = taskCostProjection({
        cost_usd: 1.1, cost_accounting_status: 'available', cost_final: true,
    }, '2026-07-29T00:09:00Z');
    assert.equal(mergeStickyCostMeta(finalLate, finalMid), finalLate);
    assert.equal(mergeStickyCostMeta(finalMid, finalLate), finalLate);
});

test('an unreadable timestamp never defeats a timestamped projection (v6.82 P1)', () => {
    const stamped = taskCostProjection(
        { cost_usd: 1.5, cost_accounting_status: 'available', cost_final: false },
        '2026-07-29T05:00:00Z',
    );
    const unstamped = taskCostProjection(
        { cost_usd: 0.2, cost_accounting_status: 'available', cost_final: false },
        'not-a-timestamp',
    );
    // Equal finality, unreadable incoming stamp: the timestamped value stands.
    assert.deepEqual(mergeStickyCostMeta(stamped, unstamped), stamped);
    // Mirror case: an unreadable STORED stamp yields to a timestamped frame.
    assert.deepEqual(mergeStickyCostMeta(unstamped, stamped), stamped);
});

test('unavailable accounting is an honest unknown, not a settled value', () => {
    const unavailable = taskCostProjection({
        cost_accounting_status: 'unavailable',
    }, '2026-07-29T00:01:00Z');
    assert.deepEqual(unavailable.meta, ['cost unavailable']);
    assert.equal(unavailable.final, false);
    assert.equal(unavailable.unavailable, true);
    // It survives costless frames (stickiness) but yields to a real reading.
    assert.equal(mergeStickyCostMeta(unavailable, null), unavailable);
    const pendingLater = taskCostProjection({
        cost_usd: 0.2, cost_accounting_status: 'available', cost_final: false,
    }, '2026-07-29T00:03:00Z');
    assert.equal(mergeStickyCostMeta(unavailable, pendingLater), pendingLater);
});

test('cost dashboard distinguishes loading, unavailable, pending, and final zero', () => {
    assert.deepEqual(costDashboardPresentation(), { state: 'loading' });
    assert.deepEqual(costDashboardPresentation({ accounting: { available: false } }), {
        state: 'unavailable',
    });

    const base = {
        total_calls: 0,
        by_model: {},
        accounting: {
            available: true,
            accounted_usd: 0,
            confirmed_usd: 0,
            reserved_usd: 0,
            unresolved_upper_bound_usd: 0,
            unknown_unmetered: 0,
            limit_usd: 10,
            cost_final: false,
        },
    };
    const pending = costDashboardPresentation(base);
    assert.equal(pending.accountedLimit, '$0.00 / $10.00');
    // An older payload carries no cause. Say only what is known — never invent "0 open".
    assert.equal(pending.final, 'Pending');
    assert.equal(pending.calls, '0');

    // A flag without its cause is not reconstructible. This exact snapshot — every dollar
    // bucket $0.00, unknown 0, cost_final false — is what an ESTIMATED $0.00 produces, and
    // it rendered "Pending" with the reason nowhere on the page.
    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, non_final_rows: 1 },
    }).final, 'Pending (1 open)');
    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, non_final_rows: 3 },
    }).final, 'Pending (3 open)');
    // The count never contradicts the flag it explains, and a settled ledger says "Yes".
    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: true, non_final_rows: 0 },
    }).final, 'Yes');
    // Non-final with ZERO open rows is a real shape, not a contradiction: a torn ledger
    // tail makes `_with_integrity` clear `cost_final` on its own authority. The cause is
    // then `integrity_degraded`, so this must NOT fabricate "(0 open)" and blame rows.
    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: false, non_final_rows: 0 },
    }).final, 'Pending');

    const final = costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: true },
    });
    assert.equal(final.final, 'Yes');

    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, accounted_usd: null },
    }).state, 'unavailable');
});

test('an unavailable snapshot is sticky but never pins the card (v6.82 r2)', () => {
    const unavailable = taskCostProjection({ cost_accounting_status: 'unavailable' }, '2026-07-29T00:01:00Z');
    const laterHonest = taskCostProjection(
        { cost_usd: 0.4, cost_accounting_status: 'available', cost_final: false },
        '2026-07-29T00:03:00Z',
    );
    const settled = taskCostProjection(
        { cost_usd: 0.9, cost_accounting_status: 'available', cost_final: true },
        '2026-07-29T00:04:00Z',
    );
    // A costless frame keeps it, but a later HONEST reading replaces it...
    assert.equal(mergeStickyCostMeta(unavailable, null), unavailable);
    assert.equal(mergeStickyCostMeta(unavailable, laterHonest), laterHonest);
    // ...while a settled final value still outranks both.
    assert.equal(mergeStickyCostMeta(laterHonest, settled), settled);
    assert.equal(mergeStickyCostMeta(settled, unavailable), settled);
});

test('a cost-only frame never moves the card’s Latest clock', () => {
    // "Latest" answers "when did this task last DO something". A cost frame carries
    // no narration, so letting it move the clock would make a silent card look
    // freshly active. Pinned at source: the meta line reads the activity clock, and
    // only a human/activity-bearing frame advances it.
    const view = readFileSync(new URL('../modules/chat_live_card_view.js', import.meta.url), 'utf8');
    // The clock writer moved with applyLiveCardStateMutation into the
    // live-card store (W3 wave D); the pinned pattern is unchanged.
    const source = readFileSync(new URL('../modules/chat_live_cards.js', import.meta.url), 'utf8');
    assert.match(view, /record\.latestActivityTs \? `Latest \$\{record\.latestActivityTs\}`/);
    assert.match(source, /if \(ts && \(summary\.human \|\| activityCandidate\)\) record\.latestActivityTs = ts/);
});

test('one precedence rule: the deprecated alias wins a diverged pair, in every reader', () => {
    // F7: chat.js used to prefer the additive name while the Python write seam
    // re-converged on the deprecated one, so the same record read differently on
    // the two sides of the wire. Both now ask the shared resolver.
    const diverged = {
        cost_usd: 1, accounted_upper_bound_usd: 9,
        cost_accounting_status: 'available', cost_final: true,
    };
    assert.equal(accountedUpperBound(diverged), 1);
    assert.deepEqual(taskCostMeta(diverged), ['cost=$1.00']);
    // The additive name alone still reads (a producer that only writes it).
    assert.equal(accountedUpperBound({ accounted_upper_bound_usd: 9 }), 9);
    assert.equal(accountedUpperBound({}), null);
    assert.equal(accountedUpperBoundWithChildren(
        { cost_usd_with_children: 2, accounted_upper_bound_usd_with_children: 7 }), 2);
});

test('log events read the shared cost names and stop hiding a real $0', () => {
    // F13: log_events read ONLY the deprecated names, with falsy coercion — a
    // genuine $0.0000 round rendered as nothing, indistinguishable from unknown.
    assert.equal(formatUsd4(0), '$0.0000');
    assert.equal(formatUsd4(null), '');
    assert.equal(formatUsd4(undefined), '');
    const finalized = summarizeLogEvent({
        type: 'task_cost_finalized',
        accounted_upper_bound_usd: 0,
        accounted_upper_bound_usd_with_children: 1.5,
        post_task_status: 'completed',
    });
    assert.ok(finalized.meta.includes('$0.0000'), JSON.stringify(finalized.meta));
    assert.ok(finalized.meta.includes('subtree=$1.5000'), JSON.stringify(finalized.meta));
    const done = summarizeLogEvent({
        type: 'task_done', status: 'completed',
        accounted_upper_bound_usd: 0, cost_final: true,
        cost_accounting_status: 'available',
    });
    assert.ok(done.meta.includes('$0.0000'), JSON.stringify(done.meta));
});

// A live frame's cost evidence, presented for the card. Money renders ONLY from
// the card's sticky projection, so any summarizer-built `cost=` string is
// dropped — a frame without task-scope accounting shows no money at all rather
// than a bare per-call number.
function presentedFrame(summary, payload, options) {
    return withTaskCostMeta(summary, payload, options);
}

test('a summarizer cost string is dropped unconditionally; other meta survives', () => {
    const out = presentedFrame(
        { headline: 'Working', meta: ['cost=$0.02', 'rounds=3'] },
        { cost_usd: 0.5, cost_accounting_status: 'available', cost_final: true },
        { rawTs: '2026-08-17T00:00:00Z' },
    );
    assert.deepEqual(out.meta, ['rounds=3']);
    assert.deepEqual(out.costProjection.meta, taskCostMeta({
        cost_usd: 0.5, cost_accounting_status: 'available', cost_final: true,
    }));
});

test('a replace frame keeps no summarizer meta at all, and the source object is untouched', () => {
    const summary = { headline: 'Done', meta: ['rounds=3'] };
    const out = presentedFrame(summary, {}, { replace: true });
    assert.deepEqual(out.meta, []);
    assert.deepEqual(summary.meta, ['rounds=3'], 'the presentation never mutates the summarizer output');
    assert.equal('costProjection' in out, false, 'no accounting evidence attaches no projection');
});
