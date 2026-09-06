import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    headerBudgetPresentation,
    mergeStickyCostMeta,
    taskCostMeta,
    taskCostProjection,
} from '../modules/chat.js';
import { costBucketPresentation, costDashboardPresentation } from '../modules/costs.js';
import { summarizeLogEvent } from '../modules/log_events.js';
import {
    accountedUpperBound,
    accountedUpperBoundWithChildren,
    formatUsd4,
} from '../modules/utils.js';

test('header starts loading and fails closed when ledger money is unavailable', () => {
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

    // The producer's amount is already the upper bound (settled + reserved +
    // unresolved, cost_projection.py); the openness fields ride beside it and the
    // card states the one number as a ceiling while the ledger is open.
    assert.deepEqual(taskCostMeta({
        cost_usd: 1.75,
        cost_accounting_status: 'available',
        cost_final: false,
        reserved_usd: 1.25,
        unresolved_upper_bound_usd: 0.5,
    }), ['up to $1.75']);

    assert.deepEqual(taskCostMeta({
        cost_usd: 0,
        cost_accounting_status: 'available',
        cost_final: true,
    }), ['$0.00']);
});

test('compact task cards show ONE amount whose wording carries the openness', () => {
    // Calls with no known price are not named on the card (owner: no counter);
    // the open ledger still reads as a ceiling.
    assert.deepEqual(taskCostMeta({
        cost_usd: 55.86,
        cost_usd_with_children: 76.82,
        cost_accounting_status: 'available',
        cost_final: false,
        cost_with_children_partial: true,
        reserved_usd: 1.25,
        unresolved_upper_bound_usd: 0.5,
        unknown_unmetered: 2,
    }), ['up to $76.82']);
    // Every call priced, ledger still open: a ceiling.
    assert.deepEqual(taskCostMeta({
        cost_usd: 55.86,
        cost_usd_with_children: 76.82,
        cost_accounting_status: 'available',
        cost_final: false,
        cost_with_children_partial: true,
        reserved_usd: 1.25,
        unresolved_upper_bound_usd: 0.5,
    }), ['up to $76.82']);
    assert.deepEqual(taskCostMeta({
        cost_usd: 4.25,
        cost_accounting_status: 'available',
        cost_final: true,
    }), ['$4.25']);

    const partialChild = taskCostProjection({
        cost_usd: 4.25,
        cost_usd_with_children: 6.5,
        cost_accounting_status: 'available',
        cost_final: true,
        cost_with_children_partial: true,
    }, '2026-07-29T00:00:00Z');
    assert.deepEqual(partialChild.meta, ['up to $6.50']);
    assert.equal(partialChild.final, false);
});

test('unknown zero-dollar accounting stays pending instead of becoming free', () => {
    assert.deepEqual(taskCostMeta({
        cost_usd: null,
        cost_usd_with_children: null,
        cost_accounting_status: 'available',
        cost_final: false,
        cost_with_children_partial: true,
        unknown_unmetered: 1,
    }), ['cost pending']);
    assert.deepEqual(taskCostMeta({
        cost_usd: 0,
        cost_accounting_status: 'available',
        cost_final: false,
        unknown_unmetered: 1,
    }), ['cost pending']);
});

test('a bare per-round cost_usd delta is NOT task cost (v6.82 P1)', () => {
    // llm_round_finished carries only cost_usd — no task-scope accounting
    // evidence — so it must render nothing and produce no sticky projection.
    assert.deepEqual(taskCostMeta({ cost_usd: 0.03 }), []);
    assert.equal(taskCostProjection({ cost_usd: 0.03 }, '2026-07-29T00:00:00Z'), null);
    // Nor is a frame that merely carries the NAMES: chat.js `costMetaKeys`
    // materializes all twelve as own properties valued `undefined`, and a key
    // without a value is not accounting evidence.
    assert.deepEqual(taskCostMeta({
        cost_usd: undefined, accounted_upper_bound_usd: undefined,
        cost_accounting_status: undefined, cost_final: undefined,
        unknown_unmetered: undefined, reserved_usd: undefined,
    }), []);
    // Task-scope frames (subagent progress_meta shape) still qualify.
    const projection = taskCostProjection({
        cost_usd: 0.12,
        cost_accounting_status: 'available',
        cost_final: false,
    }, '2026-07-29T00:00:00Z');
    assert.deepEqual(projection.meta, ['up to $0.12']);
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

test('legacy breakdown buckets disclose unknown and pending zero amounts', () => {
    assert.equal(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 1, non_final_rows: 1, cost_final: false,
    }), 'cost pending (unmetered=1)');
    assert.equal(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 0, non_final_rows: 1, cost_final: false,
    }), 'cost pending');
    assert.equal(costBucketPresentation({
        cost: 0.25, calls: 2, unknown_unmetered: 1, non_final_rows: 1, cost_final: false,
    }), '$0.25 (pending, unmetered=1)');
    assert.equal(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 0, non_final_rows: 0, cost_final: true,
    }), '$0.00');
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

test('a cost-only frame never moves the card’s activity clock', () => {
    // "updated" answers "when did this task last DO something". A cost frame carries
    // no narration, so letting it move the clock would make a silent card look
    // freshly active. Pinned at source: the meta line reads the activity clock, and
    // only a human/activity-bearing frame advances it.
    const source = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(source, /record\.latestActivityTs \? `updated \$\{record\.latestActivityTs\}`/);
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
    assert.deepEqual(taskCostMeta(diverged), ['$1.00']);
    // The additive name alone still reads (a producer that only writes it).
    assert.equal(accountedUpperBound({ accounted_upper_bound_usd: 9 }), 9);
    assert.equal(accountedUpperBound({}), null);
    // ABI-3: a browser producer literal materializes the retired name as an own
    // property valued `undefined`. That is a key the wire never carried, not a
    // diverged pair — so the honest name is read. An explicit `null` IS present
    // and still wins (Python parity with `old in src`).
    assert.equal(accountedUpperBound({ cost_usd: undefined, accounted_upper_bound_usd: 9 }), 9);
    assert.equal(accountedUpperBound({ cost_usd: null, accounted_upper_bound_usd: 9 }), null);
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

test('llm round rows show money from BOTH the honest backfill name and the live frame', () => {
    // Fix-round-3: /api/logs converts durable rows to the honest name, but the
    // Logs renderer read only `cost_usd ?? cost` — the LLM-round cost column
    // was empty after a page reload. The pair now resolves via the SSOT
    // helper, with the live-frame `cost` spelling as the last fallback.
    const backfill = summarizeLogEvent({
        type: 'llm_round_finished', round: 2, model: 'm',
        accounted_upper_bound_usd: 0.1234,
    });
    assert.ok(backfill.meta.includes('$0.1234'), JSON.stringify(backfill.meta));
    const live = summarizeLogEvent({
        type: 'llm_round_finished', round: 2, model: 'm', cost: 0.5,
    });
    assert.ok(live.meta.includes('$0.5000'), JSON.stringify(live.meta));
    const usage = summarizeLogEvent({
        type: 'llm_usage', model: 'm', accounted_upper_bound_usd: 0.25,
    });
    assert.ok(usage.meta.includes('$0.2500'), JSON.stringify(usage.meta));
    // Diverged stored pair keeps the ONE precedence rule (deprecated wins).
    const diverged = summarizeLogEvent({
        type: 'llm_round_finished', round: 1, model: 'm',
        cost_usd: 0.9, accounted_upper_bound_usd: 0.1,
    });
    assert.ok(diverged.meta.includes('$0.9000'), JSON.stringify(diverged.meta));
});
