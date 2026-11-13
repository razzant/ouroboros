import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    headerBudgetPresentation,
    mergeStickyCostMeta,
    taskCostMeta,
    taskCostProjection,
} from '../modules/chat_activity.js';
import { costPresentationDetail, costPresentationMeta, renderLiveCardMeta } from '../modules/chat_activity.js';
import { TABLES, allowancePresentation, breakdownRowName, costBucketPresentation, costDashboardPresentation,
    roomTimeText } from '../modules/costs.js';
import { normalizeConsciousness } from '../modules/consciousness_api.js';
import { summarizeLogEvent } from '../modules/log_events.js';
import {
    accountedUpperBound,
    accountedUpperBoundWithChildren,
    formatUsd4,
} from '../modules/utils.js';

test('header starts loading and fails closed when ledger money is unavailable', () => {
    assert.deepEqual(headerBudgetPresentation(), {
        state: 'loading', label: 'Loading…', fillPct: 0, attention: false,
    });
    assert.deepEqual(headerBudgetPresentation({ accounting: { available: false } }), {
        state: 'unavailable', label: 'Unavailable', fillPct: 0, attention: false,
    });
    assert.equal(
        headerBudgetPresentation({ accounting: { available: true }, spent_usd: null, budget_limit: 10 }).state,
        'unavailable',
    );
});

test('header accepts the legacy numeric state shape without fabricating null as zero', () => {
    assert.deepEqual(headerBudgetPresentation({ spent_usd: 0, budget_limit: 10 }), {
        state: 'available', label: '$0 / $10', fillPct: 0, attention: false,
    });
});

// The budget is a neutral fact; it is the owner's move only while a task the spending limit stopped waits for the owner
// (gateway/main_decisions.py: a Main attention row whose reason is the budget), even at the limit itself.
test('the budget pill asks for the owner only while a spending-limit stop waits for the owner', () => {
    const atLimit = { spent_usd: 10, budget_limit: 10 };
    assert.deepEqual([headerBudgetPresentation(atLimit).fillPct, headerBudgetPresentation(atLimit).attention], [100, false]);
    const resume = { task_id: 'r', attention_reason: 'restart' };
    const access = { task_id: 'a', attention_reason: 'model_access' };
    assert.equal(headerBudgetPresentation({ ...atLimit, main_attention: [resume, access] }).attention, false);
    const stopped = { task_id: 's', phase: 'budget_paused', can_resume: true, attention_reason: 'budget' };
    assert.equal(headerBudgetPresentation({ ...atLimit, main_attention: [resume, stopped] }).attention, true);
    // The fact rides beside the money, whatever the ledger read says.
    assert.equal(headerBudgetPresentation({ accounting: { available: false }, main_attention: [stopped] }).attention, true);
    assert.equal(headerBudgetPresentation({ ...atLimit, main_attention: null }).attention, false);
    // The bar's owner (chat_header.js) writes it on the pill and the stylesheet keeps amber for it alone; Panic stays danger.
    const header = readFileSync(new URL('../modules/chat_header.js', import.meta.url), 'utf8');
    assert.match(header, /budget\.dataset\.attention = view\.attention \? '1' : '0'/);
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const rules = (selector) => [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)]
        .filter(([, head]) => head.includes(selector)).map(([, head, body]) => `${head.trim()} { ${body.trim()} }`);
    const amber = /--amber|--status-warn|--tone-amber|245,\s*158,\s*11/;
    const budget = rules('chat-budget');
    assert.ok(budget.length >= 6, 'the budget rules are read');
    assert.deepEqual(budget.filter((rule) => amber.test(rule) && !rule.includes('[data-attention="1"]')), [],
        'nothing paints the budget amber without the owner\'s move');
    assert.ok(budget.some((rule) => amber.test(rule) && rule.includes('[data-attention="1"]')));
    assert.ok(rules('.chat-header-btn.danger').some((rule) => /--panic-fill|--panic-ink|--tone-red|--status-error/.test(rule)));
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
    }), ['Cost unknown']);
    assert.deepEqual(taskCostMeta({
        cost_usd: 0,
        cost_accounting_status: 'available',
        cost_final: false,
        unknown_unmetered: 1,
    }), ['Cost unknown']);
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
            settled_usd: 0,
            accounted_usd: 0,
            confirmed_usd: 0,
            estimated_usd: 0,
            reserved_usd: 0,
            unresolved_upper_bound_usd: 0,
            unknown_unmetered: 0,
            limit_usd: 10,
            cost_final: false,
        },
    };
    const fact = (view, label) => Object.fromEntries(view.facts)[label];
    const pending = costDashboardPresentation(base);
    assert.equal(pending.figure, '$0.00 of $10.00');
    assert.equal(pending.left, '$10.00 left');
    // An older payload carries no cause. Say only what is known — never invent "0 open".
    assert.equal(fact(pending, 'Final'), 'No');
    assert.equal(fact(pending, 'Physical attempts'), '0');

    // A flag without its cause is not reconstructible. This exact snapshot — every dollar
    // bucket $0.00, unknown 0, cost_final false — is what an ESTIMATED $0.00 produces, and
    // it rendered "Pending" with the reason nowhere on the page.
    assert.equal(fact(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, non_final_rows: 1 },
    }), 'Final'), 'No (1 open)');
    assert.equal(fact(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, non_final_rows: 294359 },
    }), 'Final'), 'No (294,359 open)', 'a count reads grouped like money');
    // Open rows are the ledger's open ones: calls that will never get a price (subscription
    // work, in non_final_rows) are not "open" (05.11: 3,176 open among 294,141 non-final).
    assert.equal(fact(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, non_final_rows: 294141, accounting_open_rows: 3176 },
    }), 'Final'), 'No (3,176 open)');
    // The count never contradicts the flag it explains, and a settled ledger says "Yes".
    assert.equal(fact(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: true, non_final_rows: 0 },
    }), 'Final'), 'Yes');
    // Non-final with ZERO open rows is a real shape, not a contradiction: a torn ledger
    // tail makes `_with_integrity` clear `cost_final` on its own authority. The cause is
    // then `integrity_degraded`, so this must NOT fabricate "(0 open)" and blame rows.
    assert.equal(fact(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: false, non_final_rows: 0 },
    }), 'Final'), 'No');

    const final = costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, cost_final: true },
    });
    assert.equal(fact(final, 'Final'), 'Yes');

    assert.equal(costDashboardPresentation({
        ...base,
        accounting: { ...base.accounting, settled_usd: null },
    }).state, 'unavailable');
});

test('cost dashboard sets the limit against known spend, open holds beside it (#1487)', () => {
    const shown = costDashboardPresentation({
        total_calls: 3,
        by_model: {},
        accounting: {
            available: true, limit_usd: 10, cost_final: false, non_final_rows: 1, unknown_unmetered: 0,
            settled_usd: 2, confirmed_usd: 1.5, estimated_usd: 0.5,
            reserved_usd: 4, unresolved_upper_bound_usd: 20, accounted_usd: 26,
        },
    });
    // $2 known of $10: the $24 of holds is said beside it and kept in Details, never added in as spending.
    assert.equal(shown.figure, '$2.00 of $10.00');
    assert.equal(shown.left, '$8.00 left');
    assert.equal(shown.open, 'up to $20.00 for calls without a known price', 'the producer\'s words: a worst case, not spending');
    assert.equal(shown.reached, false);
    assert.equal(shown.share, 20);
    assert.deepEqual(shown.facts.map(([label]) => label), ['Final', 'Confirmed', 'Estimated', 'Reserved',
        'Unresolved upper bound', 'Calls without a price', 'Physical attempts'], 'the accounting terms are the record\'s');
    assert.deepEqual(['Confirmed', 'Estimated', 'Reserved', 'Unresolved upper bound'].map((label) => Object.fromEntries(shown.facts)[label]),
        ['$1.50', '$0.50', '$4.00', '$20.00']);
});

test('the known spend leads grouped like money, says what is left, and a reached limit is the owner\'s move', () => {
    const view = (accounting) => costDashboardPresentation({ total_calls: 423015, by_model: {}, accounting: {
        available: true, cost_final: true, non_final_rows: 0, unknown_unmetered: 0, confirmed_usd: 0, estimated_usd: 0,
        reserved_usd: 0, unresolved_upper_bound_usd: 0, ...accounting } });
    const near = view({ settled_usd: 29438.2, limit_usd: 29800, remaining_known_usd: 361.8 });
    assert.equal(near.figure, '$29,438.20 of $29,800.00');
    assert.equal(near.left, '$361.80 left', 'the room the admission rule reports, never recomputed past it');
    assert.equal(near.open, '', 'nothing open is said only when something is');
    assert.equal(near.reached, false);
    assert.equal(Object.fromEntries(near.facts)['Physical attempts'], '423,015');
    const reached = view({ settled_usd: 30500, limit_usd: 30500, remaining_known_usd: 0 });
    assert.equal(reached.reached, true);
    assert.equal(reached.left, 'Limit reached: new paid work is refused');
    assert.equal(reached.share, 100);
    const unlimited = view({ settled_usd: 12.5, limit_usd: 0 });
    assert.deepEqual([unlimited.figure, unlimited.left, unlimited.share], ['$12.50', 'No limit set', null]);
});

test('a breakdown cell is its amount in the card\'s words, what it leaves out on the face under it, never a free zero', () => {
    // Subscription calls settle without a price: none will arrive, so not pending; a session row may have no call.
    assert.deepEqual(costBucketPresentation({
        cost: 0, calls: 0, unknown_unmetered: 3, non_final_rows: 3, accounting_open_rows: 0, cost_final: false,
    }), { text: 'Cost not priced', open: false, detail: '3 without a price' });
    assert.deepEqual(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 1, non_final_rows: 1, accounting_open_rows: 1, cost_final: false,
    }), { text: 'Cost pending', open: true, detail: '1 without a price' });
    assert.deepEqual(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 0, non_final_rows: 1, accounting_open_rows: 1, cost_final: false,
    }), { text: 'Cost pending', open: true, detail: '' });
    assert.deepEqual(costBucketPresentation({
        cost: 0.25, calls: 2, unknown_unmetered: 1, non_final_rows: 1, accounting_open_rows: 0, cost_final: false,
    }), { text: 'Tracked: $0.25', open: false, detail: '1 without a price' });
    assert.deepEqual(costBucketPresentation({
        cost: 0.0004, calls: 1, unknown_unmetered: 0, non_final_rows: 0, accounting_open_rows: 0, cost_final: true,
    }), { text: '$0.0004', open: false, detail: '' }, 'a positive sub-cent amount is never $0.00');
    assert.deepEqual(costBucketPresentation({
        cost: 0.25, calls: 2, unknown_unmetered: 0, non_final_rows: 1, accounting_open_rows: 1, cost_final: false,
    }), { text: '$0.25', open: true, detail: 'pending' });
    assert.deepEqual(costBucketPresentation({
        cost: 5623.32, calls: 2, unknown_unmetered: 764, non_final_rows: 765, accounting_open_rows: 1, cost_final: false,
    }), { text: 'Tracked: $5,623.32', open: true, detail: 'pending · 764 without a price' });
    // An older payload without the open count: its non-final rows past the unpriced ones are the open ones.
    assert.deepEqual(costBucketPresentation({ cost: 0, calls: 1, unknown_unmetered: 1, non_final_rows: 1, cost_final: false }),
        { text: 'Cost not priced', open: false, detail: '1 without a price' });
    assert.deepEqual(costBucketPresentation({
        cost: 0, calls: 1, unknown_unmetered: 0, non_final_rows: 0, cost_final: true,
    }), { text: '$0.00', open: false, detail: '' }, 'a settled zero is a fact');
    assert.deepEqual(costBucketPresentation({ cost: null, calls: 1 }), { text: 'unknown', open: false, detail: '' });
    for (const cell of [costBucketPresentation({ cost: 5623.32, cost_final: false, unknown_unmetered: 764 })]) {
        assert.doesNotMatch(cell.text, /unmetered|pending|without a price|\(/, 'the counts leave the amount');
        assert.equal(cell.detail, '764 without a price', 'and are said under it');
    }
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
    // freshly active. The card's clock (a root's stands in its head, beside the chip) reads
    // the activity clock; which frames advance it: progress_narration_voice.test.js.
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }) };
    try {
        const when = { textContent: '', hidden: true };
        const record = { latestActivityTs: '10:04', metaEl: { innerHTML: '', isConnected: true },
            summaryButtonEl: { querySelector: (sel) => (sel === '[data-live-when]' ? when : null) },
            costMeta: { meta: ['$1.00'] } };
        renderLiveCardMeta(record);
        assert.equal(when.textContent, 'updated 10:04');
        assert.equal(when.hidden, false);
        record.costMeta = { meta: ['$2.00'] };
        renderLiveCardMeta(record);
        assert.equal(when.textContent, 'updated 10:04');
        assert.match(record.metaEl.innerHTML, /\$2\.00/);
        assert.doesNotMatch(record.metaEl.innerHTML, /updated/, 'the clock moved whole to the head: once, never twice');
    } finally { globalThis.document = priorDocument; }
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

test('#498 the scoped carrier answers the whole money question from one bucket', () => {
    const carrier = (over = {}) => ({
        scope: 'own', tracked_amount: 1.2, has_unpriced: false,
        tracked_final: true, accounting_open: false, has_rows: true, ...over,
    });
    // A proven zero: priced rows evidenced it and the accounting is closed.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: 0 })), ['$0.00']);
    // Under a cent is not that zero: four places, as review records print it.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: 0.0004, has_unpriced: true })), ['Tracked: $0.0004']);
    // Below four places a positive amount still never reads as zero.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: 0.00003 })), ['<$0.0001']);
    // An open subtotal is a ceiling, not a receipt.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_final: false, accounting_open: true })),
        ['up to $1.20']);
    // An estimated $0.00 is open, so it never reads as a settled free result.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: 0, tracked_final: false, accounting_open: true })),
        ['up to $0.00']);
    // An open empty ledger is pending; a closed all-unpriced one is not free.
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: null, tracked_final: false, accounting_open: true })),
        ['Cost pending']);
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_amount: null, has_unpriced: true, tracked_final: false, accounting_open: true,
    })), ['Cost pending']);
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_amount: null, has_unpriced: false, tracked_final: false, accounting_open: true,
    }), { ledger_integrity_degraded: true }), ['Cost unknown']);
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_amount: null, has_unpriced: true, tracked_final: false, accounting_open: true,
    }), { cost_custody: { status: 'unavailable' } }), ['Cost unknown']);
    // A known subtotal survives an unreadable custody view or a damaged ledger,
    // qualified as its producer qualified it; only the missing amount reads unknown.
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_amount: 2, has_unpriced: true, tracked_final: false, accounting_open: true,
    }), { cost_custody: { status: 'unavailable' } }), ['Tracked: up to $2.00']);
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_final: false, accounting_open: true,
    }), { ledger_integrity_degraded: true }), ['up to $1.20']);
    assert.equal(costPresentationDetail(carrier({
        tracked_amount: 2, has_unpriced: true, tracked_final: false, accounting_open: true,
    }), { cost_custody: { status: 'unavailable' } }), 'Cost: some steps have no price, so Tracked counts only the priced steps.');
    assert.equal(costPresentationDetail(carrier({
        tracked_amount: null, has_unpriced: true, tracked_final: false, accounting_open: true,
    }), { cost_custody: { status: 'unavailable' } }), '');
    assert.deepEqual(costPresentationMeta(carrier({
        tracked_amount: null, has_unpriced: true, tracked_final: true, accounting_open: false,
    })), ['Cost not priced']);
    assert.equal(costPresentationDetail(carrier({
        tracked_amount: null, has_unpriced: true, tracked_final: true, accounting_open: false,
    })), 'Cost: settled usage has no price, so it is not counted as known spend or as $0.');
    // Mixed: the tracked number is a subtotal and says so by its own word; the reason is
    // stated in words in the card's details (owner 2026-10-12, option B), never in a tooltip.
    const why = 'Cost: some steps have no price, so Tracked counts only the priced steps.';
    assert.deepEqual(costPresentationMeta(carrier({ has_unpriced: true })), ['Tracked: $1.20']);
    assert.equal(costPresentationDetail(carrier({ has_unpriced: true })), why);
    assert.deepEqual(costPresentationMeta(carrier({ tracked_amount: 0, has_unpriced: true })), ['Tracked: $0.00']);
    assert.equal(costPresentationDetail(carrier({ tracked_amount: 0, has_unpriced: true })), why);
    // Open tracked accounting and closed unpriced accounting each explain their state.
    assert.equal(costPresentationDetail(carrier({})), '');
    assert.equal(costPresentationDetail(carrier({ tracked_final: false, accounting_open: true, has_unpriced: true })),
        'Cost: some steps have no price, so Tracked counts only the priced steps.');
    assert.equal(costPresentationDetail(carrier({ tracked_amount: null, has_unpriced: true, accounting_open: false })),
        'Cost: settled usage has no price, so it is not counted as known spend or as $0.');
    // No accountable row at all is not a free result and not a money line.
    assert.deepEqual(costPresentationMeta(carrier({ has_rows: false, tracked_amount: null })), []);
    // No carrier at all defers to the legacy derivation.
    assert.equal(costPresentationMeta(undefined), null);
    assert.equal(costPresentationMeta(null), null);
});

test('#498 a card with a carrier uses it; legacy frames keep the old derivation', () => {
    // The root's carrier describes the SUBTREE, so a null subtree amount stays
    // unknown instead of falling back to the root's own (often zero) number.
    assert.deepEqual(taskCostMeta({
        cost_accounting_status: 'available',
        accounted_upper_bound_usd: 0,
        cost_final: true,
        cost_presentation: {
            scope: 'root_tree', tracked_amount: null, has_unpriced: true,
            tracked_final: false, accounting_open: true, has_rows: true,
        },
    }), ['Cost pending']);
    // An unreadable ledger still outranks the carrier.
    assert.deepEqual(taskCostMeta({
        cost_accounting_status: 'unavailable', cost_presentation: null,
    }), ['cost unavailable']);
    // A carrier alone is task-scope accounting evidence.
    assert.deepEqual(taskCostMeta({
        cost_presentation: {
            scope: 'own', tracked_amount: 4.25, has_unpriced: false,
            tracked_final: true, accounting_open: false, has_rows: true,
        },
    }), ['$4.25']);
    // Legacy frames (no carrier) keep their existing, deliberately conservative reading.
    assert.deepEqual(taskCostMeta({
        cost_usd: 4.25, cost_accounting_status: 'available', cost_final: true,
    }), ['$4.25']);
    // A late receipt just changes the carrier — nothing is sticky in the projection.
    const open = taskCostProjection({
        cost_presentation: {
            scope: 'own', tracked_amount: 1.2, has_unpriced: false,
            tracked_final: false, accounting_open: true, has_rows: true,
        },
    }, '2026-07-29T00:00:00Z');
    const settled = taskCostProjection({
        cost_presentation: {
            scope: 'own', tracked_amount: 1.35, has_unpriced: false,
            tracked_final: true, accounting_open: false, has_rows: true,
        },
    }, '2026-07-29T00:01:00Z');
    assert.equal(open.final, false);
    assert.equal(settled.final, true);
    assert.deepEqual(mergeStickyCostMeta(open, settled).meta, ['$1.35']);
    assert.deepEqual(mergeStickyCostMeta(settled, open).meta, ['$1.35']);
});

test('#498 mixed exact subtotal is not sticky final while prices are unknown', () => {
    const frame = (amount, final, open = false) => ({ cost_presentation: {
        scope: 'root_tree', tracked_amount: amount, tracked_final: final,
        accounting_open: open, has_unpriced: true, has_rows: true,
    } });
    const exactSubtotal = taskCostProjection(frame(0, true), '2026-09-23T00:00:00Z');
    const lateEstimatedPrice = taskCostProjection(frame(1.2, false, true), '2026-09-23T00:01:00Z');
    assert.equal(exactSubtotal.final, false);
    assert.deepEqual(mergeStickyCostMeta(exactSubtotal, lateEstimatedPrice).meta, ['Tracked: up to $1.20']);
    assert.deepEqual(mergeStickyCostMeta(lateEstimatedPrice, exactSubtotal).meta, ['Tracked: up to $1.20']);
    // The reason rides with the amount it explains through the sticky choice.
    assert.equal(mergeStickyCostMeta(exactSubtotal, lateEstimatedPrice).detail,
        'Cost: some steps have no price, so Tracked counts only the priced steps.');
    const ownZero = taskCostProjection({ cost_presentation: {
        scope: 'own', tracked_amount: 0, tracked_final: true,
        accounting_open: false, has_unpriced: false, has_rows: true,
    } }, '2026-09-23T00:02:00Z');
    assert.deepEqual(mergeStickyCostMeta(lateEstimatedPrice, ownZero).meta, lateEstimatedPrice.meta);
});

test('#498 legacy null tree is never replaced by own zero; empty damaged ledger stays unknown', () => {
    assert.deepEqual(taskCostMeta({ accounted_upper_bound_usd_with_children: null,
        accounted_upper_bound_usd: 0, cost_final: true }), ['Cost unknown']);
    assert.deepEqual(costPresentationMeta({ scope: 'root_tree', tracked_amount: null,
        has_rows: true, has_unpriced: false, tracked_final: false, accounting_open: true }), ['Cost pending']);
    assert.deepEqual(costPresentationMeta({ scope: 'root_tree', tracked_amount: null,
        has_rows: false, has_unpriced: true, tracked_final: false, accounting_open: true }), ['Cost pending']);
    assert.deepEqual(costPresentationMeta({ scope: 'root_tree', tracked_amount: null,
        has_rows: false, has_unpriced: false, tracked_final: false, accounting_open: true }), ['Cost unknown']);
    assert.deepEqual(taskCostMeta({ cost_presentation: { scope: 'root_tree', tracked_amount: null,
        has_rows: false, has_unpriced: true, tracked_final: false, accounting_open: true,
    }, cost_custody: { status: 'unavailable' } }), ['Cost unknown']);
    assert.deepEqual(taskCostMeta({ cost_presentation: { scope: 'root_tree', tracked_amount: null,
        has_rows: false, has_unpriced: false, tracked_final: false, accounting_open: true,
    }, ledger_integrity_degraded: true }), ['Cost unknown']);
    assert.deepEqual(taskCostMeta({ cost_presentation: { scope: 'root_tree', tracked_amount: 2,
        has_rows: true, has_unpriced: true, tracked_final: false, accounting_open: true,
    }, cost_custody: { status: 'unavailable' } }), ['Tracked: up to $2.00']);
});

test('#498 exact live subtotal cannot freeze a still-running root heartbeat', () => {
    const exact = taskCostProjection({ cost_final: false, cost_with_children_partial: true,
        cost_presentation: { scope: 'root_tree', tracked_amount: 0, tracked_final: true,
            accounting_open: false, has_unpriced: false, has_rows: true } }, '2026-09-23T00:00:00Z');
    const next = taskCostProjection({ cost_final: false, cost_presentation: {
        scope: 'root_tree', tracked_amount: 0.3, tracked_final: false,
        accounting_open: true, has_unpriced: true, has_rows: true } }, '2026-09-23T00:01:00Z');
    assert.equal(exact.final, false);
    assert.deepEqual(mergeStickyCostMeta(exact, next).meta, ['Tracked: up to $0.30']);
    // A null carrier beside a readable ledger is an UNKNOWN amount (owner
    // vocabulary), never the unreadable-ledger phrase.
    assert.deepEqual(taskCostMeta({ accounted_upper_bound_usd: 0, cost_final: true,
        cost_presentation: null }), ['Cost unknown']);
});

test('#498 legacy unknown price never becomes sticky-final', () => {
    const previous = taskCostProjection({ cost_final: true, accounted_upper_bound_usd: null,
        unknown_unmetered: 1 }, '2026-09-23T00:00:00Z');
    assert.equal(previous.final, false);
    const receipt = taskCostProjection({ cost_final: false, accounted_upper_bound_usd: 0.35 },
        '2026-09-23T00:01:00Z');
    assert.deepEqual(mergeStickyCostMeta(previous, receipt).meta, ['up to $0.35']);
});


test('unknown legacy descendant rollup cannot retain a narrower own zero', () => {
    const own = taskCostProjection({cost_final:true, cost_presentation:{scope:'own', tracked_amount:0,
        tracked_final:true, accounting_open:false, has_rows:true, has_unpriced:false}}, '2026-09-24T00:00:00Z');
    const rollup = taskCostProjection({cost_presentation:null, accounted_upper_bound_usd_with_children:null}, '2026-09-24T00:01:00Z');
    assert.deepEqual(mergeStickyCostMeta(own, rollup).meta, ['Cost unknown']);
    assert.deepEqual(mergeStickyCostMeta(rollup, own).meta, ['Cost unknown']);
    // The unreadable-ledger phrase is reserved for the status flag.
    assert.deepEqual(taskCostMeta({ cost_accounting_status: 'unavailable', cost_presentation: null }),
        ['cost unavailable']);
});

test('#2828 a closed-empty carrier is a versioned clear; closed custody without a receipt reads unknown', () => {
    const carrier = (over = {}) => ({ scope: 'own', tracked_amount: null, has_unpriced: false, tracked_final: false,
        accounting_open: false, has_rows: false, ...over });
    const frame = (over = {}) => ({ cost_accounting_status: 'available', cost_final: false,
        cost_presentation: carrier(over) });
    // Closed with an unpriced run and no row: no amount will arrive, so unknown (never pending, never blank).
    assert.deepEqual(costPresentationMeta(carrier({ has_unpriced: true })), ['Cost unknown']);
    assert.deepEqual(costPresentationMeta(carrier({ has_unpriced: true, accounting_open: true })), ['Cost pending']);
    assert.deepEqual(costPresentationMeta(carrier()), []);

    const pending = taskCostProjection(frame({ has_unpriced: true, accounting_open: true }), '2026-11-09T00:00:00Z');
    const clear = taskCostProjection(frame(), '2026-11-09T00:01:00Z');
    assert.deepEqual(clear, { meta: [], cleared: true, ts: Date.parse('2026-11-09T00:01:00Z'), scope: 'own',
        final: false, unavailable: false });
    // Only a readable, intact closed-empty scope clears; an unreadable one still says unknown.
    assert.equal(taskCostProjection({ cost_accounting_status: 'unavailable', cost_presentation: carrier() }, '').meta[0],
        'cost unavailable');
    assert.deepEqual(taskCostProjection({ ...frame(), ledger_integrity_degraded: true }, '').meta, ['Cost unknown']);
    assert.equal(taskCostProjection({ cost_usd: 0.03 }, '2026-11-09T00:02:00Z'), null);
    assert.equal(mergeStickyCostMeta(pending, clear), clear);
    assert.equal(mergeStickyCostMeta(clear, pending), clear, 'an older pending replay cannot revive a clear');
    assert.equal(mergeStickyCostMeta(clear, null), clear);
    const retry = taskCostProjection(frame({ has_unpriced: true, accounting_open: true }), '2026-11-09T00:03:00Z');
    assert.equal(mergeStickyCostMeta(clear, retry), retry, 'a newer start or retry is pending again');
    const unavailable = taskCostProjection({ cost_accounting_status: 'unavailable' }, '2026-11-09T00:04:00Z');
    assert.equal(mergeStickyCostMeta(clear, unavailable), clear, 'an unavailable read never replaces honest evidence');
    const settled = taskCostProjection({ cost_final: true, cost_presentation: carrier({ has_rows: true,
        tracked_amount: 0.4, tracked_final: true }) }, '2026-11-09T00:00:30Z');
    assert.equal(mergeStickyCostMeta(clear, settled), settled, 'a final amount outranks the clear');
    assert.equal(mergeStickyCostMeta(settled, clear), settled);
    // Scope precedence comes before rank: a root_tree clear holds over an own pending, and yields to a tree reading.
    const treeClear = taskCostProjection(frame({ scope: 'root_tree' }), '2026-11-09T00:01:00Z');
    const ownLater = taskCostProjection(frame({ has_unpriced: true, accounting_open: true }), '2026-11-09T00:09:00Z');
    assert.equal(mergeStickyCostMeta(treeClear, ownLater), treeClear);
    assert.equal(mergeStickyCostMeta(clear, treeClear), treeClear);
    const treePending = taskCostProjection(frame({ scope: 'root_tree', has_unpriced: true, accounting_open: true }),
        '2026-11-09T00:00:10Z');
    assert.equal(mergeStickyCostMeta(clear, treePending), treePending, 'an own clear yields to a root_tree reading');
    // An unstamped clear never defeats a stamped value of equal rank.
    const unstamped = taskCostProjection(frame(), '');
    assert.equal(mergeStickyCostMeta(pending, unstamped), pending);
    // The card draws no money line from a clear.
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ style: {}, setAttribute() {}, append() {} }) };
    try {
        const record = { metaEl: { innerHTML: '', isConnected: true }, finished: true, costMeta: clear };
        renderLiveCardMeta(record);
        assert.doesNotMatch(record.metaEl.innerHTML, /Cost|\$/);
    } finally { globalThis.document = priorDocument; }
});

test('Costs tables say what they group by: provider, a model over every provider, the roles set now (Track09 A)', () => {
    assert.deepEqual(TABLES.map(([id, title, column, key]) => [id, title, column, key]), [
        ['cost-by-model', 'By model', 'Model', 'by_model'],
        ['cost-by-key', 'By provider', 'Provider', 'by_api_key'],
        ['cost-by-model-cat', 'By current role', 'Role', 'by_model_category'],
        ['cost-by-task-cat', 'By task category', 'Category', 'by_task_category'],
    ]);
    const notes = Object.fromEntries(TABLES.map(([id, , , , note]) => [id, note]));
    assert.match(notes['cost-by-model'], /over every provider and connection/);
    assert.match(notes['cost-by-model-cat'], /roles set in Settings now, not when each call ran/);
    assert.equal(breakdownRowName('cost-by-key', 'openrouter'), 'OpenRouter');
    assert.equal(breakdownRowName('cost-by-key', 'claude'), 'Claude Code');
    assert.equal(breakdownRowName('cost-by-key', 'some-new-provider'), 'some-new-provider', 'an unknown key stays itself');
    assert.equal(breakdownRowName('cost-by-model-cat', 'other'), 'No current role');
    assert.equal(breakdownRowName('cost-by-model-cat', 'main'), 'Main');
    assert.equal(breakdownRowName('cost-by-model-cat', 'unattributed'), 'unattributed');
    assert.equal(breakdownRowName('cost-by-model', 'anthropic/claude-opus-5.5'), 'anthropic/claude-opus-5.5');
});


// --- Costs › Budget: the consciousness allowance (moved from Settings, owner 10 Nov 2026) ---------------------------

// Its field is a setting row: what a typed limit sends or refuses, and what a save says, are checked in
// costs_limits.test.js and settings_rows.test.js.

test('room returns by a whole minute after its instant, in a named zone, never early and never from the past', () => {
    const now = Date.UTC(2026, 10, 10, 19, 0);
    assert.equal(roomTimeText('2026-11-10T21:40:30+00:00', { now, timeZone: 'UTC' }), '21:41 UTC');
    assert.equal(roomTimeText('2026-11-10T21:41:00+00:00', { now, timeZone: 'UTC' }), '21:42 UTC', 'after the instant itself');
    assert.equal(roomTimeText('2026-11-10T21:40:30+00:00', { now, timeZone: 'Europe/Moscow' }), 'Nov 11, 00:41 GMT+3',
        'another day there carries its date');
    assert.equal(roomTimeText('2026-11-10T18:59:00+00:00', { now, timeZone: 'UTC' }), '', 'a passed instant says nothing');
    assert.equal(roomTimeText('', { now }), '');
    assert.equal(roomTimeText('soon', { now }), '');
});

const state = (bg) => normalizeConsciousness(bg, { enabled: bg.enabled ?? true });
const HELD = { status: 'wake_paused', live_wake_task_id: 'wake-1', live_wake_held: true, spent_24h_usd: 301.2, daily_usd: 390,
    allowance_status: 'available', live_wake_hold: { cause: 'allowance', ceiling_usd: 2.99 } };

test('the allowance under its field: the clock\'s own spend, what it leaves out, never a zero for unknown', () => {
    const now = Date.UTC(2026, 10, 10, 19, 0);
    assert.deepEqual(allowancePresentation(state({ status: 'sleeping', spent_24h_usd: 12.4, daily_usd: 300,
        allowance_status: 'available', allowance_open_rows: 0 }), { now }),
        { spend: '$12.40 of $300.00 in the last 24 hours', detail: '', lines: [] });
    const open = allowancePresentation(state({ status: 'sleeping', spent_24h_usd: 12.4, daily_usd: 300,
        allowance_status: 'available', allowance_open_rows: 2, unknown_unmetered: 3 }), { now });
    assert.deepEqual([open.spend, open.detail], ['≥ $12.40 of $300.00 in the last 24 hours', 'pending · 3 without a price']);
    const pending = allowancePresentation(state({ status: 'sleeping', spent_24h_usd: 0, daily_usd: 20,
        allowance_status: 'available', allowance_open_rows: 1 }), { now });
    assert.deepEqual([pending.spend, pending.detail], ['$0.00 so far of $20.00 in the last 24 hours', 'pending'],
        'a call still open is no final $0.00');
    const unread = allowancePresentation(state({ status: 'allowance_unknown', daily_usd: 300, allowance_status: 'allowance_unknown' }), { now });
    assert.equal(unread.spend, 'What it spent in the last 24 hours is unknown.');
    assert.deepEqual(unread.lines, [{ text: 'The usage ledger could not be read, so whether consciousness has room is unknown.', tone: 'muted' }]);
    assert.deepEqual(allowancePresentation(null), { spend: 'What it spent in the last 24 hours could not be read.', detail: '', lines: [] });
});

test('a reached limit says what waits and, only when it can, a conditional time in a named zone', () => {
    const now = Date.UTC(2026, 10, 10, 19, 0);
    const reached = { status: 'allowance_exhausted', spent_24h_usd: 300.4, daily_usd: 300, allowance_status: 'exhausted',
        allowance_room_after: '2026-11-10T21:40:30+00:00' };
    assert.deepEqual(allowancePresentation(state(reached), { now, timeZone: 'UTC' }).lines, [{ tone: 'warn',
        text: 'Consciousness limit reached: new wake-ups, replies in places and its own tasks wait.',
        note: 'If its charges stay as they are, there is room again by 21:41 UTC.' }]);
    for (const roomAfter of ['', '2026-11-10T18:00:00+00:00']) {  // withheld by the host, or stale by now
        assert.deepEqual(allowancePresentation(state({ ...reached, allowance_room_after: roomAfter }), { now }).lines, [{ tone: 'warn',
            text: 'Consciousness limit reached: new wake-ups, replies in places and its own tasks wait.',
            note: 'When there is room again cannot be said now.' }]);
    }
    // The current window decides; a stale skip word does not, and an older server without the verdict keeps its word.
    assert.deepEqual(allowancePresentation(state({ ...reached, allowance_status: 'available' }), { now }).lines, []);
    assert.equal(allowancePresentation(state({ ...reached, allowance_status: '' }), { now, timeZone: 'UTC' }).lines.length, 1);
    // A $0 limit is the owner's setting, said plainly, not an alarm and not a forecast.
    assert.deepEqual(allowancePresentation(state({ ...reached, daily_usd: 0, spent_24h_usd: 0 }), { now }).lines, [{ tone: 'muted',
        text: 'At $0 consciousness may not spend: new wake-ups, replies in places and its own tasks do not start.' }]);
    assert.deepEqual(allowancePresentation(state({ ...reached, enabled: false }), { now }).lines, [{ tone: 'muted',
        text: 'Consciousness is off: it starts nothing that counts here until it is started again.' }]);
});

test('a wake held at the allowance it started with is said apart from room returning, with its card\'s door', () => {
    const now = Date.UTC(2026, 10, 10, 19, 0);
    const held = { wake: 'wake-1', tone: 'warn', text: 'The consciousness wake-up is paused at the allowance it had left when it started ($2.99).',
        note: 'Resume keeps that limit, so it pauses again while its known spend is at it; room returning or a higher limit '
            + 'does not raise it. Stop now on its card ends it and frees the clock for its next check.' };
    assert.deepEqual(allowancePresentation(state(HELD), { now }).lines, [held], 'room is back, the wake still waits');
    const both = allowancePresentation(state({ ...HELD, allowance_status: 'exhausted', allowance_room_after: '2026-11-10T21:40:30+00:00',
        live_wake_hold: { cause: 'allowance_and_cap', ceiling_usd: 50 } }), { now, timeZone: 'UTC' }).lines;
    assert.equal(both.length, 2, 'the window and the wake are two facts');
    assert.match(both[0].note, /^If its charges stay as they are, there is room again by 21:41 UTC\.$/);
    assert.equal(both[1].text, 'The consciousness wake-up is paused at the allowance it had left when it started ($50.00, the per-task cap too).');
    // Other proven causes are said only where they are money; the rest keep their card's word on the Consciousness tab.
    assert.deepEqual(allowancePresentation(state({ ...HELD, live_wake_hold: { cause: 'per_task_cap', ceiling_usd: 10 } }), { now }).lines,
        [{ wake: 'wake-1', tone: 'muted', text: 'The consciousness wake-up is paused at the per-task cost cap it started with; its card shows why.' }]);
    assert.deepEqual(allowancePresentation(state({ ...HELD, live_wake_hold: { cause: 'launch_ceiling', ceiling_usd: 2.99 } }), { now }).lines,
        [{ wake: 'wake-1', tone: 'muted', text: 'The consciousness wake-up is paused at the spending limit it started with; its card shows why.' }]);
    assert.deepEqual(allowancePresentation(state({ ...HELD, live_wake_hold: { cause: 'owner', ceiling_usd: null } }), { now }).lines,
        [{ wake: 'wake-1', tone: 'muted', text: 'The consciousness wake-up is on an explicit pause; its card shows who paused it and how to continue.' }]);
    for (const cause of ['sleep', 'technical', 'total_budget', 'per_task_fence', '']) {
        assert.deepEqual(allowancePresentation(state({ ...HELD, live_wake_hold: { cause, ceiling_usd: null } }), { now }).lines, [], cause);
    }
});

test('the model its wake-ups think with: its own slot, or Main\'s said as Main, a local one as Local model; unknown says nothing', () => {
    const now = Date.UTC(2026, 10, 10, 19, 0);
    const base = { status: 'sleeping', spent_24h_usd: 1, daily_usd: 20, allowance_status: 'available' };
    const lines = (route) => allowancePresentation(state({ ...base, model_route: route }), { now }).lines;
    assert.deepEqual(lines({ value: 'anthropic/claude-opus-5-5', layer: 'consciousness', local: false }),
        [{ tone: 'muted', text: 'Thinks with anthropic/claude-opus-5-5.' }]);
    assert.deepEqual(lines({ value: 'claudexor::codex=gpt-6-astra', layer: 'main', local: false }),
        [{ tone: 'muted', text: 'Thinks with gpt-6-astra (Main model).' }]);
    assert.deepEqual(lines({ value: 'qwen/qwen3-32b', layer: 'main', local: true }),
        [{ tone: 'muted', text: 'Thinks with Local model (Main model).' }]);
    assert.deepEqual(state({ model_route: { value: 'm', layer: 'consciousness', local: 'yes' } }).modelRoute,
        { value: 'm', layer: 'consciousness', local: false }, 'only a true local flag is local');
    for (const unknown of [null, undefined, {}, { value: '', layer: 'main' }, { value: 'm', layer: 'place' }]) {
        assert.equal(state({ ...base, model_route: unknown }).modelRoute, null);
        assert.deepEqual(lines(unknown), [], 'an unknown route says nothing');
    }
    assert.deepEqual(allowancePresentation(state({ ...base, enabled: false, model_route: { value: 'm', layer: 'main' } }), { now }).lines,
        [{ text: 'Consciousness is off: it starts nothing that counts here until it is started again.', tone: 'muted' }]);
});
