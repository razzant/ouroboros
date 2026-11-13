// Money beside a limit (storage_format.js `formatLimitUsd`): one rule for the limit cards of Costs, the consciousness
// allowance, the Costs rows of Settings' and Dashboard's lists and Main's budget.
import assert from 'node:assert/strict';
import test from 'node:test';

import { headerBudgetPresentation } from '../modules/chat_activity.js';
import { allowanceMeter, costDashboardPresentation } from '../modules/costs.js';
import { allowanceText } from '../modules/consciousness_words.js';
import { formatLimitUsd } from '../modules/storage_format.js';

test('a whole-dollar limit: the spend rounds up, what is left rounds down, and the two add up to the limit', () => {
    assert.equal(formatLimitUsd(42087.2, 42500), '$42,088');
    assert.equal(formatLimitUsd(42500 - 42087.2, 42500, 'left'), '$412');
    assert.equal(formatLimitUsd(42500, 42500, 'limit'), '$42,500');
    // A spend a hair under a dollar boundary is not a dollar more.
    assert.equal(formatLimitUsd(0.1 + 0.2 + 2.7, 10), '$3');
    const card = costDashboardPresentation({ total_calls: 1, accounting: {
        settled_usd: 42087.2, confirmed_usd: 42087.2, estimated_usd: 0, reserved_usd: 0,
        unresolved_upper_bound_usd: 0, unknown_unmetered: 0, limit_usd: 42500 } });
    assert.equal(card.figure, '$42,088 of $42,500');
    assert.equal(card.left, '$412 left');
});

test('a limit set with cents keeps cents for the limit, the spend and what is left', () => {
    assert.equal(formatLimitUsd(0.25, 0.25, 'limit'), '$0.25');
    assert.equal(formatLimitUsd(0.101, 0.25), '$0.11');
    assert.equal(formatLimitUsd(0.25 - 0.101, 0.25, 'left'), '$0.14');
    const meter = allowanceMeter({ spentUsd: 0.101, dailyUsd: 0.25 });
    assert.equal(`${meter.spent} ${meter.of}`, '$0.11 of $0.25');
    assert.equal(meter.left, '$0.14 left');
});

test('a positive amount below one dollar is said in words, on both sides of a whole limit', () => {
    assert.equal(formatLimitUsd(0.3, 20), 'less than $1');
    assert.equal(formatLimitUsd(0.4, 20, 'left'), 'less than $1');
    assert.equal(formatLimitUsd(0, 20), '$0');
    assert.equal(allowanceText({ spentUsd: 0.3, dailyUsd: 20 }), 'less than $1 of $20 in the last 24 hours');
    assert.equal(allowanceMeter({ spentUsd: 19.6, dailyUsd: 20 }).left, 'Less than $1 left');
});

test('an amount that is not known is the dash, never $0', () => {
    assert.equal(formatLimitUsd(null, 20), '—');
    assert.equal(formatLimitUsd(undefined, 20), '—');
    assert.equal(formatLimitUsd('', 20), '—');
    assert.equal(formatLimitUsd('not a number', 20), '—');
});

test('thousands are grouped, and Main\'s budget reads the same words as the cards', () => {
    assert.equal(formatLimitUsd(1234567.01, 2000000), '$1,234,568');
    assert.equal(formatLimitUsd(1234.5, 2000.5, 'limit'), '$1,234.50');
    assert.equal(headerBudgetPresentation({ spent_usd: 42087.2, budget_limit: 42500 }).label, '$42,088 of $42,500');
});
