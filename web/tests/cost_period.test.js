import assert from 'node:assert/strict';
import test from 'node:test';

import { COST_PERIODS, compactTokens, costPeriodPresentation } from '../modules/cost_period.js';

// The period view as `GET /api/cost-breakdown?period=7d` answers it (ouroboros/gateway/cost_period.py).
const VIEW = {
    available: true, period: '7d', since: '2026-11-05T07:04:00+00:00', until: '2026-11-12T07:04:00+00:00',
    tariff_observed_at: '2026-11-12T11:00:58+00:00',
    models: [
        { model: 'anthropic/claude-opus-5.5', calls: 10741, cost: 6055.13, open_rows: 14, unpriced_rows: 165, unshared_cost: 0,
            tokens: { cache_write: 538_000_000, cache_read: 5_500_000_000, output: 26_400_000, uncached_input: 29_800_000 },
            by_class: { cache_write: 4306.97, cache_read: 1101.77, output: 527.36, uncached_input: 119.03 }, write_price_in_reads: 40,
            write_price_lifetime: '1h' },
        { model: 'claudexor::codex=gpt-6-astra', calls: 56724, cost: 0, open_rows: 0, unpriced_rows: 0, unshared_cost: 0,
            tokens: { cache_write: 0, cache_read: 34_500_000_000, output: 210_000_000, uncached_input: 2_900_000_000 },
            by_class: null, write_price_in_reads: null },
        { model: 'vendor/unlisted', calls: 3, cost: 0.4, open_rows: 0, unpriced_rows: 0, unshared_cost: 0.4,
            tokens: { cache_write: 50, cache_read: 100, output: 7, uncached_input: 350 }, by_class: null, write_price_in_reads: null },
        { model: 'vendor/never-called', calls: 0, cost: 0, open_rows: 0, unpriced_rows: 0, unshared_cost: 0, tokens: {}, by_class: null },
    ],
    cache_writes: [
        { class: 'entered_model', detail: 'gpt-6-astra', calls: 204, write_tokens: 149_000_000, write_cost: 1250.93, unshared_calls: 0 },
        { class: 'history_rewritten', detail: 'compaction, low_retry', calls: 444, write_tokens: 146_000_000, write_cost: 1189.29, unshared_calls: 0 },
        { class: 'system_changed', detail: 'my story and identity', calls: 9, write_tokens: 1_200_000, write_cost: 9.6, unshared_calls: 2 },
        { class: 'not_recorded', detail: '', calls: 5, write_tokens: 700, write_cost: 0, unshared_calls: 5 },
    ],
};

test('a period shows each model\'s recorded cost by class, and tokens alone where no cost can be shared', () => {
    const shown = costPeriodPresentation(VIEW);
    assert.equal(shown.state, 'available');
    assert.deepEqual(shown.models.map((model) => model.name), ['anthropic/claude-opus-5.5', 'claudexor::codex=gpt-6-astra', 'vendor/unlisted']);
    const [opus, codex, unlisted] = shown.models;
    assert.equal(opus.sum, '$6,055.13 · 10,741 calls');
    assert.equal(opus.split, true);
    assert.deepEqual(opus.parts.map((part) => part.text), [
        '$4,306.97 · 71% · 538M tokens', '$1,101.77 · 18% · 5.5B tokens', '$527.36 · 9% · 26.4M tokens', '$119.03 · 2% · 29.8M tokens']);
    assert.ok(Math.abs(opus.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);  // the bar is the whole shared cost
    assert.deepEqual(opus.notes, ['14 calls are still open.', '165 ended without a price.', 'One token written for 1 hour costs 40 read tokens here.']);
    // A subscription model has no cost to share: its tokens by class are the fact shown, and its bar is by tokens and says so.
    assert.deepEqual([codex.split, codex.bar], [false, true]);
    assert.deepEqual(codex.parts.map((part) => part.text), ['0 tokens · 0%', '34.5B tokens · 92%', '210M tokens · 1%', '2.9B tokens · 8%']);
    assert.deepEqual(codex.notes, ['The bar is by tokens: there is no recorded cost to split.']);
    assert.ok(Math.abs(codex.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);
    // A priced call with no price list at hand: the amount stands, the split is said to be absent, never invented.
    assert.deepEqual([unlisted.split, unlisted.bar], [false, true]);
    assert.deepEqual(unlisted.notes, ['Not split by class: these calls have no price list, token counts or named cache lifetime at hand.',
        'The bar is by tokens: there is no recorded cost to split.']);
    assert.match(shown.note, /share of the recorded cost, not the provider’s invoice lines/);
    assert.match(shown.note, /as recorded on 2026-11-12/);
    assert.match(shown.note, /delegated subscription sessions are not listed here/);
});

test('cache writes read by what they followed, in the fact\'s own words', () => {
    const { writes } = costPeriodPresentation(VIEW);
    assert.deepEqual(writes.map((row) => row.label), [
        'A running task switched to this model after gpt-6-astra',
        'Earlier history was rewritten (compaction, low retry)',
        'A part of the request start changed (my story and identity)',
        'Not recorded',
    ]);
    assert.deepEqual(writes.map((row) => [row.calls, row.tokens, row.cost]), [
        ['204', '149M', '$1,250.93'], ['444', '146M', '$1,189.29'], ['9', '1.2M', '$9.60 · 2 not split'], ['5', '700', '—']]);
});

test('an unavailable or empty period says so and invents nothing', () => {
    for (const view of [null, undefined, { available: false, period: '7d', error_code: 'ledger_unavailable' }]) {
        const shown = costPeriodPresentation(view);
        assert.equal(shown.state, 'unavailable');
        assert.deepEqual([shown.models, shown.writes], [[], []]);
        assert.match(shown.note, /could not be read/);
    }
    const empty = costPeriodPresentation({ available: true, period: '24h', models: [], cache_writes: [] });
    assert.equal(empty.state, 'empty');
    assert.equal(empty.note, 'No model calls started in this period.');
});

test('token counts keep a glance\'s precision, and the periods are the three the endpoint reads', () => {
    assert.deepEqual([812, 630_400, 12_400_000, 345_000_000, 1_200_000_000, 0, undefined].map(compactTokens),
        ['812', '630k', '12.4M', '345M', '1.2B', '0', '0']);
    assert.deepEqual(COST_PERIODS.map(([key]) => key), ['24h', '7d', '30d']);
});
