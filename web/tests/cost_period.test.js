import assert from 'node:assert/strict';
import test from 'node:test';

import { COST_PERIODS, compactTokens, costPeriodPresentation, mountCostPeriod } from '../modules/cost_period.js';

// The period view as `GET /api/cost-breakdown?period=7d` answers it (ouroboros/gateway/cost_period.py).
const VIEW = {
    available: true, period: '7d', since: '2026-11-05T07:04:00+00:00', until: '2026-11-12T07:04:00+00:00',
    tariff_observed_at: '2026-11-12T11:00:58+00:00',
    models: [
        { model: 'anthropic/claude-opus-5.5', calls: 10741, cost: 6055.13, open_rows: 14, unpriced_rows: 165, unshared_cost: 0,
            tokens: { cache_write: 538_000_000, cache_read: 5_500_000_000, output: 26_400_000, uncached_input: 29_800_000 },
            tokens_shared: { cache_write: 538_000_000, cache_read: 5_500_000_000, output: 26_400_000, uncached_input: 29_800_000 },
            by_class: { cache_write: 4306.97, cache_read: 1101.77, output: 527.36, uncached_input: 119.03 }, write_price_in_reads: 40,
            write_price_lifetime: '1h' },
        // A subscription reports no write count: its unread input is counted apart, as `period_models` answers it.
        { model: 'claudexor::codex=gpt-6-astra', calls: 56724, cost: null, open_rows: 0, unpriced_rows: 56724, unshared_cost: 0,
            tokens: { cache_write: 0, cache_read: 34_500_000_000, output: 210_000_000, uncached_input: 0 },
            by_class: null, write_price_in_reads: null, uncounted_calls: 56724, uncounted_input_tokens: 2_900_000_000 },
        { model: 'vendor/unlisted', calls: 3, cost: 0.4, open_rows: 0, unpriced_rows: 0, unshared_cost: 0.4,
            tokens: { cache_write: 50, cache_read: 100, output: 7, uncached_input: 350 }, by_class: null, write_price_in_reads: null,
            uncounted_calls: 1, uncounted_input_tokens: 2_400 },
        { model: 'vendor/cheap', calls: 2, cost: 0.008, open_rows: 0, unpriced_rows: 0, unshared_cost: 0.004, unshared_calls: 1,
            tokens: { cache_write: 0, cache_read: 0, output: 10, uncached_input: 90 }, tokens_shared: { cache_write: 0, cache_read: 0, output: 4, uncached_input: 40 },
            by_class: { cache_write: 0, cache_read: 0, output: 0.002, uncached_input: 0.002 } },
        { model: 'vendor/never-called', calls: 0, cost: 0, open_rows: 0, unpriced_rows: 0, unshared_cost: 0, tokens: {}, by_class: null },
    ],
    cache_writes: [
        { class: 'entered_model', detail: 'gpt-6-astra', calls: 204, write_tokens: 149_000_000, write_cost: 1250.93, unshared_calls: 0 },
        { class: 'history_rewritten', detail: 'compaction, low_retry', calls: 444, write_tokens: 146_000_000, write_cost: 1189.29, unshared_calls: 0 },
        { class: 'system_changed', detail: 'my story and identity', calls: 9, write_tokens: 1_200_000, write_cost: 9.6, unshared_calls: 2,
            shared_write_tokens: 900_000 },
        { class: 'not_recorded', detail: '', calls: 5, write_tokens: 700, write_cost: 0, unshared_calls: 5 },
    ],
};

test('a period shows each model\'s recorded cost by class, and tokens alone where no cost can be shared', () => {
    const shown = costPeriodPresentation(VIEW);
    assert.equal(shown.state, 'available');
    assert.deepEqual(shown.models.map((model) => model.name),
        ['anthropic/claude-opus-5.5', 'claudexor::codex=gpt-6-astra', 'vendor/unlisted', 'vendor/cheap']);
    const [opus, codex, unlisted, cheap] = shown.models;
    assert.equal(opus.sum, '$6,055.13 · 10,741 calls');
    assert.equal(opus.split, true);
    assert.deepEqual(opus.parts.map((part) => part.text), [
        '$4,306.97 · 71% · 538M tokens', '$1,101.77 · 18% · 5.5B tokens', '$527.36 · 9% · 26.4M tokens', '$119.03 · 2% · 29.8M tokens']);
    assert.ok(Math.abs(opus.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);  // the bar is the whole shared cost
    assert.deepEqual(opus.notes, ['14 calls are still open.', '165 ended without a price.',
        'One token written for 1 hour costs 40 read tokens at this model’s base price.']);
    // A subscription model has no cost to share: its tokens by class are the fact shown, and its bar is by tokens and says so.
    assert.equal(codex.sum, 'no recorded cost · 56,724 calls');  // unknown, never $0.00
    assert.deepEqual([codex.split, codex.bar], [false, true]);
    assert.deepEqual(codex.parts.map((part) => part.text),
        ['0 tokens · 0%', '34.5B tokens · 92%', '210M tokens · 1%', '0 tokens · 0%', '2.9B tokens · 8%']);
    assert.equal(codex.parts.at(-1).label, 'Unclassified input');  // never shown as 99% reads and no uncached input
    assert.deepEqual(codex.notes, ['56,724 calls reported no complete cache counts: 2.9B input tokens are in no class.',
        '56,724 ended without a price.', 'The bar is by tokens: there is no recorded cost.']);
    assert.ok(Math.abs(codex.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);
    // A priced call with no price list at hand: the amount stands, the split is said to be absent, never invented.
    assert.deepEqual([unlisted.split, unlisted.bar], [false, true]);
    assert.deepEqual(unlisted.notes, ['Not split by class: these calls have no price list, token counts or named cache lifetime at hand.',
        '1 call reported no complete cache counts: 2k input tokens are in no class.', 'The bar is by tokens: the recorded cost could not be split.']);
    // Every positive amount left out is said, however small: the parts and it add up to the sum shown.
    assert.deepEqual(cheap.notes, ['Not split: no price list, token counts or named cache lifetime for those calls.']);
    // By money every figure of a part is of the same calls, and what was not split is a part of the head's sum.
    assert.deepEqual(cheap.parts.map((part) => part.text), ['$0.00 · 0% · 0 tokens', '$0.00 · 0% · 0 tokens',
        '$0.0020 · 25% · 4 tokens', '$0.0020 · 25% · 40 tokens', '$0.0040 · 50% · 1 call']);
    assert.ok(Math.abs(cheap.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);
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
        ['204', '149M', '$1,250.93'], ['444', '146M', '$1,189.29'], ['9', '1.2M', '$9.60 for 900k tokens · 2 not split'], ['5', '700', '—']]);
});

test('the hourly write note distinguishes zero, a small price, and an unavailable comparison', () => {
    for (const [ratio, reads] of [[0, '0'], [0.002, 'less than 0.1'], [null, null]]) {
        const row = { ...VIEW.models[0], write_price_in_reads: ratio };
        const [model] = costPeriodPresentation({ available: true, models: [row] }).models;
        const notes = model.notes.filter((note) => note.startsWith('One token written'));
        assert.deepEqual(notes, reads === null ? [] : [`One token written for 1 hour costs ${reads} read tokens at this model’s base price.`]);
        assert.equal(model.notes.includes('The base price of a write for 1 hour cannot be expressed in read tokens.'), reads === null);
    }
});

test('a write’s price in read tokens is promised only where one is known, and a missing price list is said once', () => {
    const notes = (row) => costPeriodPresentation({ available: true, models: [{ model: 'vendor/m', calls: 4, cost: 0.125, open_rows: 0,
        unpriced_rows: 0, tokens: { cache_write: 900, cache_read: 100, output: 0, uncached_input: 0 }, write_price_lifetime: '1h', ...row }] }).models[0].notes;
    const ratioWords = (list) => list.filter((note) => /read tokens/.test(note));
    // None of the model's calls could be shared (the catalogue holds no price list, or no price for the hourly write): that is
    // the one fact, said where the split is said, and no ratio and no second sentence about one follows.
    const none = notes({ by_class: null, unshared_cost: 0.125, unshared_calls: 4, write_price_in_reads: null });
    assert.deepEqual(none.filter((note) => /price list/.test(note)), ['Not split by class: these calls have no price list, token counts or named cache lifetime at hand.']);
    assert.deepEqual(ratioWords(none), []);
    // Shared by a price list but a write cannot be set against a read (a read priced at nothing, routes that disagree): that
    // is a fact of its own, still said, and still no ratio.
    const shared = notes({ by_class: { cache_write: 0.1, cache_read: 0, output: 0, uncached_input: 0.025 }, unshared_cost: 0, unshared_calls: 0,
        tokens_shared: { cache_write: 900, cache_read: 100, output: 0, uncached_input: 0 }, write_price_in_reads: null });
    assert.deepEqual(ratioWords(shared), ['The base price of a write for 1 hour cannot be expressed in read tokens.']);
    assert.deepEqual(shared.filter((note) => /price list/.test(note)), []);
    // Part of the cost shared and part not: each fact is its own sentence.
    const part = notes({ by_class: { cache_write: 0.05, cache_read: 0, output: 0, uncached_input: 0.025 }, unshared_cost: 0.05, unshared_calls: 2,
        tokens_shared: { cache_write: 400, cache_read: 100, output: 0, uncached_input: 0 }, write_price_in_reads: null });
    assert.deepEqual([part.filter((note) => /price list/.test(note)).length, ratioWords(part).length], [1, 1]);
    // A known ratio is said with its lifetime, zero and a small price as they are, never as an unknown.
    for (const [ratio, reads] of [[40, '40'], [0, '0'], [0.002, 'less than 0.1']]) {
        assert.deepEqual(ratioWords(notes({ by_class: { cache_write: 0.1, cache_read: 0.025 }, unshared_cost: 0, write_price_in_reads: ratio })),
            [`One token written for 1 hour costs ${reads} read tokens at this model’s base price.`]);
    }
});

test('a standard and a priority payment show only the standard tokens beside the shared dollars', async () => {
    const tiny = 0.00000025;
    const view = { available: true, models: [{
        model: 'vendor/mixed', calls: 2, cost: 1 + tiny, tokens: { cache_write: 100 }, tokens_shared: { cache_write: 10 },
        by_class: { cache_write: tiny }, unshared_calls: 1, unshared_cost: 1, write_price_lifetime: '1h', write_price_in_reads: null,
    }], cache_writes: [{ class: 'not_recorded', calls: 2, write_tokens: 100, shared_write_tokens: 10, write_cost: tiny, unshared_calls: 1 }] };
    const { models: [model], writes: [write] } = costPeriodPresentation(view);
    assert.equal(model.sum, '$1.00 · 2 calls');
    assert.equal(model.parts[0].text, '<$0.0001 · <1% · 10 tokens');
    assert.deepEqual([model.parts.at(-1).label, model.parts.at(-1).text], ['Not split', '$1.00 · >99% · 1 call']);
    assert.equal(write.cost, '<$0.0001 for 10 tokens · 1 not split');
    assert.equal(write.tokens, '100');

    const cells = {};
    const host = { querySelector: (selector) => (cells[selector] ??= {}), querySelectorAll: () => [], addEventListener() {} };
    await mountCostPeriod(host, { load: async () => view }).reload();
    const html = cells['.costs-period-models'].innerHTML;
    assert.ok(html.includes('&lt;$0.0001 · &lt;1% · 10 tokens'));
    assert.ok(cells['.costs-period-writes-host'].innerHTML.includes('&lt;$0.0001 for 10 tokens · 1 not split'));
    const width = html.match(/data-cost-class="cache_write" style="--share:([^"]+)"/);
    assert.ok(width && Number(width[1]) > 0 && Number(width[1]) < 0.005);
});

test('one hourly write distinguishes a tiny payment, measured zero and missing money', () => {
    for (const [cost, sum, writeCost] of [[0.00000025, '<$0.0001', '<$0.0001'], [0, '$0.00', '$0.00'], [null, 'no recorded cost', '—']]) {
        const view = { available: true, models: [{ model: 'vendor/model', calls: 1, cost, tokens: { cache_write: 2 },
            tokens_shared: { cache_write: 2 }, by_class: cost ? { cache_write: cost } : null }],
            cache_writes: [{ class: 'not_recorded', calls: 1, write_tokens: 2, write_cost: cost, unshared_calls: cost === null ? 1 : 0 }] };
        const { models: [model], writes: [write] } = costPeriodPresentation(view);
        assert.equal(model.sum, `${sum} · 1 call`);
        assert.equal(write.cost, writeCost);
        if (cost) assert.equal(model.parts[0].text, '<$0.0001 · 100% · 2 tokens');
    }
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

test('the page leaving ends the period read out and drops its answer, and a read already out is kept when asked', async () => {
    const reads = [];
    const cells = {};
    const host = { innerHTML: '', querySelector: (selector) => (cells[selector] ??= { textContent: '', innerHTML: '' }), querySelectorAll: () => [],
        addEventListener: () => {} };
    const card = mountCostPeriod(host, { load: (period, signal) => new Promise((resolve) => reads.push({ period, signal, resolve })) });
    assert.equal(card.leave(), false, 'nothing is out before the first read');
    const first = card.reload();
    card.reload({ ifIdle: true });
    assert.equal(reads.length, 1, 'a read out is kept when the caller asks for one only if none is');
    assert.equal(cells['.costs-period-status'].textContent, 'Counting…');
    assert.equal(card.leave(), true, 'the page leaves with a read out');
    assert.equal(reads[0].signal.aborted, true, 'its transport is released at once');
    assert.equal(cells['.costs-period-status'].textContent, '', 'and the card does not say it is counting');
    reads[0].resolve({ available: true, models: [{ model: 'late/model', calls: 1, cost: 1 }], cache_writes: [] });
    await first;
    assert.equal(cells['.costs-period-models'].innerHTML, '', 'an answer that comes after leaving paints nothing');
    assert.equal(card.leave(), false, 'nothing is out now');
    const second = card.reload({ ifIdle: true });
    assert.equal(reads.length, 2, 'an idle card reads when asked');
    reads[1].resolve({ available: true, models: [{ model: 'now/model', calls: 1, cost: 1 }], cache_writes: [] });
    await second;
    assert.match(cells['.costs-period-models'].innerHTML, /now\/model/);
    assert.equal(card.leave(), false, 'an answered read is not cut short by leaving');
    assert.equal(cells['.costs-period-models'].innerHTML.includes('now/model'), true, 'and its answer stays');
});

test('a later period choice abandons the read it replaces', async () => {
    const signals = [];
    const host = { innerHTML: '', querySelector: () => ({ textContent: '', innerHTML: '' }), querySelectorAll: () => [],
        addEventListener: () => {} };
    const card = mountCostPeriod(host, { load: (period, signal) => { signals.push(signal); return new Promise(() => {}); } });
    card.reload();
    card.reload();
    assert.equal(signals.length, 2);
    assert.equal(signals[0].aborted, true);
    assert.equal(signals[1].aborted, false);
});

test('a recorded cost of nothing is said, never called unsplit', () => {
    const view = { available: true, models: [{ model: 'local-model', calls: 4, cost: 0, open_rows: 0, unpriced_rows: 0, unshared_cost: 0,
        tokens: { cache_write: 0, cache_read: 0, output: 9, uncached_input: 0 }, by_class: null, uncounted_calls: 4, uncounted_input_tokens: 30 }] };
    const [local] = costPeriodPresentation(view).models;
    assert.match(local.notes.at(-1), /^The bar is by tokens: the recorded cost is \$0\.00\.$/);
    assert.deepEqual(local.parts.filter((part) => part.share > 0).map((part) => [part.key, Math.round(part.share * 100)]),
        [['output', 23], ['unclassified', 77]]);  // 30 input tokens in no class beside 9 output, not 100% output
});
