import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizeLogEvent } from '../modules/log_events.js';

// #1578: record passport v2 usage rows distinguish "not reported" from a reported 0;
// task_done ledger totals and historical rows keep their own unchanged view.
const meta = (evt) => summarizeLogEvent(evt).meta.join(' · ');

for (const type of ['llm_usage', 'llm_round', 'llm_round_finished']) {
    test(`${type}: v2 shows unknown, reported zero and partial counters independently`, () => {
        assert.match(meta({ type, record_contract_version: 2 }), /tokens not reported/);
        assert.match(meta({ type, record_contract_version: 2, prompt_tokens: null, completion_tokens: null }),
            /tokens not reported/);
        assert.match(meta({ type, record_contract_version: 2, prompt_tokens: 0, completion_tokens: 0 }), /0→0 tok/);
        assert.match(meta({ type, record_contract_version: 2, prompt_tokens: 420, completion_tokens: null }),
            /420→\? tok/);
        const invalid = meta({ type, record_contract_version: 2, prompt_tokens: '', completion_tokens: -1 });
        assert.match(invalid, /tokens not reported/);
        assert.doesNotMatch(invalid, /NaN|0→/);
        // Cache counters add no noise to the summary line, known or not.
        assert.doesNotMatch(meta({ type, record_contract_version: 2, prompt_tokens: 7, completion_tokens: 3,
            cached_tokens: null, cache_write_tokens: null }), /cache|unknown/);
    });
}

for (const type of ['llm_usage', 'llm_round', 'llm_round_finished']) {
    for (const stamp of [{}, { record_contract_version: 1 }]) {
        const label = stamp.record_contract_version ? 'v1' : 'unstamped';
        test(`${type}: ${label} zeros stay ambiguous, missing is not reported, positives are kept`, () => {
            // LEGACY_RULE: a legacy 0 may be an unreported counter, never a proven zero.
            const partial = { type, ...stamp, prompt_tokens: 12, completion_tokens: 0 };
            const before = { ...partial };
            const shown = meta(partial);
            assert.match(shown, /12→0\? tok \(legacy 0 may be unreported\)/);
            assert.doesNotMatch(shown, /12→0 tok/);
            assert.deepEqual(partial, before);  // the row itself is never relabelled
            assert.match(meta({ type, ...stamp, prompt_tokens: 0, completion_tokens: 0 }),
                /0\?→0\? tok \(legacy 0 may be unreported\)/);
            const positive = meta({ type, ...stamp, prompt_tokens: 12, completion_tokens: 3 });
            assert.match(positive, /12→3 tok/);
            assert.doesNotMatch(positive, /legacy|\?|not reported|cache/);
            assert.match(meta({ type, ...stamp, prompt_tokens: 12 }), /12→\? tok/);
            assert.match(meta({ type, ...stamp }), /tokens not reported/);
            assert.match(meta({ type, ...stamp, prompt_tokens: -1, completion_tokens: '7' }), /tokens not reported/);
            // Absent cache fields add no noise on legacy rows either.
            assert.doesNotMatch(meta({ type, ...stamp, prompt_tokens: 5, completion_tokens: 2 }), /cache|unknown/);
        });
    }
}

test('a session row shows its whole input from its split, unknown when the split has no total', () => {
    const session = { type: 'llm_usage', record_contract_version: 2, prompt_tokens: 596, completion_tokens: 289411,
        cached_tokens: 146374357, input_token_usage: { total_tokens: 146374953, cache_read_tokens: 146374357,
            cache_write_tokens: null } };
    assert.match(meta(session), /146374953→289411 tok/);
    assert.doesNotMatch(meta(session), /596→/);
    const unsplit = { ...session, input_token_usage: { total_tokens: null, cache_read_tokens: null, cache_write_tokens: null } };
    assert.match(meta(unsplit), /\?→289411 tok/);
    // A direct call has no split: its prompt_tokens already is the whole input.
    assert.match(meta({ type: 'llm_usage', record_contract_version: 2, prompt_tokens: 7, completion_tokens: 3 }), /7→3 tok/);
});

test('task_done ledger totals never borrow the usage passport or cache noise', () => {
    const done = { type: 'task_done', task_id: 't1', status: 'completed', prompt_tokens: 0, completion_tokens: 0 };
    assert.doesNotMatch(meta(done), /tok|not reported|unknown|cache/);
    const totals = meta({ ...done, prompt_tokens: 1200, completion_tokens: 300 });
    assert.match(totals, /1200→300 tok/);
    assert.doesNotMatch(totals, /cache|unknown|not reported/);
    // The task_done formatter itself is unchanged: no legacy marker on its own totals.
    const partial = meta({ ...done, prompt_tokens: 12, completion_tokens: 0 });
    assert.match(partial, /12→0 tok/);
    assert.doesNotMatch(partial, /legacy|\?|not reported/);
    // A sum no row reported (a session without its input split) is unknown, never a zero.
    assert.match(meta({ ...done, prompt_tokens: null, completion_tokens: 300 }), /\?→300 tok/);
    assert.doesNotMatch(meta({ ...done, prompt_tokens: null, completion_tokens: null }), /tok/);
});
