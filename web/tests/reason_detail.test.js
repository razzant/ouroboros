import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { taskReasonDetail, taskReasonPhrase } from '../modules/log_events.js';

// A degraded delivery used to name one generic cause on every card. The record
// keeps the machine code; the card says what actually happened.

test('a typed cause is stated in the owner\'s words', () => {
    assert.equal(
        taskReasonDetail({ reason_code: 'plan_review_advisory' }),
        'Reason: plan review never closed; the work continued under advisory enforcement',
    );
    assert.equal(
        taskReasonDetail({ reason_code: 'delivery_control_degraded' }),
        'Reason: delivery finished in a degraded control state',
    );
});

test('an unknown cause stays raw rather than becoming a wrong sentence', () => {
    assert.equal(taskReasonDetail({ reason_code: 'some_future_code' }), 'Reason: some_future_code');
    assert.equal(taskReasonPhrase('some_future_code'), 'some_future_code');
});

test('no cause and an owner-requested stop both render nothing', () => {
    assert.equal(taskReasonDetail({}), '');
    assert.equal(taskReasonDetail({ reason_code: '' }), '');
    // The soft stop is a SUCCESS and carries its own marker instead.
    assert.equal(taskReasonDetail({ reason_code: 'owner_requested_finalization' }), '');
});

test('every typed cause the loop can record has a sentence', () => {
    // Cross-language completeness: the reachable literals live in the Python
    // loop, the sentences live here, and a new cause must not silently fall
    // back to its machine code on the card.
    const loop = readFileSync(new URL('../../ouroboros/loop.py', import.meta.url), 'utf8');
    const literals = new Set(
        [...loop.matchAll(/degraded_reason\s*=\s*"([a-z_]+)"/g)].map((m) => m[1]),
    );
    assert.ok(literals.size >= 3, `expected the known typed causes, saw ${[...literals]}`);
    for (const code of literals) {
        assert.notEqual(
            taskReasonPhrase(code), code,
            `no owner-facing sentence for degraded_reason "${code}" — add one to TASK_REASON_PHRASES`,
        );
    }
});
