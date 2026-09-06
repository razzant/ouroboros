import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { quotaConstraintFact } from '../modules/claudexor_status_store.js';
import { quotaSummary } from '../modules/harness_accounts.js';
import { sessionRouteVerdict } from '../modules/subagent_status_primitives.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/quota_window_facts.json', import.meta.url), 'utf8'));
const nowMs = Date.parse(fixture.now);
const row = { route: { kind: 'agent_session', target_id: 'claude=fable', credential_profile_id: 'chosen' } };

function stateFor(constraints) {
    return {
        catalogKnown: true, accountsKnown: true, quotaKnown: true,
        snapshot: {
            harnesses: [{ id: 'claude', enabled: true, status: 'ok', models: ['fable'] }],
            profiles: { profiles: [{
                profile: { harness_id: 'claude', profile_id: 'chosen', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [{ subject: { harness: 'claude', subject_id: 'chosen' },
                freshness: 'fresh', constraints }],
        },
    };
}

for (const example of fixture.cases) {
    test(`quota evidence: ${example.name}`, () => {
        assert.deepEqual(quotaConstraintFact(example.constraint, nowMs), {
            exhausted: example.exhausted, resetsAt: example.resetsAt, unknown: example.unknown,
        });
        const state = stateFor([example.constraint]);
        const summary = quotaSummary(state.snapshot.quota, 'claude', 'chosen', { nowMs });
        assert.equal(summary.exhausted, example.exhausted);
        if (example.unknown) {
            assert.match(summary.label, /100% used · availability not proven/);
            assert.equal(summary.tone, 'muted');
        }
        const verdict = sessionRouteVerdict(row, state, nowMs);
        assert.equal(verdict.label, example.exhausted ? 'Limit reached' : example.unknown ? 'Not checked' : 'Available');
    });
}

test('model scope, account pin and stale evidence remain independent', () => {
    const spent = { used_ratio: 1, resets_at: '2030-01-01T02:00:00Z', applies_to_models: ['other-model'] };
    const state = stateFor([spent]);
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Available');
    state.snapshot.quota[0].constraints[0].applies_to_models = ['fable'];
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Limit reached');
    state.snapshot.quota[0].subject.subject_id = 'sibling';
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
    state.snapshot.quota[0].subject.subject_id = 'chosen';
    state.snapshot.quota[0].freshness = 'stale';
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
});

test('one incomplete profile cannot certify or refuse the whole rotation pool', () => {
    const state = stateFor([{ used_ratio: 1 }]);
    state.snapshot.quota.push({ subject: { harness: 'claude', subject_id: 'sibling' },
        freshness: 'fresh', constraints: [{ used_ratio: 1, resets_at: '2030-01-01T02:00:00Z' }] });
    const unpinned = { route: { ...row.route, credential_profile_id: '' } };
    assert.equal(sessionRouteVerdict(unpinned, state, nowMs).label, 'Not checked');
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
});
