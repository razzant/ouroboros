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
        assert.equal(verdict.label, 'Not checked'); // ratios are useful display, not an exact engine verdict
    });
}

test('model scope, account pin and stale evidence remain independent', () => {
    const spent = { used_ratio: 1, resets_at: '2030-01-01T02:00:00Z', applies_to_models: ['other-model'] };
    // The engine's "available" names this request only beside one measured every-model window (OB1851).
    const state = stateFor([spent, { id: 'primary', used_ratio: 0.2 }]);
    state.snapshot.availability_scope = { model: 'fable', requestKind: 'text', basis: 'quota_block' };
    state.snapshot.quota[0].availability = { state: 'available' };
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Available');
    // No fuzzy matcher: raw aliases/ratios cannot overturn the engine's verdict.
    state.snapshot.quota[0].constraints[0].applies_to_models = ['fable'];
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Available');
    state.snapshot.quota[0].availability = { state: 'exhausted', resets_at: spent.resets_at };
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Limit reached');
    state.snapshot.quota[0].subject.subject_id = 'sibling';
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
    state.snapshot.quota[0].subject.subject_id = 'chosen';
    state.snapshot.quota[0].freshness = 'stale';
    // A stale row's dated, unexpired engine block stays a live cap (OB1851); a stale positive does not.
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Limit reached');
    assert.equal(sessionRouteVerdict(row, state, Date.parse(spent.resets_at)).label, 'Not checked');
    state.snapshot.quota[0].availability = { state: 'available' };
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
});

test('one incomplete profile cannot certify or refuse the whole rotation pool', () => {
    const state = stateFor([{ used_ratio: 1 }]);
    state.snapshot.availability_scope = { model: 'fable', requestKind: 'text', basis: 'quota_block' };
    state.snapshot.quota.push({ subject: { harness: 'claude', subject_id: 'sibling' },
        freshness: 'fresh', availability: { state: 'exhausted' },
        constraints: [{ used_ratio: 1, resets_at: '2030-01-01T02:00:00Z' }] });
    const unpinned = { route: { ...row.route, credential_profile_id: '' } };
    assert.equal(sessionRouteVerdict(unpinned, state, nowMs).label, 'Not checked');
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
});


test('only an honored exact text scope may turn cached quota into an actor verdict', () => {
    const state = stateFor([{ used_ratio: 1, applies_to_models: ['fable'], freshness: 'fresh' }, { id: 'primary', used_ratio: 0.2 }]);
    state.snapshot.quota[0].availability = { state: 'available' }; // engine, not host ratio/matcher
    for (const scope of [undefined, {}, { model: null, requestKind: 'text', basis: 'quota_block' },
        { model: 'fable-extra', requestKind: 'text', basis: 'quota_block' },
        { model: 'fable', requestKind: 'image', basis: 'quota_block' }]) {
        state.snapshot.availability_scope = scope;
        assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Not checked');
    }
    state.snapshot.availability_scope = { model: 'fable', requestKind: 'text', basis: 'quota_block' };
    assert.equal(sessionRouteVerdict(row, state, nowMs).label, 'Available');
});

test('a default pin never borrows an unprofiled window, whatever the roster says', () => {
    // #244, as in Python inspection: a null/empty subject is an unprofiled route's
    // own window, never `claude-default`'s; only the exact subject answers.
    const state = stateFor([{ id: 'primary', used_ratio: 0.2 }]);
    const defaultRow = { route: { ...row.route, credential_profile_id: 'claude-default' } };
    state.snapshot.profiles.profiles[0].profile.profile_id = 'claude-default';
    state.snapshot.availability_scope = { model: 'fable', requestKind: 'text', basis: 'quota_block' };
    state.snapshot.quota[0].availability = { state: 'available' };
    for (const pools of [undefined, []]) {
        state.snapshot.profiles.accountPools = pools;
        for (const subject of [null, '', 'healthy-sibling']) {
            state.snapshot.quota[0].subject.subject_id = subject;
            assert.equal(sessionRouteVerdict(defaultRow, state, nowMs).label, 'Not checked', String(subject));
        }
    }
    state.snapshot.quota[0].subject.subject_id = 'claude-default';
    assert.equal(sessionRouteVerdict(defaultRow, state, nowMs).label, 'Available');
});
