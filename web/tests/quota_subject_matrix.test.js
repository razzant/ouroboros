// OB1851: the card reader and claudexor_observations.quota_availability judge
// one shared matrix (tests/fixtures/quota_subject_matrix, engine-computed rows).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { routeQuotaFact, sessionRouteVerdict } from '../modules/subagent_status_primitives.js';

const MATRIX = JSON.parse(readFileSync(new URL('../../tests/fixtures/quota_subject_matrix/matrix.json', import.meta.url)));
const CASES = Object.fromEntries(MATRIX.cases.map((item) => [item.name, item]));

// The passive status read carries the quota envelope under these names (docs/PASSIVE_QUOTA_READ.md).
const snapshotOf = (envelope) => ({ quota: envelope.snapshots, quota_absences: envelope.absences,
    ...(envelope.availability_scope ? { availability_scope: envelope.availability_scope } : {}) });

for (const item of MATRIX.cases) {
    test(`shared matrix: ${item.name}`, () => {
        const fact = routeQuotaFact(snapshotOf(item.envelope), item.harness, item.model, item.profile,
            Date.parse(item.now), item.request_kind);
        assert.equal(fact.state, item.expect.state);
        if ('resets_at' in item.expect) assert.equal(fact.resets_at, item.expect.resets_at);
        assert.equal(fact.known, fact.state !== 'unknown');
        assert.equal(fact.exhausted, ['exhausted', 'cooldown'].includes(fact.state));
    });
}

const account = (id) => ({ profile: { harness_id: 'codex', profile_id: id, enabled: true }, status: { verification: 'passed' } });

function cardState(item, profiles = ['a', 'b']) {
    return {
        catalogKnown: true, accountsKnown: true, quotaKnown: true,
        snapshot: {
            ...snapshotOf(item.envelope),
            harnesses: [{ id: 'codex', status: 'ok', enabled: true, models: [{ id: item.model }] }],
            profiles: { harnessAccounts: [], profiles: profiles.map(account),
                accountPools: [{ harness_id: 'codex', next_up: { kind: 'profile', profile_id: profiles[0] } }] },
        },
    };
}

const card = (item, pin = item.profile) => sessionRouteVerdict({ route: { kind: 'agent_session',
    target_id: `codex=${item.model}`, credential_profile_id: pin } }, cardState(item), Date.parse(item.now));

test('Agents card words follow the subject verdict, Auto and pinned', () => {
    assert.equal(card(CASES.split_primary20_secondary100).label, 'Limit reached');
    assert.equal(card(CASES.split_primary20_secondary100).text, 'codex · all known accounts reached a limit');
    assert.equal(card(CASES.split_primary20_secondary100, 'a').text, 'codex · pinned account a limit reached');
    assert.equal(card(CASES.available_beside_blocked_subject).label, 'Available');
    assert.equal(card(CASES.available_beside_blocked_subject, 'b').label, 'Limit reached');
    assert.equal(card(CASES.available_beside_blocked_subject, 'a').label, 'Available');
    for (const name of ['absence_only_alternative', 'model_scoped_available_beside_blocked_subject',
        'image_only_window_for_text', 'early_reset_repoll', 'fresh_available_beside_stale_nonblock']) {
        const verdict = card(CASES[name]);
        assert.equal(verdict.label, 'Not checked', name);
        assert.equal(verdict.text, 'codex · live availability not checked', name);
    }
    assert.equal(card(CASES.absence_only_pin).text, 'codex · pinned account ready; quota availability not proven');
    assert.equal(card(CASES.strict_pin_blocked_beside_free_sibling).label, 'Limit reached');
    assert.equal(card(CASES.strict_pin_own_available).label, 'Available');
});

test('an unpinned card judges only the accounts carrying the model, absences included', () => {
    const item = CASES.absence_only_alternative;  // a: dated block, b: absence only
    const state = cardState(item);
    state.snapshot.harnesses[0].models = [{ id: item.model, credential_profile_id: 'a' }];
    const row = { route: { kind: 'agent_session', target_id: `codex=${item.model}`, credential_profile_id: '' } };
    assert.equal(sessionRouteVerdict(row, state, Date.parse(item.now)).label, 'Limit reached',
        'a non-carrier gap cannot hide the only carrier\'s block');
    state.snapshot.harnesses[0].models.push({ id: item.model, credential_profile_id: 'b' });
    assert.equal(sessionRouteVerdict(row, state, Date.parse(item.now)).label, 'Not checked');
});
