import assert from 'node:assert/strict';
import test from 'node:test';

import { sessionRouteVerdict } from '../modules/subagent_status_primitives.js';
import { ROUTE_KIND_AGENT_SESSION } from '../modules/route_editor_primitives.js';

// The card judges a route the way dispatch admits it (`subagent_route_health.route_health`):
// the aggregate doctor `status` describes only the default credential store and never
// refuses, while the owner's `enabled=false` switch does.
const row = { subagent_id: 'builder', route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-5.6-sol-high' } };

function state(harness) {
    return {
        catalogKnown: true, accountsKnown: true, quotaKnown: true,
        snapshot: {
            harnesses: [{ id: 'codex', models: [{ id: 'gpt-5.6-sol-high' }], ...harness }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'work', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [],
        },
    };
}

test('the aggregate doctor status never paints an unpinned session Unavailable', () => {
    const healthy = sessionRouteVerdict(row, state({ status: 'ok', enabled: true }));
    for (const status of ['degraded', 'error', 'unavailable']) {
        const verdict = sessionRouteVerdict(row, state({ status, enabled: true }));
        assert.notEqual(verdict.label, 'Unavailable', status);
        assert.deepEqual(verdict, healthy, `doctor status ${status} changes nothing dispatch would not`);
    }
});

test("the owner's enabled=false switch still refuses, whatever the doctor says", () => {
    for (const status of ['ok', 'degraded']) {
        assert.deepEqual(sessionRouteVerdict(row, state({ status, enabled: false })),
            { label: 'Unavailable', tone: 'warn', text: 'codex · currently unavailable', reason: '' });
    }
});
