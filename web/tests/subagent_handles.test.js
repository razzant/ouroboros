// A roster row is NAMED by a projection of its route. The Python owner is
// ouroboros/configured_subagents.py; this module pins the JS twin against the
// SAME table (tests/test_subagent_handles.py reads it too).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { rosterHandles, sameEngineAs, subagentHandle } from '../modules/route_editor_primitives.js';

const PARITY = JSON.parse(readFileSync(
    new URL('./fixtures/subagent_handle_parity.json', import.meta.url), 'utf8')).rosters;

for (const roster of PARITY) {
    test(`handle parity with Python: ${roster.case}`, () => {
        const inherited = roster.global_processing;
        const labels = rosterHandles(roster.items, inherited);
        roster.expected.forEach((want, index) => {
            const row = roster.items[index];
            assert.equal(subagentHandle(row, inherited), want.handle);
            assert.equal(labels.get(row.subagent_id), want.roster);
            assert.equal(sameEngineAs(roster.items, index, inherited), want.same_engine_as ?? -1);
        });
    });
}
