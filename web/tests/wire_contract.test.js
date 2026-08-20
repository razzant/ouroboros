// The ONE test that crosses the endpoint -> module WIRE.
//
// Every other web test in this range exercises a PURE PROJECTOR: it hands a
// hand-written object to a render function and checks the string. That is why
// three separate defects shipped at once and none of them was caught —
// `non_final_rows` was read by the client and never emitted by the server,
// `executor_route` was emitted by the server and never forwarded by the live
// chat path, and the reviewer-slot profile index read a shape the wire does not
// send. A projector test cannot see any of those, because it never asks whether
// the object it was handed is the object the server actually produces.
//
// So this file asserts the SEAM instead of either side of it: the field names
// the client reads off a server payload must be field names the server emits,
// derived from the real Python projection tuples and the real client source
// rather than restated by hand in the test.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { accountRows } from '../modules/harness_accounts.js';
import { nextUpAccount } from '../modules/claudexor_status_store.js';
import { indexProfilesByHarness } from '../modules/reviewer_slots.js';

const repoFile = (rel) => readFileSync(
    fileURLToPath(new URL(`../../${rel}`, import.meta.url)), 'utf-8',
);
const moduleFile = (rel) => readFileSync(
    fileURLToPath(new URL(`../modules/${rel}`, import.meta.url)), 'utf-8',
);

/** Names inside a Python tuple literal assigned to `name = (...)`. */
function pythonTupleNames(source, name) {
    const start = source.indexOf(`${name} = (`);
    assert.notEqual(start, -1, `${name} not found — the wire contract moved, update this test`);
    const body = source.slice(start, source.indexOf('\n)\n', start));
    return new Set([...body.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]));
}

test('every accounting field the costs page reads is a field the history endpoint emits', () => {
    const history = repoFile('ouroboros/gateway/history.py');
    const taskResults = repoFile('ouroboros/task_results.py');
    // What the server puts on the wire: the projected ledger fields plus the four
    // keys the endpoint attaches itself right after the projection.
    const emitted = pythonTupleNames(history, '_ACCOUNTING_SUMMARY_FIELDS');
    for (const literal of ['available', 'authority', 'limit_usd', 'remaining_known_usd']) {
        emitted.add(literal);
    }
    // What the client reads off it, taken from the client source.
    const read = new Set(
        [...moduleFile('costs.js').matchAll(/\baccounting\.([a-z_]+)/g)].map((m) => m[1]),
    );
    assert.ok(read.size >= 8, 'no accounting reads found — the regex or the module moved');
    const missing = [...read].filter((field) => !emitted.has(field));
    assert.deepEqual(missing, [], `costs.js reads server fields that are never emitted: ${missing}`);
    // `cost_final`'s disclosed cause must travel WITH the flag, in both directions.
    assert.ok(emitted.has('non_final_rows') && read.has('non_final_rows'));
    // The durable/replay carry list is no longer a hand-typed copy: it is DERIVED
    // from the cost SSOT (C12), so assert the SSOT owns the name and that
    // task_results really derives from it — a re-typed literal there is the drift
    // this test exists to catch.
    const openness = pythonTupleNames(repoFile('ouroboros/cost_projection.py'), 'COST_OPENNESS_FIELDS');
    assert.ok(openness.has('non_final_rows') && openness.has('ledger_integrity_degraded'));
    assert.match(taskResults, /TASK_COST_META_FIELDS = tuple\(/);
    assert.match(taskResults, /COST_OPENNESS_FIELDS/);
});

test('the live progress path forwards every progress field the endpoint emits and the chat UI consumes', () => {
    const emitted = pythonTupleNames(repoFile('ouroboros/gateway/history.py'), '_PROGRESS_META_FIELDS');
    for (const field of ['executor_route', 'model_lane', 'status', 'subagent_event']) {
        assert.ok(emitted.has(field), `${field} is no longer emitted by the history endpoint`);
    }
    // executor_route drives the executor chip in log_events.js; it must reach the
    // live summarizer, not only the replayed history row.
    assert.match(moduleFile('log_events.js'), /evt\?\.executor_route/);
    // Every LIVE call site that ENUMERATES fields instead of spreading the frame is
    // a whitelist, and a whitelist silently drops whatever it forgot — which is how
    // a chip came back on reload and was missing while the task ran.
    // The enumerated live call site moved into the task-frame router (W3 wave D).
    const chat = moduleFile('chat_task_frames.js');
    const whitelists = chat.split('summarizeChatLiveEvent({').slice(1)
        .map((chunk) => chunk.slice(0, chunk.indexOf('});')))
        .filter((chunk) => !chunk.includes('...evt'));
    assert.ok(whitelists.length > 0, 'no enumerated live call site found — update this test');
    for (const chunk of whitelists) {
        const forwarded = new Set([...chunk.matchAll(/^\s+([a-z_]+):/gm)].map((m) => m[1]));
        assert.ok(forwarded.has('executor_route'),
            'a chat.js live whitelist drops executor_route: the chip only appears after a reload');
    }
});

test('both consumers of the credential-profiles wire read the SAME shape', () => {
    // One golden body, two independent readers. The reviewer-slot index used to
    // read flat camelCase off the `{profile,status,identity}` wrapper and matched
    // nothing, so every session row had an empty profile picker while the harness
    // account list beside it rendered the same accounts correctly.
    const payload = { profiles: JSON.parse(readFileSync(
        fileURLToPath(new URL('./fixtures/credential_profiles_response.json', import.meta.url)),
        'utf-8',
    )) };
    const fromAccounts = accountRows(payload)
        .filter((row) => row.kind === 'profile')
        .map((row) => `${row.harness}/${row.profile_id}`)
        .sort();
    const index = indexProfilesByHarness(payload);
    const fromSlots = Object.entries(index)
        .flatMap(([harness, entries]) => entries.map((entry) => `${harness}/${entry.id}`))
        .sort();
    assert.ok(fromAccounts.length > 0, 'fixture carries no profile rows');
    assert.deepEqual(fromSlots, fromAccounts);
});

test('the UNIFIED wire shape feeds the same readers: all rows named, pools carry routing', () => {
    // The second golden body — a unified engine's ControlCredentialProfilesResponse
    // (frozen contract §L.1): the migrated default login is an ordinary
    // registry row (reserved `<harness>-default` id), `harnessAccounts` is the
    // empty compatibility key, and the routing verdict lives in the additive
    // `accountPools`. Same two readers, no unified-specific branch in either.
    const payload = {
        unified_accounts: true,
        profiles: JSON.parse(readFileSync(
            fileURLToPath(new URL('./fixtures/credential_profiles_response_unified.json', import.meta.url)),
            'utf-8',
        )),
    };
    const rows = accountRows(payload);
    assert.ok(rows.length > 0, 'fixture carries no rows');
    assert.ok(rows.every((row) => row.kind === 'profile'),
        'a unified payload synthesizes no native pseudo-rows');
    // The migrated default is pinnable through the same index the reviewer
    // rows read — the reviewer-side half of the unification.
    const index = indexProfilesByHarness(payload);
    assert.ok((index.codex || []).some((entry) => entry.id === 'codex-default'));
    // The enabled projection reaches both consumers from one reader.
    const byId = Object.fromEntries(rows.map((row) => [row.profile_id, row.enabled]));
    assert.deepEqual(byId, { 'codex-default': true, koshak: false });
    // …and the index carries the same fact, so the pin selects can label a
    // disabled account instead of offering it bare.
    const indexEnabled = Object.fromEntries(Object.values(index).flat()
        .map((entry) => [entry.id, entry.enabled]));
    assert.deepEqual(indexEnabled, byId);
    // Routing rides the pool, read through the store's one dual-wire reader.
    assert.deepEqual(nextUpAccount(payload, 'codex'),
        { kind: 'profile', profileId: 'codex-default' });
});
