// Regenerates matrix.json: every snapshot's `availability` is computed by the
// Claudexor engine's own `withQuotaAvailability` (packages/schema/src), exactly
// as /v2/quota decorates rows; only zod's parse is replaced by identity so no
// dependency install is needed. No daemon, vendor read or credential is used.
//
//   node --experimental-strip-types tests/fixtures/quota_subject_matrix/generate.mjs <claudexor-checkout>
//
// The checkout is only read: three schema sources are copied into a temp dir.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const checkout = path.resolve(process.argv[2] || '');
const SOURCES = ['quota-availability.ts', 'quota.ts', 'quota-request-kind.ts'];
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'quota-matrix-'));
const identity = {};
for (const name of SOURCES) {
    const text = fs.readFileSync(path.join(checkout, 'packages/schema/src', name), 'utf8');
    identity[`packages/schema/src/${name}`] = createHash('sha256').update(text).digest('hex');
    fs.writeFileSync(path.join(work, name), text.replace(/from "\.\/([\w-]+)\.js"/g, 'from "./$1.ts"')
        .replace(/from "zod\/v3"/g, 'from "./zod.mjs"'));
}
fs.writeFileSync(path.join(work, 'zod.mjs'), `const z = new Proxy(function () {}, {
    get: (_target, key) => (key === 'parse' ? (value) => value : key === 'then' ? undefined : z),
    apply: () => z,
});
export { z };
`);
fs.writeFileSync(path.join(work, 'primitives.ts'), 'import { z } from "./zod.mjs";\nexport const Id = z;\n');
fs.writeFileSync(path.join(work, 'account-target.ts'), 'import { z } from "./zod.mjs";\nexport const ControlQuotaRefreshRequest = z;\n');
const { withQuotaAvailability } = await import(path.join(work, 'quota-availability.ts'));

const T0 = Date.parse('2026-10-29T12:00:00.000Z');
const at = (minutes) => new Date(T0 + minutes * 60_000).toISOString();
const MODEL = 'gpt-test';
const SCOPE = { model: MODEL, requestKind: 'text', basis: 'quota_block' };
const subject = (id, route = 'vendor_native') => ({ harness: 'codex', credential_route: route, plan_label: null, subject_id: id });
const window = (id, ratio, resetMinutes, extra = {}) => ({
    id, label: id, used_ratio: ratio, window_seconds: 18_000, resets_at: resetMinutes == null ? null : at(resetMinutes),
    cooldown_until: null, ...extra,
});
const PRIMARY = window('primary', 0.2, 120);
const SECONDARY = window('secondary', 1, 3 * 24 * 60);
const row = (id, constraints, { freshness = 'fresh', route, source = 'codex_app_server', observed = -1 } = {}) => ({
    subject: subject(id, route), source, observed_at: at(observed), freshness,
    constraints: constraints.map((constraint) => ({ ...constraint, freshness })),
});
const absence = (id, reason = 'refresh_failed', route) => ({ subject: subject(id, route), reason, detail: null, observed_at: at(-1) });

// `expect` is the reviewed contract; `catches` names the retired readers each case refutes.
const CASES = [
    ['full_primary20_secondary100', [row('a', [PRIMARY, SECONDARY])], [], {}, { state: 'exhausted', resets_at: at(4320) }],
    ['split_primary20_secondary100', [row('a', [PRIMARY]), row('a', [SECONDARY], { source: 'codex_app_server_event' })], [], {},
        { state: 'exhausted', resets_at: at(4320) }, ['flat_or']],
    ['blocked_beside_stale_same_subject', [row('a', [SECONDARY]), row('a', [PRIMARY], { source: 'codex_rollout', freshness: 'stale' })], [], {},
        { state: 'exhausted', resets_at: at(4320) }, ['flat_or']],
    ['stale_block_still_dated_beside_free', [row('a', [SECONDARY], { freshness: 'stale', source: 'codex_rollout' }), row('a', [PRIMARY])], [], {},
        { state: 'exhausted', resets_at: at(4320) }, ['flat_or']],
    ['blocked_beside_free_same_subject', [row('a', [window('cap', 1, 60)], { source: 'codex_app_server_event' }), row('a', [PRIMARY])], [], {},
        { state: 'exhausted', resets_at: at(60) }, ['flat_or']],
    ['blocked_beside_absence_same_subject', [row('a', [SECONDARY])], [absence('a')], {}, { state: 'exhausted', resets_at: at(4320) }, ['flat_or']],
    ['fresh_available_beside_stale_nonblock', [row('a', [PRIMARY]), row('a', [window('old', 0.4, 600)], { source: 'codex_rollout', freshness: 'stale' })], [], {},
        { state: 'unknown' }],
    ['fresh_available_beside_absence', [row('a', [PRIMARY])], [absence('a')], {}, { state: 'unknown' }],
    ['early_reset_repoll', [row('a', [window('cap', 1, 30)])], [], { nowMinutes: 45 }, { state: 'unknown' }, ['flat_or']],
    ['live_reset_before_repoll', [row('a', [window('cap', 1, 30)])], [], { nowMinutes: 29 }, { state: 'exhausted', resets_at: at(30) }],
    ['available_beside_blocked_subject', [row('a', [PRIMARY]), row('b', [SECONDARY])], [], {}, { state: 'available' }],
    ['unknown_beside_blocked_subject', [row('a', [PRIMARY], { freshness: 'stale' }), row('b', [SECONDARY])], [], {}, { state: 'unknown' }],
    ['model_scoped_available_beside_blocked_subject', [row('a', [window('model', 0.2, 120, { applies_to_models: [MODEL] })]), row('b', [SECONDARY])], [], {},
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['all_blocked_earliest_recheck', [row('a', [SECONDARY]), row('b', [window('cool', null, null, { cooldown_until: at(90) })])], [], {},
        { state: 'exhausted', resets_at: at(90) }],
    ['all_cooling', [row('a', [window('cool', null, null, { cooldown_until: at(90) })])], [], {}, { state: 'cooldown', resets_at: at(90) }],
    ['absence_only_alternative', [row('a', [SECONDARY])], [absence('b', 'not_logged_in')], {}, { state: 'unknown' }],
    ['absence_only_pin', [row('a', [PRIMARY])], [absence('b')], { profile: 'b' }, { state: 'unknown' }],
    ['strict_pin_blocked_beside_free_sibling', [row('a', [SECONDARY]), row('b', [PRIMARY])], [], { profile: 'a' },
        { state: 'exhausted', resets_at: at(4320) }],
    ['strict_pin_own_available', [row('a', [SECONDARY]), row('b', [PRIMARY])], [], { profile: 'b' }, { state: 'available' }],
    ['strict_pin_never_borrows_null_subject', [row(null, [PRIMARY])], [], { profile: 'codex-default' }, { state: 'unknown' }],
    ['same_id_different_credential_route', [row('a', [SECONDARY]), row('a', [PRIMARY], { route: 'managed_api_key' })], [], {},
        { state: 'available' }],
    ['same_id_route_absence_is_its_own_subject', [row('a', [SECONDARY])], [absence('a', 'refresh_failed', 'managed_api_key')], {},
        { state: 'unknown' }],
    ['null_subject_is_an_auto_alternative', [row(null, [PRIMARY]), row('a', [SECONDARY])], [], {}, { state: 'available' }],
    ['null_and_named_both_blocked', [row(null, [window('cap', 1, 200)]), row('a', [SECONDARY])], [], {},
        { state: 'exhausted', resets_at: at(200) }],
    ['malformed_reset_is_no_live_block', [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], 'next week']] },
        { state: 'unknown' }, ['flat_or']],
    ['offsetless_reset_is_no_live_block', [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], '2026-11-01T12:00:00']] },
        { state: 'unknown' }, ['flat_or']],
    ['impossible_date_is_no_live_block', [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], '2026-13-01T12:00:00Z']] },
        { state: 'unknown' }, ['flat_or']],
    ['microsecond_reset_parses', [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], '2026-11-01T12:00:00.123456+00:00']] },
        { state: 'exhausted', resets_at: '2026-11-01T12:00:00.123456+00:00' }],
    // Calendar/clock overflow that a lenient parser would roll forward is no live block in either reader.
    ...[['nonleap_feb29', '2099-02-29T12:00:00Z'], ['feb30_day_overflow', '2028-02-30T12:00:00Z'],
        ['apr31_day_overflow', '2099-04-31T12:00:00Z'], ['hour24', '2099-01-01T24:00:00Z'],
        ['leap_second', '2099-01-01T23:59:60Z'], ['offset_minute_overflow', '2099-01-01T12:00:00+05:60'],
        ['offset_hour24', '2099-01-01T12:00:00+24:00']].map(([name, reset]) => [`${name}_is_no_live_block`,
        [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], reset]] }, { state: 'unknown' }, ['flat_or']]),
    ...[['leap_day', '2028-02-29T12:00:00Z'], ['positive_offset', '2026-11-01T17:30:00+05:30'],
        ['negative_offset', '2026-11-01T04:00:00-08:00'], ['one_digit_fraction', '2026-11-01T12:00:00.5Z'],
        ['nine_digit_fraction', '2026-11-01T12:00:00.123456789Z']].map(([name, reset]) => [`${name}_reset_parses`,
        [row('a', [SECONDARY])], [], { post: [[0, ['availability', 'resets_at'], reset]] }, { state: 'exhausted', resets_at: reset }]),
    ['offset_resets_order_by_instant', [row('a', [SECONDARY]), row('b', [SECONDARY])], [], { post: [
        [0, ['availability', 'resets_at'], '2026-11-01T17:30:00+05:30'], [1, ['availability', 'resets_at'], '2026-11-01T11:00:00-02:00']] },
        { state: 'exhausted', resets_at: '2026-11-01T17:30:00+05:30' }],
    ['boolean_ratio_is_no_measurement', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'used_ratio'], true]] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['null_ratio_is_no_measurement', [row('a', [window('unmeasured', null, 120)])], [], {}, { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['out_of_range_ratio_is_malformed', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'used_ratio'], 1.5],
        [0, ['availability'], { state: 'available', blocking_constraints: [], resets_at: null, model_scoped_exhaustions: [] }]] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['null_prefixes_are_malformed_scope', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'applies_to_model_prefixes'], null]] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['string_model_scope_is_malformed', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'applies_to_models'], 'all']] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['empty_request_kinds_apply_to_nothing', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'applies_to_request_kinds'], []]] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['legacy_scope_missing', [row('a', [SECONDARY])], [], { scope: null }, { state: 'unknown', reason: 'availability_scope_unhonored' }],
    ['scope_for_another_model', [row('a', [PRIMARY])], [], { scope: { ...SCOPE, model: 'other' } }, { state: 'unknown', reason: 'availability_scope_unhonored' }],
    ['empty_envelope', [], [], {}, { state: 'unknown', reason: 'quota_evidence_missing' }],
    ['image_only_window_for_text', [row('a', [window('image', 0.1, 120, { applies_to_request_kinds: ['image'] })])], [], {},
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['image_only_available_beside_blocked_subject', [row('a', [window('image', 0.1, 120, { applies_to_request_kinds: ['image'] })]),
        row('b', [SECONDARY])], [], {}, { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['null_request_kinds_are_malformed_scope', [row('a', [PRIMARY])], [], { post: [[0, ['constraints', 0, 'applies_to_request_kinds'], null]] },
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['text_window_for_text', [row('a', [window('text', 0.1, 120, { applies_to_request_kinds: ['text'] })])], [], {}, { state: 'available' }],
    ['image_window_for_image', [row('a', [window('image', 0.1, 120, { applies_to_request_kinds: ['image'] })])], [], { requestKind: 'image' },
        { state: 'available' }],
    ['image_block_does_not_touch_text', [row('a', [PRIMARY, window('image', 1, 120, { applies_to_request_kinds: ['image'] })])], [], {},
        { state: 'available' }],
    ['foreign_model_scope', [row('a', [window('other', 0.1, 120, { applies_to_models: ['other-model'] })])], [], {},
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['declared_prefix_scope', [row('a', [window('family', 0.1, 120, { applies_to_model_prefixes: ['gpt-'] })])], [], {},
        { state: 'unknown' }, ['flat_or', 'aggregate_fresh_only']],
    ['empty_model_list_is_global', [row('a', [window('wide', 0.1, 120, { applies_to_models: [] })])], [], {}, { state: 'available' }],
    ['foreign_model_block_beside_global_measurement', [row('a', [PRIMARY, window('other', 1, 120, { applies_to_models: ['other-model'] })])], [], {},
        { state: 'available' }],
];

const matrix = CASES.map(([name, snapshots, absences, options, expect, catches = []]) => {
    const kind = options.requestKind || 'text';
    const envelope = withQuotaAvailability({ snapshots, absences, refreshed_at: null }, { now: new Date(T0), model: MODEL, requestKind: kind });
    for (const [index, keys, value] of options.post || []) {
        const target = keys.slice(0, -1).reduce((node, key) => node[key], envelope.snapshots[index]);
        target[keys.at(-1)] = value;
    }
    if (options.scope !== null) envelope.availability_scope = options.scope || { ...SCOPE, requestKind: kind };
    return { name, harness: 'codex', model: MODEL, profile: options.profile || '', request_kind: kind,
        now: at(options.nowMinutes ?? 1), envelope, expect, catches };
});
const commit = execFileSync('git', ['-C', checkout, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const out = new URL('./matrix.json', import.meta.url);
const source = { claudexor_commit: commit, sha256: identity,
    producer: 'withQuotaAvailability(response, {now, model, requestKind}) with zod parse as identity' };
fs.writeFileSync(out, `{"source": ${JSON.stringify(source)},\n"cases": [\n${matrix.map((item) => JSON.stringify(item)).join(',\n')}\n]}\n`);
fs.rmSync(work, { recursive: true, force: true });
console.log(`${matrix.length} cases from ${commit}`);
