// The Agents tab's own shape: family grouping, the ONE service banner, the
// account-row anatomy, and removal.
//
// The behaviours pinned here are the owner's report, one assertion each:
// accounts of one family are EQUIVALENT and grouped, the add affordance lives
// in the family header, the limit text is compact and humanized, and a daemon
// problem is explained once at the top instead of decorating every row.
//
// harness_accounts.test.js keeps the login-flow and payload-shape coverage; the
// two files split by subject, not by module.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
    createClaudexorStatusStore,
    statusUnavailableNote,
} from '../modules/claudexor_status_store.js';
import {
    accountGroups,
    accountMetaLine,
    accountName,
    confirmRemoveAccount,
    familyActionLabel,
    familyLabel,
    familyStatus,
    harnessFamilyMarkup,
    humanizeResetAt,
    nextUpBadge,
    quotaSubjectAliases,
    quotaSummary,
    removeAccount,
    removeAccountConfirmBody,
    rowActionLabel,
    rowLoginAction,
    loginStatusUnknown,
    serviceBannerLine,
    setAccountEnabled,
    vendorCredentialRetainedNotice,
} from '../modules/harness_accounts.js';
import { profileOptionsFor } from '../modules/route_editor_primitives.js';

const MULTI = JSON.parse(readFileSync(
    fileURLToPath(new URL('./fixtures/credential_profiles_multi.json', import.meta.url)), 'utf-8'));

const RUNNING = { state: 'running', engine_version: '3.3.13', runtime: { state: 'ready' } };

function payload({ harnesses = [], profiles = {}, quota = [], daemon = RUNNING } = {}) {
    return { daemon, harnesses, profiles, quota };
}

function fakeStore(reads, { error = '', snapshot = null } = {}) {
    // The fake wraps the REAL sentence factory. An invented note ("…for your
    // quota") let a copy regression pass green: the banner assertions pinned
    // the fake's words instead of the product's ("…for your subscription
    // limits"), which is exactly the wording nothing else pins either. Same
    // reason for the DETAIL rule below: it mirrors the store's own, so a banner
    // that stopped carrying the daemon's `last_error` fails here too.
    const held = snapshot || { daemon: RUNNING };
    return {
        reads,
        facet: (name) => reads[name],
        error,
        snapshot: held,
        loading: false,
        everSettled: true,
        unavailableNote: (facet, { subject = '' } = {}) => {
            const state = reads[facet];
            const detail = error
                || (state === 'failed' ? String(held?.daemon?.last_error || '') : '');
            return statusUnavailableNote(state, { error: detail, facet, subject });
        },
    };
}

const ALL = (value) => ({ catalog: value, accounts: value, quota: value });

// ---------------------------------------------------------------------------
// Grouping: zero accounts, one account, several accounts in one family.
// ---------------------------------------------------------------------------

test('a fresh install still shows every family, each with its own way in', () => {
    // Discovery needs a running daemon and a first run has none, so with the
    // catalogue empty the three bootstrap families must still be reachable —
    // otherwise the whole onboarding path is a dead end.
    const groups = accountGroups(payload({ daemon: { state: 'not_provisioned', runtime: {} } }),
        { accountsRead: 'not_read' });
    assert.deepEqual(groups.map((g) => g.harness), ['codex', 'claude', 'cursor']);
    assert.deepEqual(groups.map((g) => g.label), ['Codex', 'Claude Code', 'Cursor']);
    for (const group of groups) {
        assert.deepEqual(group.rows, []);
        // BIBLE P1: an idle daemon was never asked, so "no account connected"
        // is a claim nobody earned.
        assert.match(group.status.label, /Not checked/);
        assert.equal(familyActionLabel(group, { daemon: { state: 'not_provisioned', runtime: {} } }),
            'Connect');
    }
});

test('an empty family that WAS read says so, and its button still connects', () => {
    const groups = accountGroups(payload({ harnesses: [{ id: 'codex', display_name: 'Codex CLI' }] }),
        { accountsRead: 'ok', catalogKnown: true });
    const codex = groups.find((g) => g.harness === 'codex');
    assert.equal(codex.label, 'Codex CLI');  // discovery wins over the bootstrap name
    assert.equal(codex.status.label, 'No account connected');
    assert.equal(familyActionLabel(codex, payload()), 'Connect');
    // A runtime that needs work carries that intent into the same button.
    assert.equal(familyActionLabel(codex,
        { daemon: { state: 'not_provisioned', runtime: { state: 'missing' } } }), 'Install & connect');
});

test('several accounts of one family are grouped, counted, and called equivalent', () => {
    // The owner's ask, verbatim: «все акки клод кода должны быть эквивалентны».
    // Three codex accounts (one native + two named) become ONE card whose
    // header says they rotate — no row is the "real" one.
    const groups = accountGroups(payload({ profiles: MULTI }), { accountsRead: 'ok' });
    const codex = groups.find((g) => g.harness === 'codex');
    assert.equal(codex.rows.length, 3);
    assert.equal(codex.status.tone, 'ok');
    assert.match(codex.status.label, /3 accounts · rotating/);
    // Once a family HAS accounts its header button ADDS one, which is what
    // makes them equivalent instead of one-default-plus-extras. The affordance
    // used to hang off the native row only.
    assert.equal(familyActionLabel(codex, payload()), 'Add account');

    // Both claude rows are DISABLED in the fixture. That owner-authored state
    // outranks their failed/empty verification: the header does not shout
    // "need attention" and does not pretend the remedy is another sign-in.
    // (An ENABLED failure keeps the family-level alarm; pinned below.)
    const claude = groups.find((g) => g.harness === 'claude');
    assert.equal(claude.status.tone, 'muted');
    assert.match(claude.status.label, /2 accounts · all disabled/);
});

test('one connected account reads as connected, not as a count', () => {
    const rows = [{ harness: 'codex', profile_id: 'work', kind: 'profile',
        status: { verification: 'passed', verification_source: 'vendor' } }];
    assert.deepEqual(familyStatus(rows, { accountsRead: 'ok' }), { tone: 'ok', label: 'Connected' });
    // Present but never signed in: not an alarm, and not a lie either.
    const cold = [{ harness: 'codex', profile_id: 'work', kind: 'profile', status: {} }];
    assert.deepEqual(familyStatus(cold, { accountsRead: 'ok' }),
        { tone: 'muted', label: '1 account · not signed in' });
    const notRun = [{ harness: 'codex', profile_id: 'work', kind: 'profile',
        status: { verification: 'not_run' } }];
    assert.deepEqual(familyStatus(notRun, { accountsRead: 'ok' }),
        { tone: 'muted', label: '1 account · not verified' });
});

test('"N accounts · rotating" counts only the accounts rotation can actually use', () => {
    // Caught by LOOKING at the rendered tab: a Cursor family holding one signed-in
    // account and one cold native row announced "2 accounts · rotating", promising
    // a rotation width that did not exist.
    const mixed = [
        { harness: 'cursor', profile_id: '', kind: 'native', status: {} },
        { harness: 'cursor', profile_id: 'ultra', kind: 'profile',
          status: { verification: 'passed', verification_source: 'local_store' } },
    ];
    assert.deepEqual(familyStatus(mixed, { accountsRead: 'ok' }),
        { tone: 'ok', label: '1 of 2 connected' });
});

test('a disabled account is out of the rotation claim, and all-disabled is its own state', () => {
    // Unified-accounts sprint: the header's "rotating" promise may only count
    // accounts that are BOTH signed in and enabled. A disabled row is the
    // owner's own exclusion — counting it over-promises the pool exactly the
    // way counting a cold row used to.
    const mixed = [
        { harness: 'codex', profile_id: 'codex-default', kind: 'profile', enabled: true,
          status: { verification: 'passed', verification_source: 'local_store' } },
        { harness: 'codex', profile_id: 'work', kind: 'profile', enabled: false,
          status: { verification: 'passed', verification_source: 'vendor' } },
    ];
    assert.deepEqual(familyStatus(mixed, { accountsRead: 'ok' }),
        { tone: 'ok', label: '1 of 2 connected' });
    // Every login healthy, every account disabled: saying "not signed in"
    // would send the owner to a login that fixes nothing.
    const allOff = mixed.map((row) => ({ ...row, enabled: false }));
    assert.deepEqual(familyStatus(allOff, { accountsRead: 'ok' }),
        { tone: 'muted', label: '2 accounts · all disabled' });
    // Disabled rows are intentionally not probed by the engine. Their
    // verification=not_run remains neutral but must not overwrite the owner's
    // stronger structural state.
    const allOffNotRun = allOff.map((row) => ({
        ...row, status: { verification: 'not_run' },
    }));
    assert.deepEqual(familyStatus(allOffNotRun, { accountsRead: 'ok' }),
        { tone: 'muted', label: '2 accounts · all disabled' });
    // Once one row is enabled, that stronger all-disabled fact is gone and the
    // unresolved probe remains honestly not verified.
    const ambiguous = allOffNotRun.map((row, index) => ({
        ...row, enabled: index === 0,
    }));
    assert.deepEqual(familyStatus(ambiguous, { accountsRead: 'ok' }),
        { tone: 'muted', label: '2 accounts · not verified' });
    // …and the metadata line states the exclusion in words on the row itself.
    assert.match(accountMetaLine(allOff[0], payload()), /^disabled — excluded from rotation/);
});

test('a disabled account with a failed verification does not redden the family header', () => {
    // The owner's own exclusion covers the alarm too: rotation ignores a
    // disabled account whatever its verification says, and its row already
    // shows the error in place — so the header must not shout "need
    // attention" about an account the owner has already taken out of play.
    const rows = [
        { harness: 'codex', profile_id: 'work', kind: 'profile',
          status: { verification: 'passed', verification_source: 'vendor' } },
        { harness: 'codex', profile_id: 'old', kind: 'profile', enabled: false,
          status: { verification: 'failed' } },
    ];
    assert.deepEqual(familyStatus(rows, { accountsRead: 'ok' }),
        { tone: 'ok', label: '1 of 2 connected' });
    // The SAME failure on an ENABLED account is rotation's problem and keeps
    // the alarm.
    const enabledFailure = rows.map((row) => ({ ...row, enabled: true }));
    assert.deepEqual(familyStatus(enabledFailure, { accountsRead: 'ok' }),
        { tone: 'error', label: '1 of 2 need attention' });
});

test('the Next up badge renders every wire kind fail-safe, unknown included', () => {
    const base = { daemon: RUNNING, harnesses: [], quota: [] };
    const withPool = (next_up) => ({ ...base,
        profiles: { accountPools: [{ harness_id: 'codex', next_up }], harnessAccounts: [], profiles: [] } });
    assert.equal(nextUpBadge(withPool({ kind: 'profile', profileId: 'codex-default' }), 'codex'),
        'Next up: codex-default');
    assert.equal(nextUpBadge(withPool({ kind: 'api_key_route' }), 'codex'),
        'Next up: API key (no subscription capacity)');
    assert.equal(nextUpBadge(withPool({ kind: 'none', reason: 'nothing routable' }), 'codex'), '');
    // The legacy union, through the same dual-wire reader.
    const legacy = { ...base, profiles: { harnessAccounts: [
        { harness_id: 'codex', next_up: { kind: 'native', route: 'local_session' } },
    ], profiles: [] } };
    assert.equal(nextUpBadge(legacy, 'codex'), 'Next up: default account');
    // FAIL-SAFE: an unknown future kind is a generic unknown, never a crash
    // and never a guessed account (plan §E: old clients must degrade honestly).
    assert.equal(nextUpBadge(withPool({ kind: 'quantum_pool' }), 'codex'), 'Next up: unknown');
    assert.equal(nextUpBadge(withPool({ kind: 'profile' }), 'codex'), 'Next up: unknown');
    // No verdict, or an accounts read that did not land: no badge — a stale
    // routing claim dressed as current is the lie the facets exist to stop.
    assert.equal(nextUpBadge(base, 'codex'), '');
    assert.equal(nextUpBadge(withPool({ kind: 'profile', profileId: 'x' }), 'codex',
        { accountsRead: 'failed' }), '');
});

test('the migrated default row may inherit the LEGACY quota subject, exactly once', () => {
    // Unified migration window (plan §K.3): the quota journal is not
    // rewritten, so right after migration the row's fresh window can still be
    // keyed by the legacy ''/null subject. The alias applies ONLY to the
    // reserved `<harness>-default` row on a unified payload, and an
    // exact-keyed reading always wins.
    const unified = { unified_accounts: true };
    const row = { harness: 'claude', profile_id: 'claude-default', kind: 'profile' };
    assert.deepEqual(quotaSubjectAliases(row, unified), ['']);
    assert.deepEqual(quotaSubjectAliases({ ...row, profile_id: 'work' }, unified), []);
    assert.deepEqual(quotaSubjectAliases(row, {}), [], 'legacy engines get no alias');

    const legacyKeyed = [{ subject: { harness: 'claude', subject_id: '' }, freshness: 'fresh',
        constraints: [{ used_ratio: 0.5, resets_at: '2026-08-09T14:00:00Z' }] }];
    const now = Date.parse('2026-08-09T12:00:00Z');
    const inherited = quotaSummary(legacyKeyed, 'claude', 'claude-default',
        { nowMs: now, fallbackSubjectIds: [''] });
    assert.match(inherited.label, /^50% used/);
    // The exact subject wins the moment a re-keyed reading exists.
    const reKeyed = [...legacyKeyed,
        { subject: { harness: 'claude', subject_id: 'claude-default' }, freshness: 'fresh',
          constraints: [{ used_ratio: 0.2, resets_at: '2026-08-09T14:00:00Z' }] }];
    assert.match(quotaSummary(reKeyed, 'claude', 'claude-default',
        { nowMs: now, fallbackSubjectIds: [''] }).label, /^20% used/);
    // No alias, no inheritance: another named row never borrows the window.
    assert.equal(quotaSummary(legacyKeyed, 'claude', 'work', { nowMs: now }).label,
        'Usage unavailable');
});

test('the family button keeps its ADD intent even when the runtime needs repair', () => {
    // DISCLOSED RESIDUAL, pinned so a change to it is deliberate. `rowActionLabel`
    // hands its label to a broken runtime; this one does not, because the two
    // buttons DO different things. The family button asks for an account name
    // and then starts a login — a header reading "Fix & connect" that opens a
    // name-the-account dialog would misdescribe the click, and dropping the name
    // step would remove the add intent the card exists for. The repair is a
    // prerequisite the login card performs and reports in the foreground, and
    // the service banner above already names the fault.
    const populated = accountGroups(payload({ profiles: MULTI }), { accountsRead: 'ok' })
        .find((g) => g.harness === 'codex');
    const broken = { daemon: { state: 'stale', runtime: { state: 'error' } } };
    assert.equal(familyActionLabel(populated, broken), 'Add account');
    assert.equal(rowActionLabel(populated.rows[0], broken), 'Fix & connect');
    // An EMPTY family has no add intent to protect, so it does carry the runtime.
    const empty = { rows: [] };
    assert.equal(familyActionLabel(empty, broken), 'Fix & connect');
});

test('a row offers what it can actually do, and runtime work outranks it', () => {
    // Also from the render: a row whose vendor verification FAILED offered
    // "Connect", the label that belongs to a family with no account at all.
    const failed = { harness: 'claude', profile_id: 'valentine', kind: 'profile',
        status: { verification: 'failed', verification_source: 'vendor' } };
    const live = { harness: 'claude', profile_id: 'mironov', kind: 'profile',
        status: { verification: 'passed', verification_source: 'vendor' } };
    assert.equal(rowActionLabel(failed, payload()), 'Sign in');
    assert.equal(rowActionLabel(live, payload()), 'Sign in again');
    // A runtime that needs installing/repairing owns the label either way: that
    // work happens first whatever the row wants.
    const broken = { daemon: { state: 'not_provisioned', runtime: { state: 'error' } } };
    assert.equal(rowActionLabel(live, broken), 'Fix & connect');
});

test('an explicit auth probe failure does not turn into a re-login button', () => {
    const unknown = {
        harness: 'claude', profile_id: 'work', kind: 'profile',
        status: { availability: 'unknown', verification: 'not_run',
            detail: 'auth-status probe failed: timeout' },
    };
    assert.equal(loginStatusUnknown(unknown), true);
    assert.equal(rowActionLabel(unknown, payload()), 'Check status');
    assert.deepEqual(rowLoginAction(unknown, payload()),
        { label: 'Check status', refresh: true });
    // The compatibility wire has no availability field, so an older engine's
    // existing action stays unchanged.
    const legacy = { ...unknown, status: { verification: 'not_run' } };
    assert.equal(loginStatusUnknown(legacy), false);
    assert.deepEqual(rowLoginAction(legacy, payload()),
        { label: 'Sign in', refresh: false });
    // Runtime repair remains actionable even while the row's auth probe is
    // unknown: Connect must be able to install/update the pinned engine.
    const broken = { daemon: { state: 'stale', runtime: { state: 'error' } } };
    assert.deepEqual(rowLoginAction(unknown, broken),
        { label: 'Fix & connect', refresh: false });
});

test('the unknown-status refresh action preserves recovery for every profile harness', () => {
    for (const harness of ['claude', 'codex', 'cursor', 'agy']) {
        const row = {
            harness, profile_id: 'work', kind: 'profile',
            status: { availability: 'unknown', verification: 'not_run' },
        };
        assert.deepEqual(rowLoginAction(row, payload()),
            { label: 'Check status', refresh: true }, harness);
    }
});

// ---------------------------------------------------------------------------
// Row anatomy.
// ---------------------------------------------------------------------------

test('the legacy default row is named by who it is, never by a retired account TYPE', () => {
    // Unified-accounts sprint: "Default CLI login" and "Managed by the X CLI"
    // described a separate account type the unified model retired. A legacy
    // pseudo-row is named by the identity the daemon observed — its email —
    // and only an identity-less one falls back to the neutral "Default
    // account". The caption is gone from the metadata line entirely.
    const native = { harness: 'codex', profile_id: '', kind: 'native', identity: {},
        status: { verification: 'passed', verification_source: 'local_store' } };
    assert.equal(accountName(native), 'Default account');
    assert.equal(accountName({ ...native, identity: { email: 'owner@example.com' } }),
        'owner@example.com');
    const meta = accountMetaLine(native, payload({ harnesses: [{ id: 'codex' }] }));
    assert.doesNotMatch(meta, /Managed by/);
    assert.doesNotMatch(meta, /CLI/);
    // Removal is a NAMED-profile affordance: this app cannot honestly sign a
    // vendor CLI out, so it does not offer to.
    assert.equal(rowActionLabel(native, payload()), 'Sign in again');
});

test('the metadata line leads with humanized usage, never a raw ISO instant', () => {
    const now = Date.parse('2026-08-09T12:00:00Z');
    const row = {
        harness: 'claude', profile_id: 'work', kind: 'profile',
        identity: { email: 'a@example.com', plan: 'Max' },
        // formatRelativeAge measures against the real clock, so the "checked"
        // fixture is anchored to it while the quota reset uses the fixed now.
        status: { verification: 'passed', verification_source: 'vendor',
            last_verified_at: new Date(Date.now() - 10 * 60000).toISOString() },
    };
    const meta = accountMetaLine(row, payload({
        quota: [{ subject: { harness: 'claude', subject_id: 'work' }, freshness: 'fresh',
            constraints: [{ used_ratio: 0.38, resets_at: '2026-08-09T14:00:00Z' }] }],
    }), { nowMs: now });
    assert.match(meta, /^38% used · resets in 2h/);
    assert.match(meta, /Max/);
    // The email IS the row's name here (no display_name), so the metadata
    // line does not repeat it.
    assert.equal(accountName(row), 'a@example.com');
    assert.doesNotMatch(meta, /a@example\.com/);
    assert.match(meta, /checked 10m ago/);
    assert.doesNotMatch(meta, /2026-08-09T/);
});

test('reset times are humanized across the whole range, and absence stays absent', () => {
    const now = Date.parse('2026-08-09T12:00:00Z');
    assert.equal(humanizeResetAt('2026-08-09T12:45:00Z', now), 'in 45m');
    assert.equal(humanizeResetAt('2026-08-09T14:00:00Z', now), 'in 2h');
    assert.equal(humanizeResetAt('2026-08-12T12:00:00Z', now), 'in 3d');
    assert.equal(humanizeResetAt('2026-08-09T12:00:30Z', now), 'in a moment');
    assert.equal(humanizeResetAt('', now), '');
    assert.equal(humanizeResetAt('not-a-date', now), '');
});

test('the three limit sentences are the three different facts', () => {
    const now = Date.parse('2026-08-09T12:00:00Z');
    const subject = { harness: 'codex', subject_id: 'work' };
    const spent = quotaSummary([{ subject, freshness: 'fresh',
        constraints: [{ used_ratio: 1.0, resets_at: '2026-08-09T14:00:00Z' }] }],
    'codex', 'work', { nowMs: now });
    assert.equal(spent.label, 'Limit reached · resets in 2h');
    assert.equal(spent.tone, 'warn');
    // READ, and this account has nothing to report.
    assert.equal(quotaSummary([], 'codex', 'work', { nowMs: now }).label, 'Usage unavailable');
    // NOT read: a gap, and never dressed as a full or empty window.
    assert.equal(quotaSummary([], 'codex', 'work', { quotaRead: 'not_read' }).label,
        'Limits not checked');
});

// ---------------------------------------------------------------------------
// The ONE service banner.
// ---------------------------------------------------------------------------

test('the banner is the only place a service problem is explained', () => {
    // Owner report (2026-08-08): a stopped daemon decorated every saved row
    // with "(not in discovery)" and explained nothing. One sentence, at the
    // top, that names the whole tab.
    // A service line that EXPLAINS why nothing was read speaks first: the idle
    // daemon's own sentence carries what happens next ("starts automatically on
    // the next login"), which the generic note does not. The generic note is the
    // fallback for when the service line has nothing concrete to say.
    const line = serviceBannerLine(fakeStore(ALL('not_read'), {
        snapshot: { daemon: { state: 'stale', runtime: { state: 'ready' } } },
    }));
    assert.match(line.text, /agent daemon is not running/);
    assert.match(line.text, /starts automatically/);
    assert.equal(line.tone, 'muted');
    // And when the service line explains NOTHING about the gap — a daemon that
    // is up and healthy while the reads are unstamped — the generic note is what
    // shows, because "Claudexor ready" printed over unread facts is the
    // reassuring lie this whole precedence rule exists to prevent.
    const healthy = serviceBannerLine(fakeStore(ALL('not_read'), {
        snapshot: { daemon: { state: 'running', engine_version: '3.3.13' } },
    }));
    assert.match(healthy.text, /agents, accounts and limits/);
    assert.doesNotMatch(healthy.text, /Claudexor ready/);
    // Healthy: the ordinary lifecycle sentence, unchanged.
    assert.match(serviceBannerLine(fakeStore(ALL('ok'))).text, /Claudexor ready/);
});

test('a BROKEN service is never reported as "nothing below is missing or wrong"', () => {
    // The reachable lie: every settled state that is not `running` leaves all
    // three facets unread, so the benign not-read note used to be the ONLY
    // sentence on the tab — while the row buttons beside it said "Fix &
    // connect". The whole error/warn vocabulary daemonStatusLine speaks was
    // unreachable in exactly the states that need it.
    const broken = (daemon) => serviceBannerLine(fakeStore(ALL('not_read'), { snapshot: { daemon } }));

    const repair = broken({ state: 'stale', runtime: { state: 'error', last_error: 'checksum mismatch' } });
    assert.equal(repair.tone, 'error');
    assert.match(repair.text, /needs repair/);
    assert.match(repair.text, /checksum mismatch/);
    assert.doesNotMatch(repair.text, /Nothing below is missing or wrong/);

    const foreign = broken({ state: 'foreign_daemon', runtime: { state: 'ready' } });
    assert.equal(foreign.tone, 'warn');
    assert.match(foreign.text, /Another daemon answered/);

    const owned = broken({ state: 'stale', ownership_problem: 'home owned by another install', runtime: {} });
    assert.equal(owned.tone, 'error');
    assert.match(owned.text, /not managed from here/);

    const unknown = broken({ state: 'unreachable', last_error: 'connection refused', runtime: {} });
    assert.equal(unknown.tone, 'error');
    assert.match(unknown.text, /connection refused/);

    // The IDLE daemon is not a fault — it keeps the calm muted tone — but its
    // own line is the MORE informative one, so it speaks instead of the generic
    // note. This is what makes the first-run sentence ("No accounts connected
    // yet. Connect installs Claudexor…") reachable at all: every stopped state
    // leaves all three facets unread, so while only warn/error could win, the
    // sentence written for a fresh install could never be printed.
    const idle = broken({ state: 'stale', runtime: { state: 'ready', version: '3.3.13' } });
    assert.equal(idle.tone, 'muted');
    assert.match(idle.text, /agent daemon is not running/);
    const firstRun = broken({ state: 'not_provisioned', runtime: {} });
    assert.equal(firstRun.tone, 'muted');
    assert.match(firstRun.text, /No accounts connected yet/);

    // A read that FAILED is itself a report, not a reassurance: it survives a
    // broken runtime rather than being replaced by it.
    const refused = serviceBannerLine(fakeStore(ALL('failed'), {
        snapshot: { daemon: { state: 'stale', runtime: { state: 'error' } } },
    }));
    assert.equal(refused.tone, 'warn');
    assert.match(refused.text, /could not be read/);
});

test('the REAL store, fed a corrupted runtime, reaches the repair sentence', async () => {
    // Not a hand-set reads map: the actual payload a broken install serves,
    // through the actual provenance mapping. This is what makes the case
    // REACHABLE rather than theoretical — the daemon is not serving, so every
    // facet honestly lands on "never asked", and the benign sentence used to be
    // the only thing on the tab while the buttons beside it said "Fix & connect".
    const body = {
        daemon: {
            state: 'stale',
            runtime: { state: 'error', last_error: 'engine checksum mismatch' },
        },
        harnesses: [], profiles: {}, quota: [],
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => ({ ok: true, json: async () => body }),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    assert.deepEqual(store.reads, { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' });

    const line = serviceBannerLine(store);
    assert.equal(line.tone, 'error');
    assert.match(line.text, /Claudexor needs repair: engine checksum mismatch/);
    assert.doesNotMatch(line.text, /Nothing below is missing or wrong/);
    // The banner and the controls under it now tell the owner the same story.
    assert.equal(rowActionLabel({ status: {} }, store.snapshot), 'Fix & connect');
    store.dispose();
});

test('a card does not contradict itself: the header is dated by the same read as its rows', () => {
    // The row badge stops claiming "Verified live" when the ACCOUNTS read never
    // landed. The header lozenge counts the very same rows, so it obeys the very
    // same provenance — otherwise one card says "Connected" in green over rows
    // that each say "last known", and the owner has to decide which half to
    // believe. An account that needs ATTENTION keeps its tone: a dated warning
    // is still a warning, and muting it would hide the one row worth acting on.
    const live = [{ harness: 'codex', profile_id: 'work', kind: 'profile',
        status: { verification: 'passed', verification_source: 'vendor' } }];
    const broken = [{ harness: 'codex', profile_id: 'work', kind: 'profile',
        status: { verification: 'failed' } }];

    assert.deepEqual(familyStatus(live, { accountsRead: 'ok' }), { tone: 'ok', label: 'Connected' });
    for (const gap of ['not_read', 'failed', 'transport']) {
        const dated = familyStatus(live, { accountsRead: gap });
        assert.equal(dated.tone, 'muted', `${gap} must not paint a green aggregate`);
        assert.match(dated.label, /Connected — last known/);
        assert.equal(familyStatus(broken, { accountsRead: gap }).tone, 'error',
            `${gap} must not mute an account that needs attention`);
    }
    // Two rows, one signed in: the count is still honest, just dated.
    assert.match(familyStatus([...live, { harness: 'codex', profile_id: 'cold', kind: 'profile', status: {} }],
        { accountsRead: 'failed' }).label, /1 of 2 connected — last known/);
});

test('an UNREACHABLE daemon reaches the banner as ONE coarse verdict, carrying its own reason', async () => {
    // Where the two fixes meet. The store stopped reading `unreachable` as a
    // stopped daemon (it is what the endpoint answers when a RUNNING daemon
    // refused ONE of its fanned-out reads), and the tab's banner must carry
    // that through: the honest sentence, the error-toned service line it
    // outranks, and the daemon's OWN last_error — which on this payload is the
    // only explanation of the refusal there is. A banner that assembled the
    // sentence from the copy factory itself would drop it silently.
    //
    // SYNTHESIS: the store's second round narrowed what this payload licenses.
    // A legacy answer with no `reads` stamp does NOT say which read failed —
    // the probe against the live producer had the catalogue and the accounts
    // landing while only the quota refused — so three per-facet `failed`
    // verdicts would pin the quota's error on two reads that succeeded. The
    // banner therefore says ONE coarse thing about the whole answer. What this
    // test was written to protect is unchanged and still asserted below: the
    // daemon's reason survives to the pixels, and neither lie is printed.
    const store = createClaudexorStatusStore({
        fetchImpl: async () => ({
            ok: true,
            json: async () => ({
                daemon: { state: 'unreachable', last_error: 'quota_read_failed: window read died' },
                harnesses: [{ id: 'codex' }], profiles: {}, quota: [],
            }),
        }),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await store.refresh();
    assert.deepEqual(store.reads,
        { catalog: 'indeterminate', accounts: 'indeterminate', quota: 'indeterminate' });

    const line = serviceBannerLine(store);
    assert.match(line.text, /did not finish answering/);
    assert.match(line.text, /window read died/, "the daemon's own reason survives to the pixels");
    // Neither of the two lies: not "nobody asked" (the reads were made), and
    // not the green lifecycle line over lists that never arrived.
    assert.doesNotMatch(line.text, /was not asked/);
    assert.doesNotMatch(line.text, /Claudexor ready/);
    // …and no facet is accused: the catalogue read is right there in the
    // payload, so naming it (or the accounts) would be the misattribution.
    assert.doesNotMatch(line.text, /agent accounts could not be read/);
    assert.doesNotMatch(line.text, /Your agents could not be read/);
    assert.equal(line.tone, 'warn');
    store.dispose();
});

test('the first read in flight states its COST, not a bare "reading…"', () => {
    // The daemon re-probes every agent CLI on each read, so first paint is tens
    // of seconds. An unexplained silent panel reads as broken, not as loading
    // (owner report, 2026-08-08) — a per-facet "Reading your agents…" would
    // have thrown that sentence away.
    const store = {
        reads: ALL('unread'), facet: (n) => ALL('unread')[n], error: '',
        snapshot: null, loading: true, everSettled: false,
        unavailableNote: (facet, { subject = '' } = {}) => statusUnavailableNote('unread', { facet, subject }),
    };
    const line = serviceBannerLine(store);
    assert.match(line.text, /Checking Claudexor/);
    assert.match(line.text, /minute/);
    // The per-facet sentence the banner deliberately does NOT use here is the
    // real one, so this stays a choice between two live sentences rather than
    // between the product's and a stub's.
    assert.match(store.unavailableNote('catalog').text, /Reading your agents…/);
    assert.doesNotMatch(line.text, /Reading your/);
});

test('one refused facet never withdraws the authority of the other two', () => {
    // PER FACET, never a global verdict: with the catalogue and accounts read,
    // a quota refusal must not read as "the service is down". The facet is
    // named in the PRODUCT's words — "subscription limits", the copy the store
    // owns — not in a word this test invented for it.
    const line = serviceBannerLine(fakeStore({ catalog: 'ok', accounts: 'ok', quota: 'failed' }));
    assert.equal(line.tone, 'warn');
    assert.match(line.text, /Your subscription limits could not be read/);
    assert.match(line.text, /Your agents and agent accounts were read normally/);
});

test('a partial gap names EVERY facet it lost, and protects only what it kept', () => {
    // "Everything else on this tab was read normally" was written for one
    // failure and applied to any number of them: with two facets down it told
    // the owner one had failed and the other two were fine. Latent only until
    // the backend stamps `reads` per facet — which is precisely the change that
    // makes a mixed verdict possible.
    const two = serviceBannerLine(fakeStore({ catalog: 'ok', accounts: 'failed', quota: 'failed' }));
    assert.match(two.text, /agent accounts and subscription limits/);
    assert.match(two.text, /Your agents were read normally/);
    assert.doesNotMatch(two.text, /Everything else/);

    // DIFFERENT failures are different sentences, and the worst tone wins.
    const mixed = serviceBannerLine(fakeStore(
        { catalog: 'ok', accounts: 'transport', quota: 'not_read' },
        { error: 'HTTP 503' }));
    assert.equal(mixed.tone, 'error');
    assert.match(mixed.text, /Could not read your agent accounts \(HTTP 503\)/);
    assert.match(mixed.text, /your subscription limits were never checked/);
    assert.match(mixed.text, /Your agents were read normally/);

    // Nothing read OK at all: no reassurance is appended, because there is
    // nothing left to reassure about.
    const none = serviceBannerLine(fakeStore(
        { catalog: 'failed', accounts: 'failed', quota: 'not_read' }));
    assert.match(none.text, /agents and agent accounts/);
    assert.match(none.text, /your subscription limits were never checked/);
    assert.doesNotMatch(none.text, /were read normally/);
});

test('a partial gap obeys the SAME fault precedence as a total one', () => {
    // The full-gap and partial-gap branches are one decision, and fixing only
    // one half is how this class survives to the next review. The backend
    // stamps `reads` per facet on every answer, so a mixed verdict is an
    // ordinary state — this is exactly the shape that produces one, at which point a muted "these were
    // never asked · the rest read normally" would quietly swallow a runtime that
    // needs repair.
    const broken = { daemon: { state: 'stale', runtime: { state: 'error', last_error: 'checksum' } } };

    const muted = serviceBannerLine(fakeStore(
        { catalog: 'ok', accounts: 'not_read', quota: 'not_read' }, { snapshot: broken }));
    assert.equal(muted.tone, 'error');
    assert.match(muted.text, /needs repair: checksum/);

    // A partial gap that is itself a REPORT still outranks the fault — a
    // refused read is not a reassurance and must not be replaced by one.
    const refused = serviceBannerLine(fakeStore(
        { catalog: 'ok', accounts: 'failed', quota: 'not_read' }, { snapshot: broken }));
    assert.equal(refused.tone, 'warn');
    assert.match(refused.text, /Your agent accounts could not be read/);
    assert.match(refused.text, /Your agents were read normally/);

    // A healthy daemon leaves the partial sentence exactly as it was.
    const healthy = serviceBannerLine(fakeStore(
        { catalog: 'ok', accounts: 'not_read', quota: 'not_read' }));
    assert.equal(healthy.tone, 'muted');
    assert.match(healthy.text, /agent accounts and subscription limits were never checked/);
});

// ---------------------------------------------------------------------------
// Removal.
// ---------------------------------------------------------------------------

test('Remove consumes the non-input confirm boolean and performs the mutation once', async () => {
    const calls = [];
    const store = {
        snapshot: { harnesses: [{ id: 'claude', display_name: 'Claude Code' }] },
        refresh: async () => calls.push(['refresh']),
    };

    await confirmRemoveAccount('claude', 'work', {
        dialogImpl: async (options) => {
            assert.notEqual(options.input, true);
            assert.equal(options.danger, true);
            assert.equal(options.confirmLabel, 'Remove');
            calls.push(['dialog']);
            return true;
        },
        removeImpl: async (harness, profileId) => calls.push(['remove', harness, profileId]),
        store,
        renderImpl: () => calls.push(['render']),
    });

    assert.deepEqual(calls, [
        ['dialog'],
        ['remove', 'claude', 'work'],
        ['refresh'],
        ['render'],
    ]);

    for (const resolution of [false, undefined, null, { confirmed: true }]) {
        const skipped = [];
        await confirmRemoveAccount('claude', 'work', {
            dialogImpl: async () => { skipped.push(['dialog']); return resolution; },
            removeImpl: async () => skipped.push(['remove']),
            store: { snapshot: store.snapshot, refresh: async () => skipped.push(['refresh']) },
            renderImpl: () => skipped.push(['render']),
        });
        assert.deepEqual(skipped, [['dialog']],
            `resolution ${JSON.stringify(resolution)} must stop after the dialog`);
    }
});

test('removing a named account goes through the engine contract, and says so', () => {
    const calls = [];
    const receipt = { profile: {
        profile_id: 'work', harness_id: 'codex', display_name: 'Work',
        credential_kind: 'config_dir_login',
        isolation_locator: '/data/claudexor/profiles/codex-work', secret_ref: null,
        enabled: true, created_at: null,
    }, removed: true, credentialCleanup: 'config_dir_removed',
        cleanupWarning: 'owned profile storage cleanup needs manual inspection',
        vendorCredentialDisposition: {
            owner: 'vendor', state: 'left_unchanged', scope: 'os_user',
        } };
    const fetchImpl = async (url, init) => {
        calls.push([url, init.method]);
        return { ok: true, json: async () => receipt };
    };
    return removeAccount('codex', 'work', { fetchImpl }).then((answer) => {
        assert.deepEqual(answer, receipt);
        assert.deepEqual(calls, [['/api/claudexor/credential-profiles/codex/work', 'DELETE']]);
        // The confirmation states the two facts an owner needs before agreeing:
        // vendor/OS sign-in may remain, and a pinned reviewer row or Delegation
        // pin survives visibly instead of rerouting.
        const body = removeAccountConfirmBody('work', 'Codex');
        assert.match(body, /Vendor or OS credential storage may remain signed in/);
        assert.match(body, /deletion receipt says when it was retained/);
        assert.match(body, /Reviewer rows and a Delegation pin pointing at this account stay visible/);
    });
});

test('only the exact vendor left-unchanged disposition produces a retained warning', () => {
    const receipt = {
        profile: {
            profile_id: 'work', harness_id: 'codex', display_name: 'Work',
            credential_kind: 'config_dir_login',
            isolation_locator: '/data/claudexor/profiles/codex-work', secret_ref: null,
            enabled: true, created_at: null,
        },
        removed: true,
        credentialCleanup: 'config_dir_removed',
        vendorCredentialDisposition: {
            owner: 'vendor', state: 'left_unchanged', scope: 'os_user',
        },
    };
    const notice = vendorCredentialRetainedNotice(receipt, 'work', 'Codex');
    assert.equal(notice, 'Removed "work" from Codex. Claudexor left vendor credential '
        + 'storage for this OS user unchanged; the vendor account may still be signed in '
        + 'outside Ouroboros.');
    for (const changed of [
        {},
        { vendorCredentialDisposition: null },
        { vendorCredentialDisposition: { ...receipt.vendorCredentialDisposition, owner: 'claudexor' } },
        { vendorCredentialDisposition: { ...receipt.vendorCredentialDisposition, state: 'removed' } },
        { vendorCredentialDisposition: { ...receipt.vendorCredentialDisposition, scope: 'profile' } },
    ]) {
        assert.equal(vendorCredentialRetainedNotice(changed, 'work', 'Codex'), '');
    }
});

test('a refused removal is reported as a refusal, never as a removal', () => {
    const fetchImpl = async () => ({ ok: false, status: 409, json: async () => ({ error: 'in use' }) });
    return assert.rejects(() => removeAccount('codex', 'work', { fetchImpl }), /in use/);
});

test('the Enabled toggle is the engine PATCH contract, and a refusal changes nothing', async () => {
    const calls = [];
    const fetchImpl = async (url, init) => {
        calls.push([url, init.method, init.body]);
        return { ok: true, json: async () => ({ ok: true, enabled: false }) };
    };
    const answer = await setAccountEnabled('codex', 'work', false, { fetchImpl });
    assert.deepEqual(answer, { ok: true, enabled: false });
    assert.deepEqual(calls, [['/api/claudexor/credential-profiles/codex/work', 'PATCH',
        JSON.stringify({ enabled: false })]]);
    const refuse = async () => ({ ok: false, status: 503, json: async () => ({ error: 'daemon_unreachable' }) });
    await assert.rejects(() => setAccountEnabled('codex', 'work', true, { fetchImpl: refuse }),
        /daemon_unreachable/);
});

test('a reviewer row pinned to a removed account keeps its pin visible', () => {
    // The row must not silently reroute to automatic rotation: that would widen
    // which account the reviewer may spend without the owner deciding it.
    const options = profileOptionsFor([{ id: 'koshak', name: 'koshak', enabled: true }], 'work');
    assert.deepEqual(options.map((option) => option.value), ['', 'koshak', 'work']);
    assert.equal(options.at(-1).label, 'Account: work (not in discovery)');
    // Accounts never read: the pin only LOOKS missing (BIBLE P1).
    assert.equal(profileOptionsFor([], 'work', { accountsKnown: false }).at(-1).label, 'Account: work (not checked)');
    assert.deepEqual(profileOptionsFor([{ id: 'work', name: 'work', enabled: true }], 'work').map((option) => option.value),
        ['', 'work']);
});

test('familyLabel prefers live discovery and falls back to the product name', () => {
    assert.equal(familyLabel('claude', payload()), 'Claude Code');
    assert.equal(familyLabel('claude',
        payload({ harnesses: [{ id: 'claude', display_name: 'Claude Code CLI' }] }),
        { catalogKnown: true }),
        'Claude Code CLI');
    assert.equal(familyLabel('claude',
        payload({ harnesses: [{ id: 'claude', display_name: 'Stale daemon label' }] })),
        'Claude Code', 'omitted catalog provenance fails closed over a retained snapshot');
    // An unknown harness is named by its own id rather than invented.
    assert.equal(familyLabel('mystery', payload()), 'mystery');
});

test('account headers and dialogs withdraw retained daemon labels until catalog recovery', async () => {
    const status = (label, catalog = 'ok') => ({
        daemon: { state: 'running', engine_version: '3.8.1', runtime: {} },
        harnesses: [{ id: 'codex', display_name: label }],
        profiles: {
            harnessAccounts: [],
            profiles: [{
                profile: {
                    harness_id: 'codex', profile_id: 'work',
                    display_name: 'Work', enabled: true,
                },
                status: { availability: 'available', verification: 'passed' },
                identity: {},
            }],
        },
        quota: [],
        reads: { catalog, accounts: 'ok', quota: 'ok' },
    });
    const responses = [
        { ok: true, status: 200, body: status('Codex Live') },
        // The backend can retain the last catalog projection in a partial
        // answer while explicitly refusing authority for that facet.
        { ok: true, status: 200, body: status('Codex Live', 'failed') },
        // A transport failure is the other retained-snapshot shape: the store
        // deliberately keeps the partial answer above for usable controls.
        { ok: false, status: 503, body: { error: 'catalog transport died' } },
        { ok: true, status: 200, body: status('Codex Restored') },
    ];
    const store = createClaudexorStatusStore({
        fetchImpl: async () => {
            const response = responses.shift();
            return {
                ok: response.ok,
                status: response.status,
                json: async () => response.body,
            };
        },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const header = () => {
        const group = accountGroups(store.snapshot, {
            accountsRead: store.facet('accounts'),
            catalogKnown: store.catalogKnown,
        }).find((candidate) => candidate.harness === 'codex');
        return harnessFamilyMarkup(group, store.snapshot, {
            accountsRead: store.facet('accounts'),
            quotaRead: store.facet('quota'),
        });
    };
    const removeBody = async () => {
        let body = '';
        await confirmRemoveAccount('codex', 'work', {
            store,
            dialogImpl: async (options) => {
                body = options.body;
                return false;
            },
            removeImpl: async () => assert.fail('cancelled dialog removed an account'),
            renderImpl: () => {},
        });
        return body;
    };
    try {
        await store.refresh();
        assert.equal(store.catalogKnown, true);
        assert.match(header(), />Codex Live<\/span>/);
        assert.match(await removeBody(), /for Codex Live account/);

        await store.refresh();
        assert.equal(store.catalogKnown, false);
        assert.match(header(), />Codex<\/span>/);
        assert.doesNotMatch(header(), /Codex Live/);
        assert.match(await removeBody(), /for Codex account/);
        assert.doesNotMatch(await removeBody(), /Codex Live/);

        const retained = store.snapshot;
        await store.refresh();
        assert.equal(store.snapshot, retained, 'transport failure discarded the usable snapshot');
        assert.equal(store.catalogKnown, false);
        assert.match(header(), />Codex<\/span>/);
        assert.doesNotMatch(header(), /Codex Live/);
        assert.doesNotMatch(await removeBody(), /Codex Live/);

        await store.refresh();
        assert.equal(store.catalogKnown, true);
        assert.match(header(), />Codex Restored<\/span>/);
        assert.match(await removeBody(), /for Codex Restored account/);
    } finally {
        store.dispose();
    }
});
