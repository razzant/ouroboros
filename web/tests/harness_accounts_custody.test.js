import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
    createClaudexorStatusStore,
    statusUnavailableNote,
} from '../modules/claudexor_status_store.js';
import {
    accountRowFacts,
    bareRowStatusText,
    serviceBannerLine,
    startLogin,
    verificationBadge,
} from '../modules/harness_accounts.js';
import { cancelLoginJob, loginReleaseProven } from '../modules/harness_login_cards.js';
import { fakeResponse } from './harness_accounts_helpers.js';

// ---------------------------------------------------------------------------
// C7: login-job serialization — a new login only after the old one is gone.
// ---------------------------------------------------------------------------

test('cancelLoginJob parses canonical custody evidence instead of treating any 2xx as gone', async () => {
    const mk = (status, body) => async () => fakeResponse(status, body);
    assert.equal((await cancelLoginJob('job-1', mk(200, { job: { state: 'cancelled' } }))).status,
        'released');
    assert.equal((await cancelLoginJob('job-1', mk(200, { job: { state: 'interrupted_unknown',
        outcome: { reason: 'termination_unconfirmed' } } }))).status, 'retained');
    assert.equal((await cancelLoginJob('job-1', mk(200, { job: { state: 'cancelling' } }))).status,
        'retained');
    assert.equal((await cancelLoginJob('job-1', mk(200, {}))).status, 'unknown');
    assert.equal((await cancelLoginJob('job-1', mk(200, { job: {} }))).status, 'unknown');
    assert.equal((await cancelLoginJob('job-1', mk(404, {}))).status, 'released');
    assert.equal((await cancelLoginJob('job-1', mk(410, {}))).status, 'released');
    assert.equal((await cancelLoginJob('job-1', mk(503, { error: 'down' }))).status, 'unknown');
    assert.equal((await cancelLoginJob('job-1', async () => { throw new Error('net'); })).status, 'unknown');
    assert.equal((await cancelLoginJob('', async () => { throw new Error('must not be called'); })).status,
        'released');
});

test('startLogin centralizes the C7 guard: cancel-or-refuse BEFORE the new login POST', () => {
    // ESM keeps the controller's internal state untestable directly; pin the
    // control flow at the source level (same source-based technique as the HTML
    // pins in this file): the guard must sit inside the locked start ahead of
    // the POST, and a failed cancellation must return without starting a second
    // job. The flow moved to harness_login_cards.js in phase 2; the RULE did not.
    const src = readFileSync(fileURLToPath(new URL('../modules/harness_login_cards.js', import.meta.url)), 'utf8');
    const fn = src.slice(src.indexOf('async function _startLocked'));
    const guardAt = fn.indexOf('cancelLoginJob(prev.jobId');
    const postAt = fn.indexOf("fetchImpl('/api/claudexor/login'");
    assert.ok(guardAt > -1, 'startLogin must call cancelLoginJob for a live previous job');
    assert.ok(postAt > -1);
    assert.ok(guardAt < postAt, 'the C7 guard must run before the new login POST');
    const guarded = fn.slice(guardAt, postAt);
    assert.match(guarded, /result\.status === LOGIN_CUSTODY_UNKNOWN[\s\S]*?return;/,
        'unknown cancellation evidence must refuse the new login');
});


test('the one release predicate rejects unconfirmed termination until reconciliation proves empty', () => {
    assert.equal(loginReleaseProven(null), false);
    assert.equal(loginReleaseProven({ job: { state: 'running' } }), false);
    assert.equal(loginReleaseProven({ job: { state: 'cancelled' } }), true);
    const retained = { job: { state: 'interrupted_unknown',
        outcome: { reason: 'termination_unconfirmed' } } };
    assert.equal(loginReleaseProven(retained), false);
    assert.equal(loginReleaseProven({ job: { ...retained.job,
        terminationReconciliation: { status: 'empty' } } }), true);
    assert.equal(loginReleaseProven(null, { absent: true }), true);
    assert.equal(loginReleaseProven({ state: 'cancelled' }), false,
        'old bare jobs do not regain release authority');
});

test('a harness with no row only says "no account connected" once the store was READ', () => {
    // BIBLE P1 at the pixel: the owner's panel declared three harnesses empty
    // while two claude profiles, a cursor profile and two native sessions sat
    // in the agent home — a lazy daemon had simply never been asked.
    assert.equal(bareRowStatusText('ok'), 'No account connected');
    assert.match(bareRowStatusText('not_read'), /Not checked/);
    // NOT READ says nobody asked. It may not name a CAUSE the row cannot
    // see: a runtime awaiting repair and a foreign daemon on the stale port
    // arrive here as the same unread facet, and the tab's banner owns the why.
    assert.match(bareRowStatusText('not_read'), /daemon was never asked/);
    assert.doesNotMatch(bareRowStatusText('not_read'), /is not running/);
    assert.match(bareRowStatusText('failed'), /did not answer/);
    assert.match(bareRowStatusText('transport'), /request did not complete/);
    assert.equal(bareRowStatusText('unread'), 'Checking…');
    // The coarse state claims nothing beyond "the answer did not complete" —
    // it does not know which read failed, so it may not blame this one.
    assert.match(bareRowStatusText('indeterminate'), /answer did not complete/);
    assert.doesNotMatch(bareRowStatusText('indeterminate'), /not running/);
    assert.doesNotMatch(bareRowStatusText('indeterminate'), /never asked/);
    // Each gap is its OWN sentence — collapsing them would re-create the lie.
    const gaps = ['not_read', 'failed', 'transport', 'indeterminate'].map(bareRowStatusText);
    assert.equal(new Set(gaps).size, 4);
});

test('the ONE service banner reports a REFUSED read instead of "Claudexor ready"', () => {
    // A running daemon whose account read died would otherwise print the green
    // lifecycle line over a list that was never delivered.
    // The fake wraps the REAL sentence factory: a fake that invents its own
    // wording pins the fake, so a copy regression in the product passes green.
    const fakeStore = (reads, error = '') => ({
        reads,
        facet: (name) => reads[name],
        error,
        snapshot: { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} } },
        loading: false,
        everSettled: true,
        unavailableNote: (facet, { subject = '' } = {}) =>
            statusUnavailableNote(reads[facet], { error, facet, subject }),
    });
    const all = (v) => ({ catalog: v, accounts: v, quota: v });
    // Every facet gone the same way: ONE sentence, with the subject widened to
    // the whole tab — naming only the accounts would under-report the gap.
    assert.match(serviceBannerLine(fakeStore(all('failed'))).text, /could not be read/);
    assert.match(serviceBannerLine(fakeStore(all('failed'))).text, /agents, accounts and limits/);
    // A read that was ATTEMPTED and did not land is never dressed as a read
    // nobody made: the calm never-asked sentence belongs to the stopped states.
    assert.doesNotMatch(serviceBannerLine(fakeStore(all('failed'))).text, /was not asked/);
    assert.match(serviceBannerLine(fakeStore(all('transport'), 'net')).text, /Could not read/);
    // …and so does the COARSE state, which is what a legacy `unreachable`
    // answer becomes: the green lifecycle line over an undelivered list is the
    // lie, and the coarse sentence may name no facet as the culprit.
    const coarse = serviceBannerLine(fakeStore(all('indeterminate')));
    assert.match(coarse.text, /did not finish answering/);
    assert.doesNotMatch(coarse.text, /Claudexor ready/);
    assert.doesNotMatch(coarse.text, /was not asked/);
    assert.doesNotMatch(coarse.text, /agent accounts could not be read/);
    // A healthy read keeps the existing lifecycle sentence, unchanged.
    assert.match(serviceBannerLine(fakeStore(all('ok'))).text, /Claudexor ready/);

    // PER FACET, never one global verdict: a refused QUOTA read must not
    // withdraw the catalogue's and the accounts' authority.
    const partial = serviceBannerLine(fakeStore({ catalog: 'ok', accounts: 'ok', quota: 'failed' }));
    assert.match(partial.text, /Your subscription limits could not be read/);
    // The reassurance covers the facets that GENUINELY read, named one by one —
    // "everything else" was written for a single failure and stayed true only
    // by accident.
    assert.match(partial.text, /Your agents and agent accounts were read normally/);
});

// ---------------------------------------------------------------------------
// Per-facet provenance has to reach the PIXELS, not just the banner. This panel
// renders three reads at once — the rows come from the accounts read, the
// windows from the quota read, the bare Connect rows from the catalog — and it
// used to render all three off the retained snapshot while consulting one facet
// for the sentence above them. After a refused read it said nothing could be
// listed while a stale row sat below it, still labeled "verified live".
// ---------------------------------------------------------------------------

function storeWithReads(reads, extra = {}) {
    // A REAL store over a payload that carries per-facet stamps, so the whole
    // chain (wire → facets → copy) is exercised, not a hand-built double of it.
    // FUTURE-COMPATIBILITY: the live producer does not emit `reads` yet (see
    // the golden test in claudexor_status_store.test.js) — these cases pin what
    // the store does the day the stamp lands, and the coarse legacy behaviour
    // is pinned separately below.
    return createClaudexorStatusStore({
        fetchImpl: async () => ({
            ok: true,
            status: 200,
            json: async () => ({
                daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                config_dir: '/home/agent',
                harnesses: [{ id: 'codex' }],
                profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: true }], profiles: [] },
                quota: [],
                reads,
                ...extra,
            }),
        }),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
}

test('the ONE banner names the SECOND gap, instead of dropping it silently', async () => {
    // Written against the per-section status line, and kept against the tab's
    // ONE banner that replaced it: the sections moved onto the Agents tab and
    // the three scattered service sentences became one. The guarantee is the
    // same and is what this pins — a surface renders several facets, and none
    // of them may fail unmentioned. Driven through a REAL store over a stamped
    // payload, so the whole chain (wire → facets → copy) runs.
    const quotaDied = storeWithReads({ catalog: 'ok', accounts: 'ok', quota: 'failed' });
    await quotaDied.refresh();
    const line = serviceBannerLine(quotaDied);
    assert.match(line.text, /Your subscription limits could not be read/, 'the refused read is NAMED');
    assert.match(line.text, /Your agents and agent accounts were read normally/,
        'and the two facets that landed keep their authority, one by one');
    assert.equal(line.tone, 'warn');
    quotaDied.dispose();

    const catalogDied = storeWithReads({ catalog: 'failed', accounts: 'ok', quota: 'ok' });
    await catalogDied.refresh();
    assert.match(serviceBannerLine(catalogDied).text, /Your agents could not be read/);
    catalogDied.dispose();

    const accountsDied = storeWithReads({ catalog: 'ok', accounts: 'failed', quota: 'ok' });
    await accountsDied.refresh();
    const accountsLine = serviceBannerLine(accountsDied);
    assert.match(accountsLine.text, /Your agent accounts could not be read/);
    assert.doesNotMatch(accountsLine.text, /Claudexor ready/, 'never the green line over an undelivered list');
    // A daemon that could not be read is NEVER reported as a daemon nobody
    // asked — the live endpoint answers `unreachable` for a single refused
    // read, and the panel used to print "the agent daemon is not running".
    assert.doesNotMatch(accountsLine.text, /was not asked/);
    assert.doesNotMatch(accountsLine.text, /not running/);
    accountsDied.dispose();

    // All three read: the ordinary lifecycle sentence, nothing appended.
    const healthy = storeWithReads({ catalog: 'ok', accounts: 'ok', quota: 'ok' });
    await healthy.refresh();
    assert.match(serviceBannerLine(healthy).text, /Claudexor ready/);
    assert.doesNotMatch(serviceBannerLine(healthy).text, /could not be read/);
    healthy.dispose();

    // All three in the SAME gap: ONE sentence covering all of them, never one
    // per facet — and never the refused-read wording over a read nobody made.
    const allDown = storeWithReads({ catalog: 'not_read', accounts: 'not_read', quota: 'not_read' });
    await allDown.refresh();
    const down = serviceBannerLine(allDown);
    assert.match(down.text, /agents, accounts and limits/);
    assert.equal(down.text.match(/could not be read/g), null, 'one sentence covers all three');
    assert.doesNotMatch(down.text, /were read normally/, 'nothing left to reassure about');
    allDown.dispose();

    const allFailed = storeWithReads({ catalog: 'failed', accounts: 'failed', quota: 'failed' });
    await allFailed.refresh();
    // The banner leads with no facet, so every failed subject rides ONE
    // sentence. The rule the section line encoded — a secondary subject is
    // never dropped because its STATE matched the primary's — survives as
    // "all three subjects are named, once".
    const failedLine = serviceBannerLine(allFailed).text;
    assert.match(failedLine, /could not be read/);
    assert.match(failedLine, /agents, accounts and limits/);
    assert.doesNotMatch(failedLine, /were read normally/, 'nothing landed to reassure about');
    allFailed.dispose();
});

test("the LEGACY wire's global refusal blames no facet and quotes no facet's error", async () => {
    // Today's producer: no `reads` stamp, one global `unreachable`, and the
    // successful catalog + account data sitting in the very same payload. The
    // round-one fix turned that into three per-facet failures and hung the
    // QUOTA probe's error off the ACCOUNTS sentence — an accusation aimed at a
    // read that had succeeded. (Golden payload: fixtures/status_quota_refused.)
    const legacy = createClaudexorStatusStore({
        fetchImpl: async () => ({
            ok: true,
            status: 200,
            json: async () => ({
                daemon: {
                    state: 'unreachable', engine_version: '3.3.11', runtime: {},
                    last_error: 'quota_probe_failed: quota read refused by the daemon',
                },
                config_dir: '/home/agent',
                harnesses: [{ id: 'claude' }],
                profiles: { harnessAccounts: [{ harness_id: 'claude', native_login_detected: true }], profiles: [] },
                quota: [],
            }),
        }),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    await legacy.refresh();
    const text = serviceBannerLine(legacy).text;
    assert.doesNotMatch(text, /Your agent accounts could not be read/,
        'the ACCOUNTS read succeeded — its data is in this very payload');
    assert.doesNotMatch(text, /Agents .*were not read/,
        'and so did the CATALOG read: no per-facet verdict may be minted here');
    assert.doesNotMatch(text, /not running/);
    assert.match(text, /did not finish answering/, 'one coarse, global sentence');
    assert.match(text, /quota_probe_failed/, "with the daemon's own global reason");
    assert.equal(legacy.accountsKnown, false, 'and nothing is claimed as discovered');
    legacy.dispose();
});

test('each row projection is gated by ITS OWN facet, and a stale value says it is last known', () => {
    const row = {
        harness: 'codex', profile_id: '', kind: 'native', identity: {},
        status: { verification: 'passed', verification_source: 'vendor', last_verified_at: '2026-08-09' },
    };
    const payload = { quota: [{
        subject: { harness: 'codex', subject_id: '' },
        freshness: 'fresh',
        constraints: [{ used_ratio: 1.0, resets_at: '2026-08-09T12:00:00Z' }],
    }] };

    // Both facets read: exactly today's row, in the two-line anatomy — line 1
    // the account and its status, line 2 the humanized metadata (D-10).
    const fresh = accountRowFacts(row, payload, { accountsRead: 'ok', quotaRead: 'ok' });
    assert.equal(fresh.badge.tone, 'ok');
    assert.equal(fresh.badge.label, 'Verified live');
    // Honest naming (unified-accounts sprint): the retired "Default CLI
    // login" claimed a separate account TYPE; an identity-less legacy
    // pseudo-row is simply the default account.
    assert.equal(fresh.name, 'Default account');
    assert.equal(fresh.quota.exhausted, true);
    assert.match(fresh.quota.label, /^Limit reached/);
    assert.match(fresh.meta, /Limit reached/);

    // The QUOTA read refused: nothing may be claimed about the window, so the
    // remembered percentage is not re-shown as current — and the row stops
    // being painted red, because the exhausted styling is a claim about RIGHT
    // NOW and the reset it promises may already have come.
    const staleQuota = accountRowFacts(row, payload, { accountsRead: 'ok', quotaRead: 'failed' });
    assert.equal(staleQuota.badge.tone, 'ok', 'the accounts facet is untouched by the quota gap');
    assert.equal(staleQuota.quota.exhausted, false);
    assert.equal(staleQuota.quota.label, 'Limits not checked');
    assert.match(staleQuota.meta, /Limits not checked/);

    // The ACCOUNTS read refused: the row is memory of an account, so its
    // verification claim is dated — and the window, read fine, is not.
    const staleAccount = accountRowFacts(row, payload, { accountsRead: 'failed', quotaRead: 'ok' });
    assert.match(staleAccount.badge.label, /last known/);
    assert.equal(staleAccount.badge.tone, 'muted', 'no green "Verified live" over a read that never landed');
    assert.equal(staleAccount.quota.exhausted, true);
    assert.doesNotMatch(staleAccount.quota.label, /last known/);

    // The default is the fresh reading, so nothing else changes shape.
    assert.deepEqual(accountRowFacts(row, payload), fresh);
    assert.deepEqual(verificationBadge(row), fresh.badge);
});
