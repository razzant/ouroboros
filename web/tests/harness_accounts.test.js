import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
    STATUS_FACETS,
    accountLoginConfirmed,
    accountRows,
    createClaudexorStatusStore,
    statusUnavailableNote,
} from '../modules/claudexor_status_store.js';
import {
    READ_FACETS,
    accountRowFacts,
    bareRowStatusText,
    daemonAnswered,
    daemonStatusLine,
    destroyHarnessAccounts,
    initHarnessAccounts,
    normalizeProfileName,
    promptProfileName,
    quotaSummary,
    refreshActionKind,
    refreshActionLabel,
    runtimeActionLabel,
    serviceBannerLine,
    startLogin,
    unreadFacets,
    harnessFamilyMarkup,
    verificationBadge,
    wakeDaemon,
} from '../modules/harness_accounts.js';
// The login machinery moved to the shared controller module (phase 2) so the
// onboarding wizard can mount the same flow; the assertions below are
// unchanged, which is what makes the extraction behavior-preserving.
import {
    ATTACH_FALLBACK_MS,
    UNCONFIRMED_TEXT,
    attachFallbackDue,
    cancelLoginJob,
    confirmLoginLive,
    deviceCodeDisclosure,
    failureText,
    jobDetail,
    jobStateSummary,
    loginCardFace,
    loginCardHtml,
    loginInputSupport,
    loginReleaseProven,
    loginStatusLine,
    loginVerdict,
    pollResponseApplies,
    preserveCardFocus,
    submitLoginInput,
} from '../modules/harness_login_cards.js';

test('account actions stack at the app shell compact breakpoint', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8')
        .replace(/\r\n?/g, '\n');
    assert.ok(css.includes(`@media (max-width: 980px) {
    .harness-account-row {
        grid-template-columns: minmax(0, 1fr);
        grid-template-areas: "main" "meta" "actions";`),
    'account actions must drop below status before the persistent sidebar squeezes the row');
});

test('managed runtime keeps one contextual Connect intent across install, repair, and update', () => {
    const payload = (runtime, daemon = {}) => ({ daemon: { state: 'not_provisioned', runtime, ...daemon } });

    // The owner-locked dictionary is exactly four labels, independent of the
    // connected state: Connect | Install & connect | Update & connect | Fix & connect.
    assert.equal(runtimeActionLabel(payload({ state: 'missing' })), 'Install & connect');
    assert.equal(runtimeActionLabel(payload({ state: 'error' })), 'Fix & connect');
    assert.equal(runtimeActionLabel(payload({ state: 'update_available' })), 'Update & connect');
    assert.equal(runtimeActionLabel(payload({ state: 'ready' })), 'Connect');

    assert.ok(daemonStatusLine(payload({ state: 'missing' })).text.includes('installs Claudexor'));
    assert.ok(daemonStatusLine(payload({ state: 'ready', version: '3.3.7' })).text.includes('3.3.7 is ready'));
    assert.ok(daemonStatusLine(payload({ state: 'installing', target_version: '3.3.7' })).text.includes('Claudexor 3.3.7'));
    const staged = daemonStatusLine(payload(
        { state: 'update_staged', staged_version: '3.3.7' },
        { state: 'running', engine_version: '3.2.1' },
    ));
    assert.equal(staged.tone, 'warn');
    assert.ok(staged.text.includes('3.3.7 is ready'));
    assert.ok(staged.text.includes('Engine 3.2.1 keeps running'));
    const repair = daemonStatusLine(payload({ state: 'error', last_error: 'checksum mismatch' }));
    assert.equal(repair.tone, 'error');
    assert.ok(repair.text.includes('Connect retries automatically'));
});

test('a slow first read says it is checking, and an idle daemon is not dressed as a fault', () => {
    // Owner report (2026-08-08): the panel sat silent for tens of seconds and then
    // showed a WARN line about the daemon "not answering" — indistinguishable from
    // breakage. Both faces are pinned: the in-flight first read announces itself
    // with its real cost, and the ordinary idle daemon reads as installed-and-lazy.
    const checking = daemonStatusLine({}, { checking: true });
    assert.equal(checking.tone, 'muted');
    assert.ok(checking.text.includes('Checking Claudexor'));
    assert.ok(/minute/.test(checking.text), 'the honest cost of the first read is stated');

    // Once ANY daemon state is known the checking line steps aside — a stable
    // line beats a flicker on every 5s poll.
    const known = daemonStatusLine(
        { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} } },
        { checking: true },
    );
    assert.equal(known.tone, 'ok');
    assert.ok(known.text.includes('3.3.13'));

    const idle = daemonStatusLine({ daemon: { state: 'stale', runtime: { state: 'ready', version: '3.3.13' } } });
    assert.equal(idle.tone, 'muted', 'a lazy daemon is not a warning');
    assert.ok(idle.text.includes('3.3.13 is installed'));
    assert.ok(/starts automatically/.test(idle.text), 'the line says what happens next');
    assert.ok(!/not answering/.test(idle.text), 'no fault language for the ordinary idle state');
});

test('the login card explains foreground runtime preparation and retries the same intent', () => {
    // No status snapshot: the card IS mid-check but has no phase evidence —
    // the honest generic, never the old "Installing or checking" hedge.
    const preparing = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: true,
        error: '', verdict: null, confirming: false,
    });
    assert.ok(preparing.includes('Checking Claudexor…'));
    assert.ok(!preparing.includes('Installing or checking'));
    assert.ok(!preparing.includes('data-login-retry'));

    // The status snapshot names the phase: a minutes-long install says so.
    const installing = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: true,
        error: '', verdict: null, confirming: false,
    }, Date.now(), { statusPayload: {
        daemon: { state: 'unreachable', runtime: { state: 'installing', target_version: '3.3.14' } },
    } });
    assert.ok(installing.includes('Installing Claudexor 3.3.14…'));

    const starting = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: true,
        error: '', verdict: null, confirming: false,
    }, Date.now(), { statusPayload: {
        daemon: { state: 'stale', runtime: { state: 'ready' } },
    } });
    assert.ok(starting.includes('Starting the Claudexor daemon…'));

    const failed = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: false,
        error: 'checksum mismatch', verdict: null, confirming: false,
    });
    assert.ok(failed.includes('checksum mismatch'));
    assert.ok(failed.includes('data-login-retry'));
    assert.ok(!failed.includes('Checking Claudexor…'));
});

// GOLDEN fixture: the real /v2/credential-profiles body, produced by PARSING a
// sample through Claudexor's own Zod ControlCredentialProfilesResponse schema
// (packages/schema/src/credential-profile.ts) — not a hand-written flat map.
// If the upstream shape drifts, regenerate this file from the schema; the JS
// must consume whatever the schema emits.
const CREDENTIAL_PROFILES_RESPONSE = JSON.parse(readFileSync(
    fileURLToPath(new URL('./fixtures/credential_profiles_response.json', import.meta.url)),
    'utf-8',
));

test('both verification statuses are honest: vendor is trusted, local is neutral, never a permanent alarm', () => {
    // Q2-а: the local status has lied before (verification: passed a minute
    // before a 401), so it must never render as trusted. Finding #2: some
    // harnesses (cursor) have NO vendor probe in the engine, so a warn-toned
    // "not verified" there is an alarm nothing can ever clear — the local
    // state stays labeled unverified in WORDS, in a neutral tone.
    const vendor = verificationBadge({ status: {
        verification: 'passed', verification_source: 'vendor', last_verified_at: '2099-08-03T10:00:00Z',
    } });
    assert.equal(vendor.tone, 'ok');
    assert.equal(vendor.label, 'Verified live');
    // The raw ISO instant left the badge entirely (owner: a row must never lead
    // with a timestamp); accountMetaLine humanizes it on line 2 instead.
    assert.doesNotMatch(vendor.label, /\d{4}-\d{2}-\d{2}/);

    const local = verificationBadge({ status: { verification: 'passed', verification_source: 'local_store' } });
    assert.equal(local.tone, 'muted');
    // Narrower claim, narrower words — shorter, but "not verified live" stays.
    assert.equal(local.label, 'Signed in — not verified live');

    assert.equal(verificationBadge({ status: {} }).label, 'Not signed in');
    assert.deepEqual(verificationBadge({ status: { verification: 'not_run' } }),
        { tone: 'muted', label: 'Not verified' });
    assert.deepEqual(verificationBadge({ status: {
        availability: 'unknown', verification: 'not_run',
    } }), { tone: 'muted', label: 'Login status unknown' });
    assert.equal(verificationBadge({ status: { verification: 'failed', verification_source: 'vendor' } }).tone, 'error');
});

// `freshness` is a REQUIRED member of the daemon's quota snapshot
// (@claudexor/schema quota.ts, `z.enum(['fresh','stale','unknown'])`), so every
// fixture here carries it exactly as the wire does.
test('an exhausted window is shown with its reset time, never hidden', () => {
    const snapshots = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
        constraints: [{ used_ratio: 1.0, resets_at: '2099-08-04T00:00:00Z' }],
    }];
    // Owner ask: the limit text compact and understandable. The RESET is
    // humanized against a fixed now, and the raw instant stays on `resetsAt`.
    const now = Date.parse('2099-08-03T22:00:00Z');
    const summary = quotaSummary(snapshots, 'codex', 'koshak', { nowMs: now });
    assert.equal(summary.exhausted, true);
    assert.equal(summary.resetsAt, '2099-08-04T00:00:00Z');
    assert.equal(summary.label, 'Limit reached · resets in 2h');
    assert.doesNotMatch(summary.label, /\d{4}-\d{2}-\d{2}/);

    const healthy = quotaSummary([{
        subject: { harness: 'codex' }, freshness: 'fresh', constraints: [{ used_ratio: 0.42 }],
    }], 'codex');
    assert.equal(healthy.exhausted, false);
    assert.equal(healthy.label, '42% used');
    // Read, and nothing to say about THIS account: absence stated as absence.
    assert.deepEqual(quotaSummary([], 'codex'),
        { label: 'Usage unavailable', exhausted: false, resetsAt: '', tone: 'muted' });
    // A REFUSED quota read licenses no usage claim at all, while the catalogue
    // and account facets beside it stay authoritative.
    assert.equal(quotaSummary([{
        subject: { harness: 'codex' }, freshness: 'fresh', constraints: [{ used_ratio: 0.42 }],
    }], 'codex', '', { quotaRead: 'failed' }).label, 'Limits not checked');
});

test('the card reads a window on the same bar the runtime dispatches on', () => {
    // Two ways the card and the runtime disagreed about the SAME snapshot.
    //
    // 1. STALENESS. `harness_window_wait_hint` skips any snapshot that is not
    //    `fresh` ("an old reading must not block a lane"), so a stale spent window
    //    still dispatches — while the card painted it red and named a reset time,
    //    telling the owner a lane was down that was in fact serving.
    const stale = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'stale',
        constraints: [{ used_ratio: 1.0, resets_at: '2099-08-04T00:00:00Z' }],
    }];
    assert.deepEqual(quotaSummary(stale, 'codex', 'koshak'),
        { label: 'Usage unavailable', exhausted: false, resetsAt: '', tone: 'muted' });
    assert.equal(quotaSummary([{ ...stale[0], freshness: 'unknown' }], 'codex', 'koshak').exhausted, false);
    assert.equal(quotaSummary([{ ...stale[0], freshness: 'fresh' }], 'codex', 'koshak').exhausted, true);

    // 2. WHICH CONSTRAINT. The runtime spends a profile when ANY of its constraints
    //    is cooling down or full; the card read exhaustion off the single highest
    //    used_ratio, so a cooling 5-hour window hid behind a busier weekly one...
    const cooling = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
        constraints: [
            { used_ratio: 0.20, cooldown_until: '2099-08-04T00:00:00Z' },
            { used_ratio: 0.80 },
        ],
    }];
    const summary = quotaSummary(cooling, 'codex', 'koshak');
    assert.equal(summary.exhausted, true);
    assert.equal(summary.resetsAt, '2099-08-04T00:00:00Z');

    // ...and vanished entirely when the cooling constraint reported no ratio at all,
    // because a non-finite used_ratio was skipped before it could be read.
    const ratioless = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
        constraints: [{ cooldown_until: '2099-08-04T00:00:00Z' }],
    }];
    assert.equal(quotaSummary(ratioless, 'codex', 'koshak').exhausted, true);
});

test('a named profile\'s exhausted window is never reported as the default account\'s', () => {
    // The daemon stamps the DEFAULT subject with subject_id null and scopes every
    // cooldown to its own subject ("a profiled limit must never cool the default
    // subject down"). The row that names ONE account has to honour that: the old
    // `!subjectId ||` wildcard made the default row match every subject on the
    // harness and paint itself red off someone else's spent window.
    const snapshots = [
        { subject: { harness: 'codex', subject_id: null }, freshness: 'fresh',
          constraints: [{ used_ratio: 0.05 }] },
        { subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
          constraints: [{ used_ratio: 1.0, resets_at: '2099-08-04T00:00:00Z' }] },
    ];
    const defaultRow = quotaSummary(snapshots, 'codex', '');
    assert.equal(defaultRow.exhausted, false);
    assert.equal(defaultRow.label, '5% used');
    const namedRow = quotaSummary(snapshots, 'codex', 'koshak');
    assert.equal(namedRow.exhausted, true);
    assert.equal(namedRow.resetsAt, '2099-08-04T00:00:00Z');
});

test('typed quota gaps are exact-subject, neutral, and distinct in words', () => {
    const absences = [
        { subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'refresh_failed', detail: 'secret-like prose is not parsed' },
        { subject: { harness: 'claude', subject_id: 'proton3' }, reason: 'rate_limited' },
    ];
    const waiting = quotaSummary([], 'claude', 'proton4', { absences });
    assert.equal(waiting.label, 'Usage refresh failed · secret-like prose is not parsed');
    assert.equal(waiting.tone, 'muted');
    assert.equal(quotaSummary([], 'claude', 'proton3', { absences }).label,
        'Usage check rate-limited');
    for (const [reason, label] of [
        ['probe_skipped_rate_limited', 'Usage check paused after a rate limit'],
        ['poll_paced', 'Usage check paced'],
        ['not_logged_in', 'Usage unavailable · not signed in'],
    ]) {
        assert.equal(quotaSummary([], 'claude', 'proton4', { absences: [{
            subject: { harness: 'claude', subject_id: 'proton4' }, reason,
        }] }).label, label, reason);
    }
    assert.equal(quotaSummary([], 'claude', 'proton2', { absences }).label,
        'Usage unavailable', 'a sibling absence must not color this subject');
    const future = quotaSummary([], 'claude', 'proton4', { absences: [
        { subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'future_reason', detail: 'auth_revoked' },
    ] });
    assert.equal(future.tone, 'muted', 'unknown reason and detail prose stay neutral');
    assert.equal(future.label, 'Usage unavailable · auth_revoked',
        'detail is displayed as text but cannot select the warning tone');
    assert.equal(quotaSummary([], 'claude', 'proton4', { absences: [
        { subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'auth_revoked' },
    ] }).tone, 'muted');

    assert.deepEqual(quotaSummary([], 'claude', 'claude-default', {
        fallbackSubjectIds: [''],
        absences: [{
            subject: { harness: 'claude', subject_id: '' }, reason: 'auth_revoked',
        }],
    }), { label: 'Usage unavailable', exhausted: false, resetsAt: '', tone: 'muted' },
    'a legacy snapshot alias must never borrow another subject\'s auth absence');

    const exactGapOverAlias = quotaSummary([{
        subject: { harness: 'claude', subject_id: '' }, freshness: 'fresh',
        constraints: [{ used_ratio: 0.5 }],
    }], 'claude', 'claude-default', {
        fallbackSubjectIds: [''],
        absences: [{
            subject: { harness: 'claude', subject_id: 'claude-default' },
            reason: 'auth_revoked',
        }],
    });
    assert.equal(exactGapOverAlias.label, '50% used · Usage unavailable · sign-in revoked');
    assert.equal(exactGapOverAlias.tone, 'muted',
        'the exact auth verdict remains explicit in words without a second warning state');

    assert.equal(quotaSummary([], 'claude', 'proton4', { absences: [{
        subject: { harness: 'claude', subject_id: 'proton4' },
        reason: 'refresh_failed', detail: { nested: 'bad' },
    }] }).label, 'Usage refresh failed', 'malformed detail is ignored');
    assert.equal(quotaSummary([], 'claude', 'proton4', { absences: [{
        subject: { harness: 'claude', subject_id: 'proton4' },
        reason: { nested: 'bad' }, detail: 'must not survive a malformed reason',
    }] }).label, 'Usage unavailable', 'malformed reason and its detail are ignored');
});

test('a contradictory same-subject absence stays visible and fail-open', () => {
    const summary = quotaSummary([{
        subject: { harness: 'claude', subject_id: 'proton4' }, freshness: 'fresh',
        constraints: [{ used_ratio: 0.25 }],
    }], 'claude', 'proton4', { absences: [{
        subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'auth_revoked',
    }] });
    assert.equal(summary.label, '25% used · Usage unavailable · sign-in revoked');
    assert.equal(summary.tone, 'muted');

    const emptyButFresh = quotaSummary([{
        subject: { harness: 'claude', subject_id: 'proton4' }, freshness: 'fresh',
        constraints: [],
    }], 'claude', 'proton4', { absences: [{
        subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'auth_revoked',
    }] });
    assert.deepEqual(emptyButFresh,
        { label: 'Usage unavailable · sign-in revoked', exhausted: false, resetsAt: '', tone: 'muted' });
});

test('stale quota does not become a current percentage beside its typed gap', () => {
    const summary = quotaSummary([{
        subject: { harness: 'claude', subject_id: 'proton4' }, freshness: 'stale',
        constraints: [{ used_ratio: 0.65, resets_at: '2099-08-31T00:00:00Z' }],
    }], 'claude', 'proton4', { absences: [{
        subject: { harness: 'claude', subject_id: 'proton4' }, reason: 'poll_paced',
        detail: 'next poll is paced',
    }] });
    assert.deepEqual(summary, {
        label: 'Usage check paced · next poll is paced',
        exhausted: false,
        resetsAt: '',
        tone: 'muted',
    });
});

test('a model-scoped window never paints the whole account exhausted — it is a compact note', () => {
    // The daemon schema's own words (@claudexor/schema quota.ts): a non-null
    // applies_to_models is a per-model cap, and "a model-specific cap never
    // cools a different model on the same subject". Painting the whole account
    // "window exhausted" off one is the same class of misreport as the
    // wildcard-subject bug above — a block reported that will not happen.
    const subject = { harness: 'claude', subject_id: 'abstractdl' };
    const mixed = quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [
            { id: 'fable-window', label: 'Fable window', applies_to_models: ['claude-fable-5'],
              used_ratio: 1.0, resets_at: '2099-08-08T00:00:00Z' },
            { applies_to_models: null, used_ratio: 0.4 },
        ],
    }], 'claude', 'abstractdl');
    assert.equal(mixed.exhausted, false);
    // The account bar stays the GLOBAL window's; the spent scope is still said.
    assert.equal(mixed.label, '40% used · Fable window spent');

    // Scoped-only spent (cooldown, no ratio): the note IS the label, no red.
    const scopedOnly = quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [{ id: 'fable-window', label: 'Fable window',
            applies_to_models: ['claude-fable-5'], cooldown_until: '2099-08-08T00:00:00Z', used_ratio: null }],
    }], 'claude', 'abstractdl');
    assert.equal(scopedOnly.exhausted, false);
    assert.equal(scopedOnly.label, 'Fable window spent');

    // A scoped window that is merely busy says nothing at account level.
    assert.deepEqual(quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [{ label: 'Fable window', applies_to_models: ['claude-fable-5'], used_ratio: 0.8 }],
    }], 'claude', 'abstractdl'),
    { label: 'Usage unavailable', exhausted: false, resetsAt: '', tone: 'muted' });

    // A GLOBAL window (applies_to_models null/omitted = every model) keeps the
    // account-level exhausted behavior exactly as before.
    const global = quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [{ applies_to_models: null, used_ratio: 1.0, resets_at: '2099-08-08T00:00:00Z' }],
    }], 'claude', 'abstractdl');
    assert.equal(global.exhausted, true);
    assert.ok(global.label.startsWith('Limit reached'));

    // Without a label, the note falls back to the constraint id, then models.
    assert.equal(quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [{ id: 'fable_5h', applies_to_models: ['claude-fable-5'], used_ratio: 1.0 }],
    }], 'claude', 'abstractdl').label, 'fable_5h availability not proven');
});

// ---------------------------------------------------------------------------
// Add account: pywebview's WKWebView implements no window.prompt (it answers
// null silently), so the flow runs on the in-house input dialog.
// ---------------------------------------------------------------------------

test('the groups rebuild wraps in preserveCardFocus so a family-mounted login card keeps its caret', () => {
    // REGRESSION guard (fable review, roster-identity sprint): the login card
    // mounts INSIDE #harness-accounts-groups now, so the poll-tick innerHTML
    // rebuild destroys a focused paste-code/name input BEFORE the card's own
    // render can capture it. The capture must wrap the whole rebuild — the
    // innerHTML assignment and the card repaint both inside ONE
    // preserveCardFocus callback.
    const source = readFileSync(new URL('../modules/harness_accounts.js', import.meta.url), 'utf8');
    const wrap = source.indexOf('preserveCardFocus(host, () => {');
    assert.ok(wrap >= 0, 'renderRows must wrap its rebuild in preserveCardFocus');
    const rebuild = source.indexOf('host.innerHTML = accountGroups(');
    const repaint = source.indexOf('state.loginCard?.render();');
    const close = source.indexOf('\n    });', wrap);
    assert.ok(wrap < rebuild && rebuild < repaint && repaint < close,
        'both the innerHTML rebuild and the card repaint live inside the capture');
});

test('Add account never touches window.prompt and asks through the in-house dialog', async () => {
    // REGRESSION guard for the dead desktop button: the module must not call
    // window.prompt at all — under pywebview it is a silent no-op. (The call
    // form, so a comment may still name the hazard.)
    const source = readFileSync(new URL('../modules/harness_accounts.js', import.meta.url), 'utf8');
    assert.ok(!/window\s*\.\s*prompt\s*\(/.test(source));
    assert.ok(source.includes("from './confirm_dialog.js'"));

    // An already-valid name asks exactly once, for TEXT input.
    const calls = [];
    const name = await promptProfileName({ dialogImpl: async (options) => {
        calls.push(options);
        return { confirmed: true, value: 'backup' };
    } });
    assert.equal(name, 'backup');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].input, true);
    // The alphabet is stated up front, so normalization is never a surprise.
    assert.ok(calls[0].body.includes('anything else becomes "-"'));

    // Cancel, and a name that normalizes to nothing, are quiet no-ops.
    assert.equal(await promptProfileName({ dialogImpl: async () => ({ confirmed: false, value: 'x' }) }), '');
    assert.equal(await promptProfileName({ dialogImpl: async () => ({ confirmed: true, value: '   ' }) }), '');
});

test('a name normalization would change is shown back, editable, BEFORE any login starts', async () => {
    // The owner types "Работа": nothing slug-legal survives (engine contract
    // ^[a-z0-9][a-z0-9_-]{0,63}$), and starting a login under a silently
    // rewritten name is exactly the trap the prompt() flow had. The dialog re-opens with the normalized name visible
    // AND editable; only an explicit confirm of a stable name proceeds.
    const rounds = [];
    const answers = [
        { confirmed: true, value: 'Работа' },
        { confirmed: true, value: 'work-2' },
    ];
    const name = await promptProfileName({ dialogImpl: async (options) => {
        rounds.push(options);
        return answers[rounds.length - 1];
    } });
    assert.equal(name, 'work-2');
    assert.equal(rounds.length, 2);
    // Under the engine slug contract (^[a-z0-9][a-z0-9_-]{0,63}$) nothing of
    // "Работа" survives normalization: the dialog re-asks with the contract
    // spelled out instead of offering an illegal all-separator name.
    assert.ok(rounds[1].body.includes('"Работа" cannot become an account name'));
    assert.equal(rounds[1].initialValue, '');

    // Accepting the shown normalized name as-is also works (one extra round).
    const folds = [];
    const folded = await promptProfileName({ dialogImpl: async (options) => {
        folds.push(options);
        return { confirmed: true, value: folds.length === 1 ? 'Work' : options.initialValue };
    } });
    assert.equal(folded, 'work');
    assert.equal(folds.length, 2);
    assert.equal(folds[1].initialValue, 'work');

    // The normalization itself, pinned to the ENGINE slug contract
    // (^[a-z0-9][a-z0-9_-]{0,63}$): nothing slug-legal survives of "Работа",
    // so it maps to '' and the dialog re-asks instead of offering '------'.
    assert.equal(normalizeProfileName(' Work '), 'work');
    assert.equal(normalizeProfileName('Работа'), '');
    assert.equal(normalizeProfileName('a b/c'), 'a-b-c');
    assert.equal(normalizeProfileName('ok_name-1'), 'ok_name-1');
    assert.equal(normalizeProfileName(''), '');
});

test('the device-code disclosure is read only from the canonical envelope level', () => {
    const envelope = { job: { state: 'waiting_for_input' }, deviceCode: {
        flow: 'chatgptDeviceCode', verificationUrl: 'https://auth.example/device', userCode: 'ABCD-1234',
    } };
    assert.deepEqual(deviceCodeDisclosure(envelope),
        { url: 'https://auth.example/device', code: 'ABCD-1234', flow: 'chatgptDeviceCode' });
    // Mutation guard: the accidental legacy double-wrap is deliberately dead.
    assert.equal(deviceCodeDisclosure({ job: { state: 'running', deviceCode: envelope.deviceCode } }), null);
    assert.equal(deviceCodeDisclosure({ job: { state: 'running' } }), null);
    assert.equal(deviceCodeDisclosure(null), null);
});

test('a URL-ONLY disclosure renders: the flow discriminates, not the code field', () => {
    // Claudexor's SetupDeviceCodeDisclosure (packages/schema/src/setup.ts):
    // `userCode` is EMPTY for the browser-callback (`chatgpt`) and `oauth_url`
    // flows — the latter is the sign-in link a TERMINAL-mode claude/cursor login
    // prints. Requiring both fields matched neither, so a published link showed
    // nothing at all; the login card is the whole point of D30's structural face.
    for (const flow of ['oauth_url', 'chatgpt', 'oauth_url_input']) {
        const envelope = { job: { state: 'waiting_for_input' }, deviceCode: {
            flow, verificationUrl: 'https://claude.ai/oauth/authorize?x=1', userCode: '',
        } };
        assert.deepEqual(deviceCodeDisclosure(envelope),
            { url: 'https://claude.ai/oauth/authorize?x=1', code: '', flow }, flow);
        // …and the card must actually pick the structural face for it.
        assert.equal(loginCardFace({ mode: 'attach', attachCommand: 'cmd', envelope }), 'device', flow);
    }
    // A node carrying neither is still not a disclosure.
    assert.equal(deviceCodeDisclosure({ job: {}, verificationUrl: 'https://a/b' }), null);
});

test('job terminal states are read from the canonical envelope only', () => {
    assert.deepEqual(jobStateSummary({ job: { state: 'succeeded' } }),
        { state: 'succeeded', phase: '', terminal: true, succeeded: true });
    for (const bad of ['failed', 'cancelled', 'timed_out', 'not_supported', 'interrupted_unknown']) {
        const summary = jobStateSummary({ job: { state: bad } });
        assert.equal(summary.terminal, true, bad);
        assert.equal(summary.succeeded, false, bad);
    }
    assert.equal(jobStateSummary({ job: { state: 'waiting_for_input', phase: 'awaiting_user' } }).terminal, false);
    assert.equal(jobStateSummary({ state: 'succeeded' }).terminal, false,
        'the old bare job shape must not regain compatibility');
});

test('the POLLED snapshot ENVELOPE is read, so the login poll can actually terminate', () => {
    // GET /v2/setup/jobs/:id/snapshot answers ControlSetupJobSnapshot —
    // {job, cursor, sequence, deviceCode?}. Every Ouroboros operation now wraps
    // its one bare job the same way, so the controller keeps one reader.
    const envelope = {
        job: { jobId: 'j1', state: 'succeeded', phase: 'completed' },
        cursor: 'c1', sequence: 7,
    };
    assert.deepEqual(jobStateSummary(envelope),
        { state: 'succeeded', phase: 'completed', terminal: true, succeeded: true });
    assert.equal(jobStateSummary({ job: { state: 'failed', phase: 'login' } }).terminal, true);
    assert.equal(jobStateSummary({ job: { state: 'waiting_for_input' } }).terminal, false);
    // An accidental second envelope never passes as canonical.
    assert.equal(jobStateSummary({ job: envelope }).terminal, false);
});

test('account rows consume the REAL schema shape: array of {profile,status,identity} wrappers + harnessAccounts array', () => {
    // The status endpoint nests the daemon body under payload.profiles.
    const rows = accountRows({ profiles: CREDENTIAL_PROFILES_RESPONSE });
    assert.equal(rows.length, 2);  // one native pseudo-row + one registered profile

    const native = rows.find((row) => row.kind === 'native');
    assert.equal(native.harness, 'codex');  // read from harness_id (snake_case), not harnessId
    // A native login detected locally is still only local_store evidence.
    assert.equal(verificationBadge(native).label, 'Signed in — not verified live');

    const profile = rows.find((row) => row.kind === 'profile');
    // Read from the NESTED wrapper.profile.* snake_case fields, not a flat map.
    assert.equal(profile.harness, 'codex');
    assert.equal(profile.profile_id, 'koshak');
    assert.equal(profile.display_name, 'Koshak');
    assert.equal(profile.identity.email, 'koshak@example.com');
    // The vendor-verified status flows straight through from wrapper.status.
    assert.equal(verificationBadge(profile).tone, 'ok');
    assert.equal(verificationBadge(profile).label, 'Verified live');
});

test('the invented flat camelCase shape yields NOTHING (guards against the regression)', () => {
    // The exact shape an earlier draft consumed — a flat map with camelCase
    // keys and harnessAccounts-as-object. The real schema never emits it, so
    // reading it must produce zero rows, not silently-empty harness fields.
    const rows = accountRows({ profiles: {
        harnessAccounts: { codex: { native_login_detected: true } },
        profiles: [{ harnessId: 'codex', profileId: 'backup' }],
    } });
    assert.equal(rows.length, 0);
});

test('DTO end-to-end: EMPTY and MULTI-ACCOUNT schema-parsed bodies', () => {
    // Both fixtures came through Claudexor's own Zod schema. Empty body:
    // zero rows, no invented natives, no crash.
    assert.deepEqual(accountRows({ profiles: { profiles: [], harnessAccounts: [] } }), []);
    assert.deepEqual(accountRows({ profiles: {} }), []);
    assert.deepEqual(accountRows({}), []);

    const MULTI = JSON.parse(readFileSync(
        fileURLToPath(new URL('./fixtures/credential_profiles_multi.json', import.meta.url)),
        'utf-8',
    ));
    const rows = accountRows({ profiles: MULTI });
    // 2 native pseudo-rows + 3 profiles, per harness.
    assert.equal(rows.length, 5);
    assert.deepEqual(rows.filter((r) => r.kind === 'profile').map((r) => `${r.harness}:${r.profile_id}`),
        ['codex:koshak', 'codex:backup', 'claude:main']);
    // Mixed verification renders each truth on its own row.
    const byId = Object.fromEntries(rows.filter((r) => r.kind === 'profile')
        .map((r) => [r.profile_id, verificationBadge(r)]));
    assert.equal(byId.koshak.tone, 'ok');                       // vendor-verified
    assert.equal(byId.backup.label, 'Signed in — not verified live');
    assert.equal(byId.main.tone, 'error');                      // vendor said failed
    // A claude native row with no login shows "not logged in", not a lie.
    const claudeNative = rows.find((r) => r.kind === 'native' && r.harness === 'claude');
    assert.equal(verificationBadge(claudeNative).label, 'Not signed in');
});

test('the attach command is DEMOTED: never a card face, only a due fallback', () => {
    // The owner rejected terminal-first login ("Via your terminal" buttons and
    // an attach-command card body). A job with a command but nothing
    // structured renders the WAITING face; the command surfaces only through
    // attachFallbackDue as a collapsed Advanced affordance.
    const attachOnly = { attachCommand: 'CLAUDEXOR_CONFIG_DIR=/d claudexor setup attach j1', startedAtMs: 1000, envelope: { job: { state: 'waiting_for_input' } } };
    assert.equal(loginCardFace(attachOnly), 'progress');
    // The SAME job once the engine surfaces a structured OAuth disclosure:
    // the structural card wins — no terminal needed.
    assert.equal(loginCardFace({ ...attachOnly, envelope: {
        job: { state: 'waiting_for_input' },
        deviceCode: { flow: 'chatgptDeviceCode', verificationUrl: 'https://a/b', userCode: 'XY-12' },
    } }), 'device');
    // Errors outrank everything; nothing at all = progress; no job = none.
    assert.equal(loginCardFace({ error: 'nope', attachCommand: 'cmd', envelope: { job: {} } }), 'error');
    assert.equal(loginCardFace({ envelope: { job: { state: 'running' } } }), 'progress');
    assert.equal(loginCardFace(null), 'none');
});

test('card shape 2 keys on the disclosure FLOW string — the typed enum decides, no harness branching', () => {
    // The engine's 3.3.7 FINAL contract: `oauth_url_input` is the disclosure
    // flow for a job that also accepts a pasted code (claude's
    // manual-callback path); `oauth_url`/`chatgpt` stay link-only. The enum
    // decides for ANY harness — no boolean sidecar, no name fallback.
    const withInput = { job: { state: 'waiting_for_input' }, deviceCode: {
        flow: 'oauth_url_input', verificationUrl: 'https://platform.claude.com/oauth/authorize?x=1', userCode: '' } };
    assert.equal(loginInputSupport(withInput), true);
    // A URL-only disclosure: shape 1, no input — even when the harness that
    // produced it happens to be claude (the flow is the truth, not the name).
    for (const flow of ['oauth_url', 'chatgpt', 'chatgptDeviceCode']) {
        const envelope = { job: { state: 'waiting_for_input' }, deviceCode: {
            flow, verificationUrl: 'https://cursor.com/loginDeepControl?x=1', userCode: '' } };
        assert.equal(loginInputSupport(envelope), false, flow);
    }
    // No disclosure at all: no input field.
    assert.equal(loginInputSupport({ job: { state: 'running' } }), false);
    assert.equal(loginInputSupport(null), false);
});

test('the verdict never contradicts the state, and never fails off a verification-race read', () => {
    // The owner's live finding: a codex login SUCCEEDED while the card said
    // "Login failed · completed" — the engine's post-login probe read the
    // auth store codex clears at login start. Verification-flavored failures
    // are 'recheck' (judged by live account status), not final failures.
    assert.equal(loginVerdict({ job: { state: 'running', phase: 'awaiting_user' } }).kind, 'pending');
    assert.equal(loginVerdict({ job: { state: 'succeeded', phase: 'completed' } }).kind, 'success');
    for (const reason of ['capability_verification_failed', 'auth_not_ready']) {
        const verdict = loginVerdict({ job: { state: 'failed', phase: 'completed', outcome: { reason } } });
        assert.equal(verdict.kind, 'recheck', reason);
        assert.equal(verdict.reason, reason);
    }
    // A failure with NO typed reason is also unproven — recheck.
    assert.equal(loginVerdict({ job: { state: 'failed', phase: 'completed' } }).kind, 'recheck');
    // Genuine failures stay final, with their typed reason carried.
    const launch = loginVerdict({ job: { state: 'failed', outcome: { reason: 'launch_failed' } } });
    assert.deepEqual(launch, { kind: 'failure', reason: 'launch_failed' });
    assert.equal(loginVerdict({ job: { state: 'timed_out', outcome: { reason: 'timed_out' } } }).kind, 'failure');
    assert.equal(loginVerdict({ job: { state: 'cancelled', outcome: { reason: 'cancelled_by_user' } } }).kind, 'failure');
    const termination = { state: 'interrupted_unknown',
        outcome: { reason: 'termination_unconfirmed' } };
    assert.equal(loginVerdict({ job: termination }).kind, 'recovery');
    assert.equal(loginVerdict({ job: { ...termination,
        terminationReconciliation: { status: 'empty' } } }).kind, 'reconciled');
    // Wording: a real failure names its reason in words, no enum glue.
    assert.equal(failureText('launch_failed'), 'Sign-in failed — launch failed.');
});

test('the live state line renders plain words and NOTHING on a terminal job', () => {
    // "Login failed · completed" is structurally impossible: terminal jobs
    // render a verdict, and this line answers '' for them.
    assert.equal(loginStatusLine({ job: { state: 'failed', phase: 'completed' } }), '');
    assert.equal(loginStatusLine({ job: { state: 'succeeded', phase: 'completed' } }), '');
    assert.equal(loginStatusLine({ job: { state: 'queued', phase: 'preparing' } }), 'Starting the sign-in…');
    assert.equal(loginStatusLine({ job: { state: 'waiting_for_input', phase: 'launching' } }), 'Waiting for the sign-in link…');
    assert.equal(loginStatusLine({ job: { state: 'running', phase: 'verifying' } }), 'Checking the sign-in…');
    const disclosed = { job: { state: 'waiting_for_input', phase: 'awaiting_user' },
        deviceCode: { flow: 'oauth_url', verificationUrl: 'https://a/b', userCode: '' } };
    assert.equal(loginStatusLine(disclosed), 'Waiting for you to finish signing in in the browser…');
});

test('accountLoginConfirmed reads the exact harness+profile row from live status', () => {
    const payload = { profiles: {
        harnessAccounts: [
            { harness_id: 'codex', native_login_detected: true, identity: {} },
            { harness_id: 'claude', native_login_detected: false, identity: {} },
        ],
        profiles: [
            { profile: { harness_id: 'codex', profile_id: 'koshak' },
              status: { verification: 'passed', verification_source: 'vendor' }, identity: {} },
        ],
    } };
    // The default account (empty profile id) is confirmed by the daemon's own
    // local-store detection — the same evidence the row badge renders.
    assert.equal(accountLoginConfirmed(payload, 'codex', ''), true);
    assert.equal(accountLoginConfirmed(payload, 'claude', ''), false);
    // A named profile is judged by ITS row, never the native pseudo-row.
    assert.equal(accountLoginConfirmed(payload, 'codex', 'koshak'), true);
    assert.equal(accountLoginConfirmed(payload, 'codex', 'other'), false);
    assert.equal(accountLoginConfirmed({}, 'codex', ''), false);
});

function fakeResponse(status, body) {
    return { ok: status >= 200 && status < 300, status, json: async () => body };
}

test('submitLoginInput posts the code once and types the 404 capability gap (mock fetch)', async () => {
    const calls = [];
    const ok = await submitLoginInput('j 1', 'ABCD-1234', { fetchImpl: async (url, init) => {
        calls.push({ url, init });
        return fakeResponse(200, { ok: true, job: {} });
    } });
    assert.deepEqual(ok, { ok: true, degraded: false, conflict: '', error: '' });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/claudexor/login/j%201/input');
    assert.equal(calls[0].init.method, 'POST');
    assert.deepEqual(JSON.parse(calls[0].init.body), { value: 'ABCD-1234' });

    // DEGRADED-ENGINE PATH: the gateway's typed 404 (input_not_supported —
    // the engine predates the route or reaped the job) is `degraded`, so the
    // card falls back to Advanced instead of dead-ending on a raw error.
    const degraded = await submitLoginInput('j1', 'X', {
        fetchImpl: async () => fakeResponse(404, { error: 'input route not available', code: 'input_not_supported' }),
    });
    assert.equal(degraded.ok, false);
    assert.equal(degraded.degraded, true);
    // Any other failure is an ordinary error, NOT a capability degrade.
    const busy = await submitLoginInput('j1', 'X', { fetchImpl: async () => fakeResponse(503, { error: 'daemon down' }) });
    assert.deepEqual(busy, { ok: false, degraded: false, conflict: '', error: 'daemon down' });
    const dead = await submitLoginInput('j1', 'X', { fetchImpl: async () => { throw new Error('network gone'); } });
    assert.equal(dead.degraded, false);
    assert.ok(dead.error.includes('network gone'));
});

test('a 409 input conflict carries the engine code: the callback already completed', async () => {
    // Typed by the engine (final contract): setup_input_not_applicable means
    // the flow moved past the code step — e.g. claude's localhost callback
    // completed on its own. An ANSWER, not an error: the card shows a quiet
    // "no code needed" note and lets the job poll land the verdict.
    const result = await submitLoginInput('j1', 'ABCD', {
        fetchImpl: async () => fakeResponse(409, {
            error: 'input is not applicable to this flow/phase',
            code: 'setup_input_not_applicable',
        }),
    });
    assert.deepEqual(result, {
        ok: false, degraded: false, conflict: 'setup_input_not_applicable',
        error: 'input is not applicable to this flow/phase',
    });
});

test('a 409 repeat is typed too: the server is authoritative over the double-submit guard', async () => {
    // setup_input_already_submitted: our busy/sent guard prevents UI repeats,
    // but the server owns the truth (e.g. a second tab already sent a code).
    // The card treats it as already-sent, never as a failure.
    const result = await submitLoginInput('j1', 'ABCD', {
        fetchImpl: async () => fakeResponse(409, {
            error: 'a code was already submitted for this job',
            code: 'setup_input_already_submitted',
        }),
    });
    assert.equal(result.conflict, 'setup_input_already_submitted');
    assert.equal(result.degraded, false);
    assert.equal(result.ok, false);
    // A 409 with no code still classifies as a conflict, never a raw error.
    const untyped = await submitLoginInput('j1', 'ABCD', {
        fetchImpl: async () => fakeResponse(409, { error: 'conflict' }),
    });
    assert.equal(untyped.conflict, 'conflict');
});

test('confirmLoginLive re-polls live account status briefly instead of trusting one stale read', async () => {
    // First poll: the account still looks logged out (the stale window).
    // Second poll: the login shows up — confirmed, loop ends early.
    const cold = { profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: false }], profiles: [] } };
    const warm = { profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: true }], profiles: [] } };
    let polls = 0;
    const slept = [];
    const confirmed = await confirmLoginLive('codex', '', {
        readStatus: async () => (++polls >= 2 ? warm : cold),
        attempts: 4, delayMs: 7, sleepImpl: async (ms) => { slept.push(ms); },
    });
    assert.equal(confirmed.confirmed, true);
    assert.equal(polls, 2);
    assert.deepEqual(slept, [7]);   // no sleep before the first poll
    assert.deepEqual(confirmed.payload, warm);

    // Still cold after every attempt: unconfirmed, with the last payload so
    // the caller can render the rows it actually saw.
    let coldPolls = 0;
    const unconfirmed = await confirmLoginLive('codex', '', {
        readStatus: async () => { coldPolls += 1; return cold; },
        attempts: 3, delayMs: 1, sleepImpl: async () => {},
    });
    assert.equal(unconfirmed.confirmed, false);
    assert.equal(coldPolls, 3);   // bounded — it does not poll forever
    assert.deepEqual(unconfirmed.payload, cold);

    // A card closed mid-check aborts without a verdict.
    const stale = await confirmLoginLive('codex', '', {
        readStatus: async () => cold,
        attempts: 3, delayMs: 1, sleepImpl: async () => {}, isStale: () => true,
    });
    assert.equal(stale.stale, true);
});

test('the Advanced fallback is due on a disclosure that never comes, or an engine that predates the modes', () => {
    const base = { attachCommand: 'CLAUDEXOR_CONFIG_DIR=/d claudexor setup attach j1', startedAtMs: 100000, engineDegraded: false, envelope: { job: { state: 'waiting_for_input' } } };
    // Inside the grace window: not due — the card just says it is waiting.
    assert.equal(attachFallbackDue(base, 100000 + ATTACH_FALLBACK_MS - 1), false);
    // Window elapsed with no disclosure: due.
    assert.equal(attachFallbackDue(base, 100000 + ATTACH_FALLBACK_MS), true);
    // An engine the create answer flagged as pre-disclosure: due immediately.
    assert.equal(attachFallbackDue({ ...base, engineDegraded: true }, 100001), true);
    // A rendered disclosure keeps the fallback hidden (link-first, always)…
    const disclosed = { ...base, envelope: { job: { state: 'waiting_for_input' }, deviceCode: {
        flow: 'oauth_url', verificationUrl: 'https://a/b', userCode: '' } } };
    assert.equal(attachFallbackDue(disclosed, 100000 + ATTACH_FALLBACK_MS * 2), false);
    // …unless the engine is degraded (the input route 404'd mid-flow).
    assert.equal(attachFallbackDue({ ...disclosed, engineDegraded: true }, 100001), true);
    // No command = nothing to fall back to (the daemon-hosted codex flow).
    assert.equal(attachFallbackDue({ ...base, attachCommand: '' }, 100000 + ATTACH_FALLBACK_MS * 2), false);
    assert.equal(attachFallbackDue(null, 999999), false);
});

// ---------------------------------------------------------------------------
// Card rendering: the sign-in link is a PRIMARY click target, the verdict owns
// the card once it lands, and a re-check that ran out is not a failure.
// ---------------------------------------------------------------------------

function cardWithUrl(url, extra = {}) {
    return {
        harness: 'claude', profile: '', jobId: 'j1', attachCommand: '', startedAtMs: 0,
        envelope: { job: { state: 'waiting_for_input', phase: 'awaiting_user' },
            deviceCode: { flow: 'oauth_url', verificationUrl: url, userCode: '' } },
        ...extra,
    };
}

test('the disclosed sign-in URL is rendered only for http/https, through the house helper', () => {
    // The link is the card's primary action now — one click, engine-supplied
    // text. utils.safeExternalHrefAttr is the single house gate for that
    // (http/https only, escaped by the helper), and everything else must
    // render NO clickable link rather than a scheme the browser will execute.
    const safe = loginCardHtml(cardWithUrl('https://platform.claude.com/oauth/authorize?x=1&y=2'), 0);
    assert.ok(safe.includes('href="https://platform.claude.com/oauth/authorize?x=1&amp;y=2"'));
    assert.ok(safe.includes('data-open-signin'));
    assert.ok(loginCardHtml(cardWithUrl('http://127.0.0.1:1455/callback'), 0).includes('data-open-signin'));

    for (const hostile of [
        'javascript:alert(document.cookie)',
        'JavaScript:alert(1)',
        'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==',
        'vbscript:msgbox(1)',
        'file:///etc/passwd',
        'not a url at all',
        '//evil.example/oauth',
    ]) {
        const html = loginCardHtml(cardWithUrl(hostile), 0);
        assert.ok(!html.includes('data-open-signin'), hostile);
        assert.ok(!html.includes('href='), hostile);
        assert.ok(html.includes('data-unsafe-signin-link'), hostile);
        // …and the raw scheme never reaches the DOM as an attribute value.
        assert.ok(!html.includes(hostile), hostile);
    }
});

test('a settled verdict silences the live status line, so the card never says both', () => {
    // The owner hit a card reading "Waiting for the sign-in link…" beside a
    // verdict: an overlapping poll tick applied a snapshot captured before the
    // job settled. Two guards, and this is the rendering half.
    const pending = cardWithUrl('https://a.example/b');
    assert.ok(loginCardHtml(pending, 0).includes('data-login-state'));

    const settled = { ...pending, verdict: { kind: 'success', reason: '' } };
    const html = loginCardHtml(settled, 0);
    assert.ok(!html.includes('data-login-state'));
    assert.ok(html.includes('Connected.'));
    // Same while the live re-check is deciding.
    assert.ok(!loginCardHtml({ ...pending, confirming: true }, 0).includes('data-login-state'));
});

test('an exhausted re-check says the sign-in is UNCONFIRMED, never that it failed', () => {
    // The row it waits for routinely lands a tick after the bounded re-poll
    // gives up, so a hard "Sign-in failed" there is a lie about a login that
    // may have succeeded. A genuine typed failure keeps its own wording.
    const unconfirmed = loginCardHtml(cardWithUrl('https://a.example/b', {
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' } }), 0);
    assert.ok(unconfirmed.includes(UNCONFIRMED_TEXT));
    assert.ok(!unconfirmed.includes('Sign-in failed'));
    assert.ok(UNCONFIRMED_TEXT.includes('Refresh'));

    const failed = loginCardHtml(cardWithUrl('https://a.example/b', {
        verdict: { kind: 'failure', reason: 'launch_failed' } }), 0);
    assert.ok(failed.includes(failureText('launch_failed')));
    assert.ok(!failed.includes(UNCONFIRMED_TEXT));
});

test("a settled non-success verdict carries the engine's own explanation", () => {
    // The masking bug the owner hit: a codex login ended `auth_not_ready` and
    // the card showed only the fixed UNCONFIRMED_TEXT ("check the account row
    // above"), which reads as "wait a moment" — while the daemon had already
    // settled it terminally and said why. That sentence was in the snapshot
    // the card was holding and reached no reader; the two verdict texts are
    // fixed constants, so nothing else could ever carry it.
    const message = 'codex native session was not ready before the verification'
        + ' deadline: native Codex session is not logged in';
    // The canonical envelope carries the bare job exactly once.
    const nested = cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', phase: 'completed', message } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    });
    const unconfirmed = loginCardHtml(nested, 0);
    assert.ok(unconfirmed.includes('data-login-detail'));
    assert.ok(unconfirmed.includes(message));
    // The verdict wording itself is unchanged — this is additive.
    assert.ok(unconfirmed.includes(UNCONFIRMED_TEXT));

    // A typed failure gets it too: its reason is a category, not a sentence.
    assert.ok(loginCardHtml({ ...nested, verdict: { kind: 'failure', reason: 'launch_failed' } }, 0)
        .includes('data-login-detail'));

    // Never beside "Connected." (a stale message must not contradict success),
    // and never while the job is unsettled (the status line owns the card).
    assert.ok(!loginCardHtml({ ...nested, verdict: { kind: 'success', reason: '' } }, 0)
        .includes('data-login-detail'));
    assert.ok(!loginCardHtml({ ...nested, verdict: null }, 0).includes('data-login-detail'));
    assert.ok(!loginCardHtml({ ...nested, confirming: true, verdict: null }, 0)
        .includes('data-login-detail'));

    // Engine-supplied text is escaped like every other disclosure on this card.
    const hostile = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: '<img src=x onerror=alert(1)>' } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    }), 0);
    assert.ok(!hostile.includes('<img'));
    assert.ok(hostile.includes('&lt;img'));

    // jobDetail itself: both levels, trimmed, and total over junk.
    assert.equal(jobDetail({ job: { message: '  hi  ' } }), 'hi');
    assert.equal(jobDetail({ job: { message: 'deep' } }), 'deep');
    assert.equal(jobDetail({ message: 'legacy' }), '', 'the old bare shape is rejected');
    assert.equal(jobDetail({ job: { message: 42 } }), '');
    assert.equal(jobDetail({ job: {} }), '');
    assert.equal(jobDetail(null), '');
});

test("the engine explanation reaches the card only through the canonical envelope", () => {
    const message = 'native Codex session is not logged in';
    const levels = {
        canonical: { job: { state: 'failed', phase: 'completed', message } },
        accidental_double_wrap: { job: { job: { state: 'failed', phase: 'completed', message } } },
    };
    for (const [label, envelope] of Object.entries(levels)) {
        const html = loginCardHtml(cardWithUrl('https://a.example/b', {
            envelope, verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
        }), 0);
        assert.equal(html.includes(message), label === 'canonical', label);
    }
    assert.equal(jobDetail({ message: 'outer', job: { message: 'inner' } }), 'inner');
});

test("the engine explanation is escaped in full and never truncated", () => {
    // Untrusted external text on an owner-facing surface, so two separate
    // properties. ESCAPING: the existing suite asserts `<img …>` only, while the
    // house helper escapes six characters — an unescaped `&` or quote is the same
    // class of defect one character over, and this line sits inside an element
    // whose attributes are built by the same interpolation.
    const hostile = `Tom & Jerry's "quoted" <b>bold</b> \`tick\``;
    const html = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: hostile } },
        verdict: { kind: 'failure', reason: 'launch_failed' },
    }), 0);
    for (const raw of ['&', '<', '>', '"', "'", '`']) {
        // Each hostile character reaches the DOM only in escaped form: the raw
        // one may still appear as HTML the card itself wrote (its own tags), so
        // the assertion is on the escaped entity being present…
        assert.ok(html.includes({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;',
        }[raw]), raw);
    }
    // …and on no fragment of the payload surviving as live markup.
    assert.ok(!html.includes('<b>bold</b>'));
    assert.ok(html.includes('&lt;b&gt;bold&lt;/b&gt;'));

    // NO TRUNCATION (BIBLE P1): this is the only place a settled login says WHY,
    // so a long engine sentence must arrive whole. The daemon's real ones already
    // chain a cause onto a summary; nothing bounds their length.
    const long = `${'the daemon explained at length: '.repeat(80)}end.`;
    const longHtml = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: long } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    }), 0);
    assert.ok(longHtml.includes(long));
    assert.ok(!longHtml.includes('…]'));   // no omission marker of any house shape
});

test('a settled failure with NO engine sentence renders the verdict alone', () => {
    // The absence path, in the render surface rather than only on jobDetail():
    // most settled jobs carry no `message` at all, so the common case must add
    // no empty element and — the specific hazard of interpolating an optional
    // field — no stringified `undefined`/`null` where a sentence would go.
    for (const envelope of [
        { job: { state: 'failed', phase: 'completed' } },          // absent
        { job: { state: 'failed', message: '' } },                 // empty
        { job: { state: 'failed', message: '   ' } },              // whitespace
        { job: { state: 'failed', message: null } },               // explicit null
    ]) {
        const html = loginCardHtml(cardWithUrl('https://a.example/b', {
            envelope, verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
        }), 0);
        assert.ok(!html.includes('data-login-detail'), JSON.stringify(envelope));
        assert.ok(!html.includes('undefined'), JSON.stringify(envelope));
        assert.ok(!html.includes('null'), JSON.stringify(envelope));
        // The verdict itself is untouched by the missing detail.
        assert.ok(html.includes(UNCONFIRMED_TEXT), JSON.stringify(envelope));
    }
});

test('the verify-race incident, composed end to end: recheck runs out and the card still says why', async () => {
    // The owner's actual incident shape. Its three steps are each asserted
    // above in isolation, which is exactly how the defect survived: every part
    // worked and the composition still rendered only a fixed constant. This
    // walks the same steps the settle path walks, in order, on one job.
    //
    // (settleVerdict itself is not exported — it re-renders the live DOM — so
    // this composes the exported steps rather than executing that function. It
    // pins the CHAIN, not settleVerdict's own wiring; that remains untested.)
    const message = 'codex native session was not ready before the verification'
        + ' deadline: native Codex session is not logged in';
    const envelope = { job: { state: 'failed', phase: 'completed', message, outcome: { reason: 'auth_not_ready' } } };

    // 1. The job settled failed, but on a reason a verification race fabricates.
    const verdict = loginVerdict(envelope);
    assert.equal(verdict.kind, 'recheck');
    assert.equal(verdict.reason, 'auth_not_ready');

    // 2. The bounded live re-check never sees the row appear.
    const cold = { profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: false }], profiles: [] } };
    const check = await confirmLoginLive('codex', '', {
        fetchImpl: async () => fakeResponse(200, cold),
        attempts: 2, delayMs: 1, sleepImpl: async () => {},
    });
    assert.equal(check.confirmed, false);

    // 3. So the card takes the unconfirmed verdict — and BOTH halves land: the
    //    honest "unknown" wording AND the daemon's own sentence. Before the fix
    //    step 3 produced the constant alone, which reads as "wait a moment" for
    //    a job the daemon had already settled terminally.
    const html = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope, verdict: check.confirmed ? { kind: 'success', reason: '' } : { kind: 'unconfirmed', reason: verdict.reason },
    }), 0);
    assert.ok(html.includes(UNCONFIRMED_TEXT));
    assert.ok(html.includes(message));
    assert.ok(!html.includes('Sign-in failed'));
});

test('a poll answer applies only to the job it was captured for, and only while unsettled', () => {
    // The ordering rule behind the contradictory card: two overlapping async
    // ticks can land out of order, so an OLDER snapshot must never be written
    // over a job that has already settled — or onto a card that has since been
    // closed or reopened for another account.
    const active = { jobId: 'j1' };
    assert.equal(pollResponseApplies(active, active), true);
    assert.equal(pollResponseApplies(active, { jobId: 'j2' }), false);   // reopened
    assert.equal(pollResponseApplies(active, null), false);              // closed
    assert.equal(pollResponseApplies(null, null), false);
    for (const kind of ['success', 'recovery', 'reconciled', 'unavailable']) {
        const owned = { ...active, verdict: { kind } };
        assert.equal(pollResponseApplies(owned, owned), false, kind);
    }
    const confirming = { jobId: 'j1', confirming: true };
    assert.equal(pollResponseApplies(confirming, confirming), false);
});

// ---------------------------------------------------------------------------
// The 3-second poll re-render must not eat the caret. Minimal element stubs
// (the repo's house idiom — no jsdom) plus node's fake timers, so the re-render
// cadence itself is what the assertion runs through.
// ---------------------------------------------------------------------------

function fakeCodeInput({ disabled = false, start = 3, end = 5 } = {}) {
    const calls = { focus: 0, range: null };
    return {
        disabled, value: 'ABCD-1234', selectionStart: start, selectionEnd: end,
        hasAttribute: (name) => name === 'data-login-code-input',
        focus() { calls.focus += 1; },
        setSelectionRange(from, to) { calls.range = [from, to]; },
        calls,
    };
}

function fakeCardHost(replacement, focused) {
    return {
        swaps: 0,
        contains: (node) => node === focused,
        querySelector: () => replacement,
    };
}

test('the paste-code field survives every poll re-render, caret and selection intact', (t) => {
    t.mock.timers.enable({ apis: ['setInterval'] });
    const typing = fakeCodeInput({ start: 3, end: 5 });
    const replacement = fakeCodeInput({ start: 0, end: 0 });
    const host = fakeCardHost(replacement, typing);
    const doc = { activeElement: typing };
    // Exactly what the job poll does: swap the card's DOM on every tick.
    setInterval(() => preserveCardFocus(host, () => { host.swaps += 1; }, doc), 3000);

    t.mock.timers.tick(3000);
    assert.equal(host.swaps, 1);
    assert.equal(replacement.calls.focus, 1);
    assert.deepEqual(replacement.calls.range, [3, 5]);
    t.mock.timers.tick(3000);
    assert.equal(host.swaps, 2);
    assert.equal(replacement.calls.focus, 2, 'every tick restores focus, not just the first');
    t.mock.timers.reset();
});

test('a re-render never STEALS focus, and never focuses a field the code already left', () => {
    // Nothing in the card focused: the swap happens, the caret stays wherever
    // the owner actually put it (another field, another section).
    const elsewhere = { hasAttribute: () => false };
    const replacement = fakeCodeInput();
    const host = fakeCardHost(replacement, null);
    preserveCardFocus(host, () => { host.swaps += 1; }, { activeElement: elsewhere });
    assert.equal(host.swaps, 1);
    assert.equal(replacement.calls.focus, 0);

    // Focused, but the code was accepted meanwhile: the replacement renders
    // disabled and must not be focused (nor asked for a selection range).
    const typing = fakeCodeInput();
    const sent = fakeCodeInput({ disabled: true });
    const host2 = fakeCardHost(sent, typing);
    preserveCardFocus(host2, () => { host2.swaps += 1; }, { activeElement: typing });
    assert.equal(host2.swaps, 1);
    assert.equal(sent.calls.focus, 0);
    assert.equal(sent.calls.range, null);

    // No document at all (module imported in node): the swap still runs.
    const host3 = fakeCardHost(replacement, null);
    preserveCardFocus(host3, () => { host3.swaps += 1; }, null);
    assert.equal(host3.swaps, 1);
});


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
        status: { verification: 'passed', verification_source: 'vendor', last_verified_at: '2099-08-09' },
    };
    const payload = { quota: [{
        subject: { harness: 'codex', subject_id: '' },
        freshness: 'fresh',
        constraints: [{ used_ratio: 1.0, resets_at: '2099-08-09T12:00:00Z' }],
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

// ---------------------------------------------------------------------------
// The custody verdict has a CONSUMER. `dispose()` answers asynchronously
// whether the daemon still runs the login, and the section used to drop that
// answer and rebuild immediately — so the advertised "one live login" held only
// inside one controller instance, and a remount created a second one.
// ---------------------------------------------------------------------------

function fakeElement(id) {
    const el = {
        id,
        innerHTML: '',
        textContent: '',
        dataset: {},
        offsetParent: {},
        listeners: [],
        addEventListener(type, fn) { el.listeners.push([type, fn]); },
        removeEventListener() {},
        querySelector: () => null,
        querySelectorAll: () => [],
        contains: () => false,
        closest: () => null,
    };
    return el;
}

function mountSection() {
    const elements = {};
    // The ids the Agents tab REALLY renders. `harness-accounts-rows` and
    // `harness-daemon-status` were the pre-tab markup; building them here let
    // the refusal below "reach the owner" through a node production no longer
    // has, so an empty, silently unmounted panel passed as a warning shown.
    for (const id of ['agents-service-banner', 'harness-login-card',
        'btn-harness-refresh']) elements[id] = fakeElement(id);
    const doc = {
        hidden: false,
        activeElement: null,
        getElementById: (id) => elements[id] || null,
        addEventListener() {}, removeEventListener() {},
    };
    const win = { addEventListener() {}, removeEventListener() {} };
    return { elements, doc, win };
}

function mountInteractiveAccountsSection() {
    const mounted = mountSection();
    const loginButton = {
        listeners: [],
        addEventListener(type, fn) { loginButton.listeners.push([type, fn]); },
    };
    const row = {
        dataset: { harness: 'claude', profile: 'work' },
    };
    loginButton.closest = () => row;
    const groups = {
        _html: '',
        set innerHTML(value) { groups._html = String(value); },
        get innerHTML() { return groups._html; },
        querySelectorAll(selector) {
            return selector === '[data-harness-login]' ? [loginButton] : [];
        },
    };
    mounted.elements['harness-accounts-groups'] = groups;
    return { ...mounted, loginButton };
}

function captureCardControls(element) {
    const controls = new Map();
    element.querySelector = (selector) => {
        const marker = selector.match(/\[([^\]]+)\]/)?.[1] || '';
        if (!marker || !element.innerHTML.includes(marker)) return null;
        const control = {
            listeners: [],
            addEventListener(type, fn) { control.listeners.push([type, fn]); },
        };
        controls.set(selector, control);
        return control;
    };
    return (selector, type = 'click') => controls.get(selector)?.listeners
        .filter(([kind]) => kind === type).map(([, fn]) => fn).at(-1);
}

test('Settings destroy detaches locally, and explicit Connect recreates the disposed controller', async () => {
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const priorFetch = globalThis.fetch;
    globalThis.document = doc;
    globalThis.window = win;

    const calls = [];
    let creates = 0;
    // The login controller talks through the app's own apiFetch (global fetch);
    // the status store is injected below, so only login traffic lands here.
    globalThis.fetch = async (url, init = {}) => {
        calls.push(`${init.method || 'GET'} ${url}`);
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            creates += 1;
            return creates === 1
                ? fakeResponse(200, { job_id: 'fenced-job', job: { state: 'running' } })
                : fakeResponse(200, { job_id: 'fenced-job', job: {
                    state: 'interrupted_unknown',
                    outcome: { reason: 'termination_unconfirmed' },
                } });
        }
        if (init.method === 'DELETE') return fakeResponse(200, { job: { state: 'cancelled' } });
        return fakeResponse(200, { job: { state: 'running' } });
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => fakeResponse(200, {
            daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
            config_dir: '/home/agent',
            harnesses: [{ id: 'codex' }],
            profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: false }], profiles: [] },
            quota: [],
        }),
        doc,
    });

    try {
        assert.equal(await initHarnessAccounts({ store }), true, 'a clean mount succeeds');
        await startLogin('codex', '');
        assert.equal(creates, 1);

        const mountedCard = elements['harness-login-card'].innerHTML;
        const mountedLifecycle = calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length;
        await store.refresh();
        assert.equal(elements['harness-login-card'].innerHTML, mountedCard,
            'ordinary mounted Settings refresh/hide-show does not tear down the card');
        assert.equal(calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length,
            mountedLifecycle, 'ordinary mounted Settings activity starts no lifecycle request');

        const lifecycleBefore = calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length;
        assert.equal(destroyHarnessAccounts(), true);
        assert.equal(calls.filter((c) => /\/api\/claudexor\/login/.test(c)).length, lifecycleBefore,
            'exported destroy initiates zero create/DELETE/reconcile requests');
        assert.equal(elements['harness-login-card'].innerHTML, '');
        assert.doesNotMatch(elements['agents-service-banner'].textContent, /holding off|could not be cancelled/);

        // A detached old handler/exported call cannot recreate outside a mount.
        await startLogin('codex', '');
        assert.equal(creates, 1);

        // Re-init owns no hidden lifecycle mutation. The next explicit Connect
        // creates exactly once and re-adopts the daemon's same fenced job.
        assert.equal(await initHarnessAccounts({ store }), true);
        const beforeConnect = creates;
        await startLogin('codex', '');
        assert.equal(creates, beforeConnect + 1);
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/,
            'the recreated controller adopted the fence into recovery');
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
        globalThis.fetch = priorFetch;
    }
});

test('Settings explicit Connect recreates a cached controller disposed by recovery-face Close', async () => {
    const { elements, doc, win } = mountSection();
    const control = captureCardControls(elements['harness-login-card']);
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const priorFetch = globalThis.fetch;
    globalThis.document = doc;
    globalThis.window = win;
    let creates = 0;
    let deletes = 0;
    const retainedJob = { state: 'interrupted_unknown',
        outcome: { reason: 'termination_unconfirmed' } };
    globalThis.fetch = async (url, init = {}) => {
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            creates += 1;
            return fakeResponse(200, { job_id: 'same-fenced-job',
                job: creates === 1 ? { state: 'running' } : retainedJob });
        }
        if (init.method === 'DELETE') {
            deletes += 1;
            return fakeResponse(200, { job: retainedJob });
        }
        return fakeResponse(200, { job: { state: 'running' } });
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => fakeResponse(200, {
            daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
            config_dir: '/home/agent', harnesses: [{ id: 'codex' }],
            profiles: { harnessAccounts: [], profiles: [] }, quota: [],
        }),
        doc,
    });
    try {
        await initHarnessAccounts({ store });
        await startLogin('codex', '');
        assert.equal(creates, 1);

        await control('[data-login-dismiss]')();
        assert.equal(deletes, 1);
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/);

        const secondClose = control('[data-login-dismiss]')();
        assert.equal(elements['harness-login-card'].innerHTML, '', 'second Close detaches synchronously');
        await secondClose;
        assert.equal(deletes, 1, 'recovery-face Close does not repeat cancel');

        await startLogin('codex', '');
        assert.equal(creates, 2, 'explicit Connect built a fresh controller and created once');
        assert.match(elements['harness-login-card'].innerHTML, /data-login-reconcile/);
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
        globalThis.fetch = priorFetch;
    }
});

// ---------------------------------------------------------------------------
// Provenance invariants carried over from the pre-store panel (9 review
// rounds), re-bound to the store-oriented structure: daemonAnswered /
// unreadFacets are pure helpers OVER the store's one facet reader, the wake is
// a STORE method (single writer), and the wake error's lifecycle lives in the
// panel. The serialization pins got simpler on purpose — one store means the
// generation bookkeeping the old panel needed is now structural.
// ---------------------------------------------------------------------------

test('READ_FACETS mirrors the store facet list, so the parity literal cannot drift', () => {
    // The store's STATUS_FACETS is the literal the backend parity test greps
    // (tests/test_gateway_parity.py ↔ ClaudexorStatusReads); READ_FACETS is
    // this module's restatement, welded here so it inherits that transitively;
    // the store's STATUS_FACETS is the one runtime reader. This pin welds the
    // two spellings together.
    assert.deepEqual(READ_FACETS, ['catalog', 'accounts', 'quota']);
    assert.deepEqual(READ_FACETS, [...STATUS_FACETS]);
});

test('a daemon that answered some reads is not called dead by the aggregate', () => {
    // The partial refusal this whole surface exists for: quota times out while
    // the catalog and the account store both land, and the backend still
    // reports the aggregate as `unreachable`. A predicate written on that
    // aggregate kept a failed wake's error standing above accounts that were
    // genuinely read, and made Refresh promise to start a daemon already
    // answering — the coarse-signal mistake, committed inside the fix for it.
    const partial = { daemon: { state: 'unreachable', engine_version: '3.3.13', runtime: {} },
                      reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
    assert.equal(daemonAnswered(partial), true, 'two landed reads read as no answer at all');
    assert.deepEqual(unreadFacets(partial), ['quota']);
    // The ACCOUNTS facet is not privileged: a catalog that landed alone still
    // proves the daemon answered.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' },
        reads: { catalog: 'ok', accounts: 'failed', quota: 'failed' } }), true);
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' },
        reads: { catalog: 'failed', accounts: 'failed', quota: 'ok' } }), true);
    // A read block that is not a facet MAP answers for nothing — an array is
    // `typeof 'object'`, and a SECOND parse of the block here once disagreed
    // with the store's reader about exactly this. The helpers go through
    // `facetReadState`, so the two can no longer split.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' }, reads: ['ok'] }), false);
    assert.equal(refreshActionKind({ daemon: { state: 'unreachable' }, reads: ['ok'] }), 'wake');
    assert.equal(refreshActionKind(partial), 'refresh',
        'the button offered to start a daemon that is answering');

    // Nothing read at all — a pre-fan-out discovery/handshake failure — still
    // means the daemon never spoke, and there the button must start it.
    const silent = { daemon: { state: 'unreachable' },
                     reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonAnswered(silent), false);
    assert.equal(refreshActionKind(silent), 'wake');
    // The aggregate is never the NEGATIVE answer, but with no facet evidence
    // either (a legacy unreachable payload) nothing is proven answered.
    assert.equal(daemonAnswered({ daemon: { state: 'unreachable' } }), false);
    assert.equal(daemonAnswered({ daemon: { state: 'running' } }), true, 'a literal running is positive evidence');
});

test('the refresh button says exactly what pressing it does', () => {
    // Label and handler were written apart once and drifted: on an
    // `unreachable` daemon the label promised a plain re-read while the click
    // called wake. One predicate now feeds both, so the button cannot promise
    // less than it does.
    assert.equal(refreshActionKind({ daemon: { state: 'running' } }), 'refresh');
    assert.equal(refreshActionLabel({ daemon: { state: 'running' } }), 'Refresh');
    for (const state of ['unreachable', 'stale', 'not_provisioned', 'foreign_daemon', '']) {
        assert.equal(refreshActionKind({ daemon: { state } }), 'wake', `state ${state}`);
        assert.match(refreshActionLabel({ daemon: { state } }), /starts the agent daemon/i,
            `the label hides the start on state ${state}`);
    }
});

test('a staged update does not claim the engine is serving when nothing answered', () => {
    // The runtime branches used to return ABOVE the facet logic, and
    // update_staged did worse than hide the gaps: "Engine X keeps running
    // until then" is a positive claim about a daemon that, in this window,
    // answered nothing — printed over a button offering to START it.
    const silentStaged = { daemon: { state: 'unreachable', engine_version: '3.3.13',
                                     runtime: { state: 'update_staged', staged_version: '3.3.14' } },
                           reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    const line = daemonStatusLine(silentStaged, {});
    assert.ok(!/keeps running/.test(line.text), 'claimed the engine is serving on a reading that saw nothing');
    assert.match(line.text, /were not read/);
    assert.match(line.text, /staged/, 'the staged update is still disclosed');

    // Everything read: the staged-update line is earned and keeps its wording.
    const servingStaged = { daemon: { state: 'running', engine_version: '3.3.13',
                                      runtime: { state: 'update_staged', staged_version: '3.3.14' } },
                            reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    assert.match(daemonStatusLine(servingStaged, {}).text, /keeps running/);

    // installing / error keep their own wording but stop hiding the facets —
    // named in the store's own subjects, never by raw facet id.
    const installing = { daemon: { state: 'unreachable', runtime: { state: 'installing', target_version: '3.3.14' } },
                         reads: { catalog: 'ok', accounts: 'failed', quota: 'failed' } };
    assert.match(daemonStatusLine(installing, {}).text,
        /[Aa]gent accounts and subscription limits were not read/);
    const broken = { daemon: { state: 'unreachable', runtime: { state: 'error', last_error: 'exit 1' } },
                     reads: { catalog: 'ok', accounts: 'failed', quota: 'ok' } };
    assert.match(daemonStatusLine(broken, {}).text, /[Aa]gent accounts were not read/);
});

test('a facet that failed without the aggregate hearing it is still reported', () => {
    // An envelope in the wrong shape is a FAILED read, not an exception, so the
    // daemon goes on reporting `running`. A status line that trusted the
    // literal printed green "Claudexor ready" directly above a row saying the
    // accounts were not checked: one screen, two contradictory claims, the
    // reassuring one on top.
    const drifted = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                      config_dir: '/x', reads: { catalog: 'failed', accounts: 'failed', quota: 'ok' } };
    const line = daemonStatusLine(drifted, {});
    assert.equal(line.tone, 'warn', 'a running daemon with two dead reads was called ready');
    assert.ok(!/Claudexor ready/.test(line.text));
    assert.match(line.text, /agents and agent accounts were not read/);

    // Everything read: the green claim is earned and must still be made.
    const whole = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                    config_dir: '/x', reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    assert.equal(daemonStatusLine(whole, {}).tone, 'ok');
    assert.match(daemonStatusLine(whole, {}).text, /Claudexor ready/);

    // A legacy payload with no read block at all keeps the old meaning.
    const legacy = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} }, config_dir: '/x' };
    assert.equal(daemonStatusLine(legacy, {}).tone, 'ok');

    // Division of labor with the banner: a gap that is merely `not_read`
    // (nobody asked) keeps the daemon's own ready line — the daemon IS proven
    // up — and the banner's muted note explains the unasked reads (pinned in
    // the serviceBannerLine tests above). Only a REAL refusal demotes the line.
    const neverAsked = { daemon: { state: 'running', engine_version: '3.3.13', runtime: {} },
                         reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonStatusLine(neverAsked, {}).tone, 'ok');
});

test('a partial refusal is not announced as a dead daemon', () => {
    // The aggregate reports `unreachable` whenever ANY read refused, so the
    // status line printed red "Daemon unreachable" directly above the account
    // rows the same poll had just delivered — the panel contradicting itself,
    // with the false half on top.
    const partial = { daemon: { state: 'unreachable', last_error: 'daemon_unreachable: quota refused', runtime: {} },
                      reads: { catalog: 'ok', accounts: 'ok', quota: 'failed' } };
    const line = daemonStatusLine(partial, {});
    assert.equal(line.tone, 'warn', 'a daemon that answered two reads was called dead');
    assert.ok(!/Daemon unreachable/.test(line.text));
    assert.match(line.text, /[Ss]ubscription limits were not read/,
        'the line must name WHICH facet is missing, not claim everything visible was read');
    assert.match(line.text, /quota refused/, 'the reason the owner needs was dropped');

    // Total silence keeps the hard verdict.
    const silent = { daemon: { state: 'unreachable', last_error: 'handshake failed', runtime: {} },
                     reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } };
    assert.equal(daemonStatusLine(silent, {}).tone, 'error');
    assert.match(daemonStatusLine(silent, {}).text, /Daemon unreachable/);
});

test('a payload with no unread facets never renders an empty list', () => {
    // The contract declares `daemon` optional, and a payload carrying only a
    // reads block used to take the partial-refusal branch with an EMPTY
    // complement — a sentence that starts with a space and names nothing.
    const noDaemon = { reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' } };
    const line = daemonStatusLine(noDaemon, {});
    assert.ok(!/^\s/.test(line.text), 'the line begins with an empty facet list');
    assert.ok(!/were not read/.test(line.text));
});

const WAKE_STILL_DOWN = {
    daemon: { state: 'stale', engine_version: '3.3.13', runtime: {} },
    config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' },
};
const WAKE_UP = {
    daemon: { state: 'running', engine_version: '3.3.14', runtime: {} },
    config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
};

test('the wake is a store method: single-flighted and serialized against the poll in both orders', async () => {
    // Two writers, two orders — now both inside the ONE store, which is what
    // makes the old generation bookkeeping unnecessary. A refresh started
    // during a wake JOINS it (no second GET runs); a wake pressed during a
    // read waits that read out before POSTing, so the wake's daemon-side read
    // causally follows the poll's commit and can never resurrect an older
    // snapshot.
    const events = [];
    let wakeGate = null;
    let readGate = null;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            if ((init.method || 'GET') === 'POST') {
                events.push('wake-posted');
                if (wakeGate) await wakeGate;
                return fakeResponse(200, WAKE_UP);
            }
            events.push('read');
            if (readGate) await readGate;
            return fakeResponse(200, WAKE_STILL_DOWN);
        },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    try {
        // Order 1: wake first — a second wake and a refresh both JOIN it.
        let releaseWake;
        wakeGate = new Promise((resolve) => { releaseWake = resolve; });
        const first = store.wake();
        const second = store.wake();
        const joined = store.refresh();
        assert.equal(events.filter((e) => e === 'wake-posted').length, 1, 'the wake is single-flighted');
        assert.equal(events.filter((e) => e === 'read').length, 0,
            'a second reader was started while the wake was in flight');
        releaseWake();
        const [a, b] = await Promise.all([first, second]);
        await joined;
        assert.equal(a.ok, true);
        assert.equal(b.ok, true);
        assert.equal(store.snapshot.daemon.engine_version, '3.3.14', 'the wake reading was committed');
        assert.equal(store.error, '', 'a committed wake reading retires staleness like every read');

        // Order 2: poll first — the wake waits it out before POSTing, and the
        // causally-later wake reading wins.
        events.length = 0;
        wakeGate = null;
        let releaseRead;
        readGate = new Promise((resolve) => { releaseRead = resolve; });
        const slow = store.refresh();
        const late = store.wake();
        assert.deepEqual(events, ['read'], 'the wake POSTed before the in-flight poll finished');
        releaseRead();
        await Promise.all([slow, late]);
        assert.deepEqual(events, ['read', 'wake-posted']);
        assert.equal(store.snapshot.daemon.engine_version, '3.3.14',
            'the wake reading (causally later) did not win');
    } finally {
        store.dispose();
    }
});

test('a wake error reaches the banner and expires only when the daemon is proven up', async () => {
    // Both edges of the same lie. The error must not outlive the daemon coming
    // up on its own (login, delegated run) — but it must also not be wiped by a
    // 200 that REPORTS the daemon still down: the reason the owner asked for
    // then vanishes within one 5s tick, before it can be read. The POST is the
    // store's; the LIFECYCLE (shown, kept, expired on `daemonAnswered`) is the
    // panel's and is what this exercises end to end.
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    globalThis.document = doc;
    globalThis.window = win;
    let statusBody = WAKE_STILL_DOWN;
    let wakeStatus = 503;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => ((init.method || 'GET') === 'POST'
            ? fakeResponse(wakeStatus, { error: 'claudexord_not_installed: no binary' })
            : fakeResponse(200, statusBody)),
        doc,
    });
    const banner = () => elements['agents-service-banner'].textContent;
    try {
        assert.equal(await initHarnessAccounts({ store }), true);
        await store.refresh();

        // The refusal reaches the owner, verbatim.
        await wakeDaemon();
        assert.match(banner(), /Could not start the agent daemon/i);
        assert.match(banner(), /claudexord_not_installed/);

        // A 200 that says the daemon is STILL down must not erase the reason.
        await store.refresh();
        assert.match(banner(), /Could not start the agent daemon/i,
            'a still-down reading erased the reason the wake failed');

        // The daemon comes up on its own: the error has stopped mattering.
        statusBody = WAKE_UP;
        await store.refresh();
        assert.ok(!/Could not start the agent daemon/i.test(banner()),
            'the stale wake error outlived the daemon that came up anyway');
        assert.match(banner(), /3\.3\.14/);

        // …and a refusal that lands while the daemon is ALREADY answering is
        // moot and never shown (the failure is real; it stopped mattering).
        await wakeDaemon();
        assert.ok(!/Could not start the agent daemon/i.test(banner()),
            'a moot refusal was printed over a daemon that is answering');
    } finally {
        await destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
    }
});

test('the refresh button routes the click through the same predicate as its label', async () => {
    // Pinning the predicate alone does not stop the handler from ignoring it —
    // replacing its body with an unconditional wake would leave every pure
    // test green. Drive the real click listener both ways.
    const { elements, doc, win } = mountSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    globalThis.document = doc;
    globalThis.window = win;
    const requests = [];
    let statusBody = WAKE_STILL_DOWN;
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push(`${init.method || 'GET'} ${url}`);
            if ((init.method || 'GET') === 'POST') return fakeResponse(200, WAKE_UP);
            return fakeResponse(200, statusBody);
        },
        doc,
    });
    try {
        assert.equal(await initHarnessAccounts({ store }), true);
        await store.refresh();
        const click = elements['btn-harness-refresh'].listeners
            .filter(([type]) => type === 'click').map(([, fn]) => fn).at(-1);
        assert.ok(click, 'the refresh button lost its listener');

        // Daemon asleep -> the press must START it (and the label says so).
        assert.match(elements['btn-harness-refresh'].textContent, /starts the agent daemon/i);
        requests.length = 0;
        await click();
        assert.ok(requests.some((r) => r.startsWith('POST /api/claudexor/wake')),
            'a sleeping daemon was only re-read, so the button could not help');

        // Daemon live (the wake committed an answering reading) -> the press
        // stays a plain re-read, and the label agrees.
        statusBody = WAKE_UP;
        requests.length = 0;
        await click();
        assert.ok(requests.every((r) => !r.includes('/api/claudexor/wake')),
            'a live daemon was provisioned by a button that says Refresh');
        assert.ok(requests.some((r) => r.startsWith('GET ')), 'the live press did not re-read');
        assert.equal(elements['btn-harness-refresh'].textContent, 'Refresh');
    } finally {
        await destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
    }
});

test('an unknown auth row refreshes status on the real click, never starts login', async () => {
    const { elements, doc, win, loginButton } = mountInteractiveAccountsSection();
    const priorDoc = globalThis.document;
    const priorWin = globalThis.window;
    const priorFetch = globalThis.fetch;
    globalThis.document = doc;
    globalThis.window = win;
    const requests = [];
    const snapshot = {
        daemon: { state: 'running', engine_version: '3.3.13', runtime: { state: 'ready' } },
        harnesses: [{ id: 'claude', display_name: 'Claude Code' }],
        profiles: {
            profiles: [{
                profile: { harness_id: 'claude', profile_id: 'work', display_name: 'Work', enabled: true },
                status: { availability: 'unknown', verification: 'not_run' },
                identity: {},
            }],
            harnessAccounts: [],
        },
        quota: [],
    };
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push(`${init.method || 'GET'} ${url}`);
            return fakeResponse(200, snapshot);
        },
        doc,
    });
    const originalRefresh = store.refresh.bind(store);
    let refreshCalls = 0;
    let refreshPromise;
    store.refresh = (...args) => {
        refreshCalls += 1;
        refreshPromise = originalRefresh(...args);
        return refreshPromise;
    };
    try {
        assert.equal(await initHarnessAccounts({ store }), true);
        await refreshPromise;
        refreshCalls = 0;
        requests.length = 0;
        const click = loginButton.listeners
            .filter(([type]) => type === 'click').map(([, fn]) => fn).at(-1);
        assert.ok(click, 'the account row lost its login listener');
        click();
        await refreshPromise;
        assert.equal(refreshCalls, 1, 'unknown status did not use the shared Refresh path');
        assert.ok(requests.some((request) => request.startsWith('GET ')),
            'Refresh did not re-read status');
        assert.ok(requests.every((request) => !request.startsWith('POST /api/claudexor/login')),
            'unknown status incorrectly started OAuth login');
        assert.match(elements['harness-accounts-groups'].innerHTML, /Login status unknown/);
    } finally {
        destroyHarnessAccounts();
        store.dispose();
        globalThis.document = priorDoc;
        globalThis.window = priorWin;
        globalThis.fetch = priorFetch;
    }
});

test('a login seen online is not un-seen by the re-check running out', () => {
    // Re-bound from the pre-store panel (which pinned its own settleVerdict) to
    // the shared controller the PR extracted: the bounded re-check can close a
    // tick before the account row lands, while a reading that arrived meanwhile
    // (the held status poll, an owner wake) already shows the login. Seeing the
    // login is MONOTONE evidence — the newest committed snapshot is judged with
    // the same predicate before "unconfirmed" may be said. Structural, because
    // settleVerdict is controller-private.
    const src = readFileSync(new URL('../modules/harness_login_cards.js', import.meta.url), 'utf8');
    const body = String(src.split('async function settleVerdict(')[1] || '');
    assert.match(body, /const confirmed = check\.confirmed\s*\n?\s*\|\|\s*accountLoginConfirmed\(store\?\.snapshot/,
        'a positive confirmation no longer wins on its own');
    assert.match(body, /active\.verdict = confirmed/,
        'the monotone predicate does not decide the verdict');
});

test('the family markup mounts a per-family login host under its header (3=A)', () => {
    const html = harnessFamilyMarkup(
        { harness: 'codex', label: 'Codex CLI', rows: [], status: { tone: 'muted', label: 'x' } },
        {}, { accountsRead: 1 },
    );
    const headIdx = html.indexOf('agent-family-head');
    const loginIdx = html.indexOf('data-family-login="codex"');
    const rowsIdx = html.indexOf('agent-family-rows');
    assert.ok(headIdx > -1 && loginIdx > -1 && rowsIdx > -1);
    // The login card appears where the owner clicked: directly under the
    // family header, above the account rows — not after every family.
    assert.ok(headIdx < loginIdx && loginIdx < rowsIdx);
});
