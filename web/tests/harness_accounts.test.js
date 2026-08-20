import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
    accountLoginConfirmed,
    accountRows,
} from '../modules/claudexor_status_store.js';
import {
    daemonStatusLine,
    normalizeProfileName,
    promptProfileName,
    quotaSummary,
    runtimeActionLabel,
    verificationBadge,
} from '../modules/harness_accounts.js';
// The login machinery moved to the shared controller module (phase 2) so the
// onboarding wizard can mount the same flow; the assertions below are
// unchanged, which is what makes the extraction behavior-preserving.
import {
    ATTACH_FALLBACK_MS,
    attachFallbackDue,
    confirmLoginLive,
    deviceCodeDisclosure,
    failureText,
    jobStateSummary,
    loginCardFace,
    loginCardHtml,
    loginInputSupport,
    loginStatusLine,
    loginVerdict,
    submitLoginInput,
} from '../modules/harness_login_cards.js';
import { fakeResponse } from './harness_accounts_helpers.js';
// Re-exported so this file keeps the exact module surface it had before the
// split: `fakeResponse` was declared here and is now owned by the helper
// sibling, and the v7 migration ledger binds that facade.
export { fakeResponse };

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
    const preparing = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: true,
        error: '', verdict: null, confirming: false,
    });
    assert.ok(preparing.includes('Installing or checking Claudexor…'));
    assert.ok(!preparing.includes('data-login-retry'));

    const failed = loginCardHtml({
        harness: 'claude', profile: '', envelope: null, preparingRuntime: false,
        error: 'checksum mismatch', verdict: null, confirming: false,
    });
    assert.ok(failed.includes('checksum mismatch'));
    assert.ok(failed.includes('data-login-retry'));
    assert.ok(!failed.includes('Installing or checking Claudexor…'));
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
        verification: 'passed', verification_source: 'vendor', last_verified_at: '2026-08-03T10:00:00Z',
    } });
    assert.equal(vendor.tone, 'ok');
    assert.equal(vendor.label, 'Verified live');
    // The raw ISO instant left the badge entirely (owner: a row must never lead
    // with a timestamp); accountMetaLine humanizes it on line 2 instead.
    assert.doesNotMatch(vendor.label, /2026-/);

    const local = verificationBadge({ status: { verification: 'passed', verification_source: 'local_store' } });
    assert.equal(local.tone, 'muted');
    // Narrower claim, narrower words — shorter, but "not verified live" stays.
    assert.equal(local.label, 'Signed in — not verified live');

    assert.equal(verificationBadge({ status: {} }).label, 'Not signed in');
    assert.equal(verificationBadge({ status: { verification: 'failed', verification_source: 'vendor' } }).tone, 'error');
});

// `freshness` is a REQUIRED member of the daemon's quota snapshot
// (@claudexor/schema quota.ts, `z.enum(['fresh','stale','unknown'])`), so every
// fixture here carries it exactly as the wire does.
test('an exhausted window is shown with its reset time, never hidden', () => {
    const snapshots = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
        constraints: [{ used_ratio: 1.0, resets_at: '2026-08-04T00:00:00Z' }],
    }];
    // Owner ask: the limit text compact and understandable. The RESET is
    // humanized against a fixed now, and the raw instant stays on `resetsAt`.
    const now = Date.parse('2026-08-03T22:00:00Z');
    const summary = quotaSummary(snapshots, 'codex', 'koshak', { nowMs: now });
    assert.equal(summary.exhausted, true);
    assert.equal(summary.resetsAt, '2026-08-04T00:00:00Z');
    assert.equal(summary.label, 'Limit reached · resets in 2h');
    assert.doesNotMatch(summary.label, /2026-/);

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
        constraints: [{ used_ratio: 1.0, resets_at: '2026-08-04T00:00:00Z' }],
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
            { used_ratio: 0.20, cooldown_until: '2026-08-04T00:00:00Z' },
            { used_ratio: 0.80 },
        ],
    }];
    const summary = quotaSummary(cooling, 'codex', 'koshak');
    assert.equal(summary.exhausted, true);
    assert.equal(summary.resetsAt, '2026-08-04T00:00:00Z');

    // ...and vanished entirely when the cooling constraint reported no ratio at all,
    // because a non-finite used_ratio was skipped before it could be read.
    const ratioless = [{
        subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh',
        constraints: [{ cooldown_until: '2026-08-04T00:00:00Z' }],
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
          constraints: [{ used_ratio: 1.0, resets_at: '2026-08-04T00:00:00Z' }] },
    ];
    const defaultRow = quotaSummary(snapshots, 'codex', '');
    assert.equal(defaultRow.exhausted, false);
    assert.equal(defaultRow.label, '5% used');
    const namedRow = quotaSummary(snapshots, 'codex', 'koshak');
    assert.equal(namedRow.exhausted, true);
    assert.equal(namedRow.resetsAt, '2026-08-04T00:00:00Z');
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
              used_ratio: 1.0, resets_at: '2026-08-08T00:00:00Z' },
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
            applies_to_models: ['claude-fable-5'], cooldown_until: '2026-08-08T00:00:00Z', used_ratio: null }],
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
        constraints: [{ applies_to_models: null, used_ratio: 1.0, resets_at: '2026-08-08T00:00:00Z' }],
    }], 'claude', 'abstractdl');
    assert.equal(global.exhausted, true);
    assert.ok(global.label.startsWith('Limit reached'));

    // Without a label, the note falls back to the constraint id, then models.
    assert.equal(quotaSummary([{
        subject, freshness: 'fresh',
        constraints: [{ id: 'fable_5h', applies_to_models: ['claude-fable-5'], used_ratio: 1.0 }],
    }], 'claude', 'abstractdl').label, 'fable_5h spent');
});

// ---------------------------------------------------------------------------
// Add account: pywebview's WKWebView implements no window.prompt (it answers
// null silently), so the flow runs on the in-house input dialog.
// ---------------------------------------------------------------------------

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
