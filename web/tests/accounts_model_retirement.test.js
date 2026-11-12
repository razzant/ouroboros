// #244: the pre-unified account model (native pseudo-rows in `harnessAccounts`, no
// `accountPools`) is retired from Accounts, onboarding and quota attribution. An engine
// that still serves it is named and explained, never shown as "no accounts"; a current
// engine keeps its normal surface; a quota window answers only for its exact subject.
import assert from 'node:assert/strict';
import test from 'node:test';

import * as store from '../modules/claudexor_status_store.js';
import * as accounts from '../modules/harness_accounts.js';
import * as status from '../modules/subagent_status_primitives.js';
import * as login from '../modules/harness_login_cards.js';
import * as agentsStep from '../modules/onboarding_agents_step.js';
import { indexProfilesByHarness } from '../modules/route_editor_primitives.js';
import { availableSubagentRowMarkup } from '../modules/subagents_settings.js';
import { sessionRow } from './helpers/subagents_editor.js';

const NOW = Date.parse('2026-10-27T12:00:00Z');
const inHours = (hours) => new Date(NOW + hours * 3600_000).toISOString();
const catalog = [
    { id: 'claude', display_name: 'Claude Code', status: 'ok', enabled: true, models: [{ id: 'claude-fable-5-1' }] },
    { id: 'codex', display_name: 'Codex', status: 'ok', enabled: true, models: [] },
    { id: 'cursor', display_name: 'Cursor', status: 'ok', enabled: true, models: [] },
];
const named = (harness, id, verification = 'passed') => ({
    profile: { harness_id: harness, profile_id: id, display_name: '', credential_kind: 'config_dir_login', enabled: true },
    status: { harness_id: harness, profile_id: id, availability: 'available', verification, verification_source: 'vendor' },
    identity: null,
});
const native = (harness) => ({ harness_id: harness, native_credentials_enabled: true, native_login_detected: true,
    identity: { email: 'native@example.invalid' }, next_up: { kind: 'native', route: 'local_session' } });
const okFacets = Object.fromEntries(['catalog', 'accounts', 'quota'].map((name) => [name,
    { observed_at: '2026-10-27T11:59:00Z', stale: false, error: null }]));
const windowFor = (harness, subject, used, resetHours) => ({
    subject: { harness, subject_id: subject }, freshness: 'fresh', observed_at: inHours(-0.1),
    constraints: [{ id: 'five_hour', label: '5h', used_ratio: used, resets_at: inHours(resetHours), freshness: 'fresh' }],
    availability: { state: used >= 1 ? 'exhausted' : 'available', resets_at: used >= 1 ? inHours(resetHours) : null },
});

function unified({ profiles = [], pools = [], quota = [], ...rest } = {}) {
    return { daemon: { state: 'running', engine_version: '4.7.0', runtime: { state: 'ready', target_version: '4.7.0' } },
        harnesses: catalog, profiles: { profiles, harnessAccounts: [], accountPools: pools }, quota, quota_absences: [],
        reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' }, facets: okFacets, unified_accounts: true, ...rest };
}

function refused(error, runtime = { state: 'update_staged', target_version: '4.7.0', staged_version: '4.7.0', source: 'release' },
    engine = '3.5.2') {
    return { daemon: { state: 'running', engine_version: engine, runtime }, harnesses: catalog, profiles: {}, quota: [],
        quota_absences: [], reads: { catalog: 'ok', accounts: 'failed', quota: 'ok' },
        facets: { ...okFacets, accounts: { observed_at: null, stale: false, error } }, unified_accounts: false };
}

async function storeOver(payload) {
    const instance = store.createClaudexorStatusStore({
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => structuredClone(payload) }),
        doc: () => null, storage: () => null,
    });
    await instance.refresh();
    return instance;
}

test('accountRows never synthesizes a native pseudo-row, whatever the payload says about the model', () => {
    for (const flag of [false, undefined]) {
        const payload = { daemon: { state: 'running' }, harnesses: catalog, quota: [],
            profiles: { profiles: [named('claude', 'work-research')], harnessAccounts: [native('claude'), native('codex')] },
            ...(flag === undefined ? {} : { unified_accounts: flag }) };
        const rows = store.accountRows(payload);
        assert.deepEqual(rows.map((row) => [row.harness, row.profile_id, row.kind]), [['claude', 'work-research', 'profile']]);
        assert.equal(store.accountLoginConfirmed(payload, 'claude', ''), false, 'no empty-id row can confirm a login');
    }
});

test('nextUpAccount reads only the pool verdict', () => {
    const legacyOnly = { profiles: { profiles: [], harnessAccounts: [native('claude')] } };
    assert.equal(store.nextUpAccount(legacyOnly, 'claude'), null);
    const pool = { kind: 'profile', profileId: 'claude-default' };
    const hybrid = { profiles: { profiles: [], harnessAccounts: [native('claude')], accountPools: [{ harness_id: 'claude', next_up: pool }] } };
    assert.deepEqual(store.nextUpAccount(hybrid, 'claude'), pool);
    // A `native` kind reaching the pool union is an unknown future kind, rendered fail-safe.
    const nativeKind = { profiles: { accountPools: [{ harness_id: 'claude', next_up: { kind: 'native', route: 'api_key' } }] } };
    assert.equal(accounts.nextUpBadge(nativeKind, 'claude'), 'Next up: unknown');
});

test('a pre-unified account model is its own state, distinct from failed, malformed and empty reads', async () => {
    assert.equal(store.accountModelUnsupported?.(refused('account_model_unsupported')), true);
    for (const payload of [refused('daemon_unreachable'), refused('malformed_response'), unified(), {}]) {
        assert.equal(store.accountModelUnsupported?.(payload), false);
    }
    const legacy = await storeOver(refused('account_model_unsupported'));
    assert.equal(legacy.accountsUnsupported, true);
    assert.equal(legacy.accountsKnown, false, 'an unsupported roster is not an authoritative empty one');
    const note = legacy.unavailableNote('accounts');
    assert.equal(note.tone, 'warn');
    assert.equal(note.text, 'Claudexor 3.5.2 keeps accounts in an older format that this version of Ouroboros '
        + 'no longer reads, so they can’t be listed or managed here. They are unchanged. '
        + 'Claudexor 4.7.0 is installed and takes over when the agent daemon next restarts.');
    const failed = await storeOver(refused('daemon_unreachable'));
    assert.equal(failed.accountsUnsupported, false);
    assert.match(failed.unavailableNote('accounts').text, /could not be read \(daemon_unreachable\)/);
    const empty = await storeOver(unified({ pools: [{ harness_id: 'claude', next_up: { kind: 'none', reason: 'no account' } }] }));
    assert.equal(empty.accountsUnsupported, false);
    assert.equal(empty.unavailableNote('accounts'), null);
    [legacy, failed, empty].forEach((instance) => instance.dispose());
});

test('the update pointer names only what the runtime projection knows', async () => {
    const said = async (runtime, engine = '3.5.2') => {
        const instance = await storeOver(refused('account_model_unsupported', runtime, engine));
        const text = instance.unavailableNote('accounts').text;
        instance.dispose();
        return text;
    };
    assert.match(await said({ state: 'update_available', target_version: '4.7.0', source: 'release' }),
        / They are unchanged\. Claudexor 4\.7\.0 takes over when the agent daemon next restarts\.$/);
    assert.match(await said({ state: 'ready', version: '3.5.2', target_version: '4.7.0', source: 'override' }),
        /This Claudexor comes from OUROBOROS_CLAUDEXOR_BIN; point it at Claudexor 3\.6 or later\.$/);
    assert.match(await said({ state: 'ready', source: 'external' }), /Update that Claudexor to 3\.6 or later\.$/);
    assert.match(await said({}, ''), /^This Claudexor keeps accounts in an older format .* Update Claudexor to 3\.6 or later\.$/);
});

test('Accounts on an unsupported model lists nothing and offers no Connect or Add; the banner explains', async () => {
    const instance = await storeOver(refused('account_model_unsupported'));
    const snapshot = instance.snapshot;
    const options = { accountsRead: instance.facet('accounts'), catalogKnown: true, accountsUnsupported: instance.accountsUnsupported };
    const groups = accounts.accountGroups(snapshot, options);
    assert.deepEqual(groups.map((group) => [group.harness, group.rows.length, group.status.label]), [
        ['claude', 0, 'Not listed — needs a newer Claudexor'], ['codex', 0, 'Not listed — needs a newer Claudexor'],
        ['cursor', 0, 'Not listed — needs a newer Claudexor']]);
    const html = groups.map((group) => accounts.harnessFamilyMarkup(group, snapshot,
        { accountsRead: options.accountsRead, quotaRead: 'ok', store: instance, accountsUnsupported: true })).join('');
    assert.doesNotMatch(html, /data-family-add|data-harness-login|data-next-up/);
    const line = accounts.serviceBannerLine(instance, {});
    assert.equal(line.tone, 'warn');
    assert.equal(line.text, instance.unavailableNote('accounts').text);
    instance.dispose();

    // A failed read on a current engine keeps the established Connect affordance.
    const failed = await storeOver(refused('daemon_unreachable', { state: 'ready', target_version: '4.7.0' }, '4.7.0'));
    const failedGroups = accounts.accountGroups(failed.snapshot, { accountsRead: 'failed', catalogKnown: true,
        accountsUnsupported: failed.accountsUnsupported });
    assert.equal(failedGroups[0].status.label, 'Not checked — the daemon did not answer this read');
    assert.match(accounts.harnessFamilyMarkup(failedGroups[0], failed.snapshot, { accountsRead: 'failed', quotaRead: 'ok' }),
        /data-family-add>Connect</);
    failed.dispose();
});

test('the unsupported banner still names a second facet that failed beside it', async () => {
    const payload = refused('account_model_unsupported');
    payload.reads.quota = 'failed';
    payload.facets.quota = { observed_at: null, stale: false, error: 'daemon_unreachable' };
    const instance = await storeOver(payload);
    const line = accounts.serviceBannerLine(instance, {});
    assert.equal(line.tone, 'warn');
    assert.ok(line.text.startsWith('Claudexor 3.5.2 keeps accounts in an older format'), line.text);
    assert.match(line.text, /subscription limits could not be read \(daemon_unreachable\)/);
    assert.doesNotMatch(line.text, /agent accounts and subscription limits/);
    instance.dispose();
});

test('a current engine keeps its normal Accounts surface', async () => {
    const instance = await storeOver(unified({
        profiles: [named('claude', 'claude-default'), named('codex', 'codex-default')],
        pools: [{ harness_id: 'claude', next_up: { kind: 'profile', profileId: 'claude-default' } }],
    }));
    const groups = accounts.accountGroups(instance.snapshot, { accountsRead: 'ok', catalogKnown: true });
    assert.deepEqual(groups.map((group) => [group.harness, group.rows.map((row) => row.profile_id), group.status.label]), [
        ['claude', ['claude-default'], 'Signed in'], ['codex', ['codex-default'], 'Signed in'], ['cursor', [], 'No account connected']]);
    const html = accounts.harnessFamilyMarkup(groups[0], instance.snapshot, { accountsRead: 'ok', quotaRead: 'ok', store: instance });
    assert.match(html, /data-family-add>Add account</);
    assert.match(html, /data-harness-remove/);
    assert.match(html, /Next up: claude-default/);
    assert.equal(accounts.serviceBannerLine(instance, {}).text, 'Claudexor ready');
    instance.dispose();
});

test('a named default row shows only its own quota window, never an unprofiled one', () => {
    const row = store.accountRows(unified({ profiles: [named('codex', 'codex-default')] }))[0];
    const unprofiledOnly = unified({ profiles: [named('codex', 'codex-default')], quota: [windowFor('codex', null, 1, 3)] });
    const borrowed = accounts.accountRowFacts(row, unprofiledOnly, { nowMs: NOW });
    assert.equal(borrowed.quota.exhausted, false);
    assert.equal(borrowed.quota.label, 'Usage unavailable');
    const own = unified({ profiles: [named('codex', 'codex-default')], quota: [windowFor('codex', null, 0.1, 1),
        windowFor('codex', 'codex-default', 1, 3)] });
    const facts = accounts.accountRowFacts(row, own, { nowMs: NOW });
    assert.equal(facts.quota.exhausted, true);
    assert.equal(facts.quota.label, 'Limit reached · resets in 3h');
});

test('a pinned session never borrows an unprofiled window for `<harness>-default`', () => {
    const scope = { model: 'claude-fable-5-1', requestKind: 'text', basis: 'quota_block' };
    const row = { route: { kind: 'agent_session', target_id: 'claude=claude-fable-5-1', credential_profile_id: 'claude-default' } };
    const at = (quota) => ({ snapshot: unified({ profiles: [named('claude', 'claude-default')], quota, availability_scope: scope }),
        catalogKnown: true, accountsKnown: true, quotaKnown: true });
    const unprofiled = status.sessionRouteVerdict(row, at([windowFor('claude', null, 1, 3)]), NOW);
    assert.equal(unprofiled.label, 'Not checked');
    assert.match(unprofiled.text, /quota availability not proven/);
    assert.equal(status.sessionRouteVerdict(row, at([windowFor('claude', 'claude-default', 1, 3)]), NOW).label, 'Limit reached');
});

test('an empty-family login confirms the exact engine-resolved profile, never a sibling or a legacy row', async () => {
    assert.equal(login.resolvedJobProfileId({ job: { profileId: 'claude-default-2' } }), 'claude-default-2');
    assert.equal(login.resolvedJobProfileId({ job: { profileId: null } }), '');
    const quick = { attempts: 2, delayMs: 0, sleepImpl: async () => {} };
    const siblingOnly = unified({ profiles: [named('claude', 'claude-default'), named('claude', 'claude-default-2', 'not_run')] });
    const resolved = unified({ profiles: [named('claude', 'claude-default'), named('claude', 'claude-default-2')] });
    assert.equal((await login.confirmLoginLive('claude', 'claude-default-2', { ...quick, readStatus: async () => siblingOnly })).confirmed, false);
    assert.equal((await login.confirmLoginLive('claude', 'claude-default-2', { ...quick, readStatus: async () => resolved })).confirmed, true);
    // A job that resolved no profile (a pre-unified engine's null) is never confirmed by a native row.
    const legacy = { daemon: { state: 'running' }, harnesses: catalog, quota: [],
        profiles: { profiles: [], harnessAccounts: [native('claude')] } };
    assert.equal((await login.confirmLoginLive('claude', '', { ...quick, readStatus: async () => legacy })).confirmed, false);
});

test('the onboarding Agents step on an unsupported model explains and offers no Connect', () => {
    const snapshot = refused('account_model_unsupported');
    const html = agentsStep.familyListHtml(snapshot, { accountsKnown: false, catalogKnown: true, accountsUnsupported: true });
    assert.doesNotMatch(html, /data-agent-connect/);
    assert.equal((html.match(/Not listed — needs a newer Claudexor/g) || []).length, agentsStep.AGENT_FAMILIES.length);
    assert.match(agentsStep.agentsOutcomeText([], { accountsKnown: false, accountsUnsupported: true }), /newer Claudexor/);
    // A failed read on a current engine keeps Connect and its own words.
    const failedHtml = agentsStep.familyListHtml(refused('daemon_unreachable'), { accountsKnown: false, catalogKnown: true });
    assert.match(failedHtml, /data-agent-connect="codex"/);
    assert.match(agentsStep.agentsOutcomeText([], { accountsKnown: false }), /Accounts could not be checked/);
});

test('an account pin never offers the held roster of a refused account model, and keeps the saved pin', async () => {
    // Beside any failed accounts read the producer serves its last unified roster
    // again, stale (`_with_facet_memory`). On a refused pre-unified model those
    // rows are not this engine's accounts, so the pin select must not offer them.
    const held = [named('codex', 'codex-default'), named('codex', 'koshak')];
    const servedAgain = (error, runtime, engine) => ({ ...refused(error, runtime, engine),
        profiles: { profiles: held, harnessAccounts: [], accountPools: [] },
        facets: { ...okFacets, accounts: { observed_at: '2026-10-27T11:59:00Z', stale: true, error } } });
    let answer = null;
    const instance = store.createClaudexorStatusStore({
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => structuredClone(answer) }),
        doc: () => null, storage: () => null,
    });
    // The Subagents editor's own derivation from the store (`adoptStatus`).
    const accountChoices = () => {
        const html = availableSubagentRowMarkup(sessionRow(), { snapshot: instance.snapshot,
            catalogKnown: instance.facet('catalog') === 'ok', accountsKnown: instance.facet('accounts') === 'ok',
            quotaKnown: instance.facet('quota') === 'ok' });
        const select = html.match(/<select[^>]*data-subagent-field="account"[^>]*>([\s\S]*?)<\/select>/)[1];
        return [...select.matchAll(/<option value="([^"]*)"( selected)?>([^<]*)<\/option>/g)]
            .map(([, value, selected, label]) => `${value}${selected ? '*' : ''}: ${label}`);
    };
    const seen = [];
    let refusedSnapshot = null;
    for (answer of [unified({ profiles: held }), servedAgain('account_model_unsupported'),
        servedAgain('daemon_unreachable', { state: 'ready', target_version: '4.7.0' }, '4.7.0'),
        unified({ profiles: [named('codex', 'codex-default')] })]) {
        await instance.refresh();
        seen.push(accountChoices());
        if (store.accountModelUnsupported(instance.snapshot)) refusedSnapshot = instance.snapshot;
    }
    assert.deepEqual(seen, [
        [': Automatic rotation', 'codex-default: codex-default (pinned)', 'koshak*: koshak (pinned)'],
        [': Automatic rotation', 'koshak*: koshak (not checked)'],
        // A failed read on a current engine still offers its last known rows.
        [': Automatic rotation', 'codex-default: codex-default (pinned)', 'koshak*: koshak (pinned)'],
        [': Automatic rotation', 'codex-default: codex-default (pinned)', 'koshak*: koshak (not in discovery)'],
    ]);
    assert.deepEqual(indexProfilesByHarness(refusedSnapshot), {});
    assert.equal(store.accountRows(refusedSnapshot).length, 2, 'the held source itself is untouched');
    instance.dispose();
});
