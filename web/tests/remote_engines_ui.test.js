// Named remote engines (OUROBOROS_ENGINE_CONNECTIONS) in the shared route editor, the Models and Agents editors, the
// composer's Model capsule and Settings › Accounts and keys: a remote engine is its own source kind ("On <engine> ·
// <source>"), a stored `claudexor@<engine>::source=model` or a session row's `route.engine` round-trips through every
// editor step and is never collapsed into the owned engine or read as an API key, a pin is always that engine's own
// account, a saved choice an unread catalog cannot confirm stays selectable as "not checked", and an engine's page
// writes one entry with that key's token, keeping unknown, not connected, needs sign-in, unavailable and owed apart.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
    changeRouteChoice, composeModelSource, decodeRouteChoice, encodeRouteChoice, parseModelSource, routeChoiceGroups,
    routeKeyProblem, routeModelFields, routeTargetFromModel, serializeRouteSpec, sourceIdentityLabel,
} from '../modules/route_editor_primitives.js';
import { createModelRolesEditor, modelSourceGroups, roleBody, routeWords, sourceChoice, sourceFromChoice } from '../modules/model_roles.js';
import { buildModelIndex, describeRoute, sourceUnlisted } from '../modules/chat_model_routes.js';
import { sourceMark } from '../modules/model_source_mark.js';
import { connectionUsage } from '../modules/settings_connections.js';
import { forgetSettingRows, readSettings } from '../modules/settings_controls.js';
import {
    ENGINE_DETAILS, engineConnectionRows, engineHeads, engineNameFor, engineRowFacts, enginePath, engineTitle,
    parseEnginePath, removedOwedLine,
} from '../modules/settings_engine_connections.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { HAND_WRITTEN } from '../modules/settings_rows.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';
import { Stub } from './helpers/settings_dom.js';
import '../modules/settings_accounts.js';  // registers the Accounts family, the engine pages included

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));

test('a remote engine is its own source and its routes round-trip', () => {
    assert.deepEqual(parseModelSource('claudexor@vps::codex=gpt-6'), { source: 'remote:vps:codex', model: 'gpt-6' });
    assert.deepEqual(parseModelSource('claudexor@default::codex=gpt-6'), parseModelSource('claudexor::codex=gpt-6'));
    assert.equal(composeModelSource('remote:vps:codex', 'gpt-6'), 'claudexor@vps::codex=gpt-6');
    const route = { kind: 'api_model', target_id: 'claudexor@vps::codex=gpt-6' };
    const fields = routeModelFields(route);
    assert.equal(fields.remote, true);
    assert.equal(fields.engine, 'vps');
    assert.equal(routeTargetFromModel(route, 'gpt-6-mini'), 'claudexor@vps::codex=gpt-6-mini');
    assert.equal(encodeRouteChoice({ route }), 'remote:vps:codex');
    assert.equal(sourceIdentityLabel(route), 'On vps · codex · model');
    assert.deepEqual(changeRouteChoice({ kind: 'api_model', target_id: 'claudexor::codex=m' }, 'remote:vps:codex'),
        { kind: 'api_model', target_id: 'claudexor@vps::codex=' });
    assert.deepEqual(changeRouteChoice(route, 'subscription:codex'), { kind: 'api_model', target_id: 'claudexor::codex=' });
    assert.equal(sourceFromChoice(sourceChoice('remote:vps:codex')), 'remote:vps:codex');
});

test('a remote session row keeps its engine and workspace through every editor step', () => {
    const row = { route: { kind: 'agent_session', target_id: 'codex=gpt-6', credential_profile_id: 'remote-main',
        engine: 'vps', workspace: '/home/ouroboros/project' } };
    assert.equal(encodeRouteChoice(row), 'remote-session:vps:codex');
    assert.deepEqual(decodeRouteChoice('remote-session:vps:codex'), { kind: 'agent_session', harness: 'codex', engine: 'vps' });
    assert.deepEqual(serializeRouteSpec(row.route), { kind: 'agent_session', target_id: 'codex=gpt-6',
        credential_profile_id: 'remote-main', engine: 'vps', workspace: '/home/ouroboros/project' });
    assert.deepEqual(serializeRouteSpec({ kind: 'agent_session', target_id: 'codex=gpt-6' }),
        { kind: 'agent_session', target_id: 'codex=gpt-6' });  // a local row's bytes stay as they were
    // Another engine is another source: the pin and the folder, both that engine's, do not travel.
    assert.deepEqual(changeRouteChoice(row.route, 'remote-session:lab:codex'), { kind: 'agent_session', target_id: 'codex', engine: 'lab' });
    assert.deepEqual(changeRouteChoice(row.route, 'session:codex'), { kind: 'agent_session', target_id: 'codex' });
    assert.equal(sourceIdentityLabel(row.route), 'On vps · codex · agent');
});

test('the Remote engines group lists connected catalogs and keeps an unconfirmed saved choice', () => {
    const groups = routeChoiceGroups({ remoteEngines: [
        { name: 'vps', label: 'Build box', state: 'connected', sources: [{ id: 'codex', label: 'Codex' }], harnesses: ['codex'] },
        { name: 'lab', label: 'Lab box', state: 'offline' },
    ] });
    const remote = groups.find((group) => group.label === 'Remote engines');
    assert.deepEqual(remote.options.map((option) => option.value), ['remote:vps:codex', 'remote-session:vps:codex', '']);
    assert.match(remote.options[0].label, /^On Build box · Codex \(model\)$/);
    assert.match(remote.options[2].label, /Lab box · not connected/);
    const saved = modelSourceGroups({ current: 'remote:vps:codex' });
    assert.deepEqual(saved[0].options, [{ value: 'remote:vps:codex', label: 'On vps · Codex (model) · not checked' }]);
    // Not connected, the saved choice reads as the connected one does: the engine's saved name, the source's word.
    const offline = routeChoiceGroups({ currentChoice: 'remote:vps:codex',
        remoteEngines: [{ name: 'vps', label: 'Build box', state: 'offline' }] })
        .find((group) => group.label === 'Remote engines');
    assert.deepEqual(offline.options.map((option) => option.label),
        ['On Build box · not connected — Check it in Accounts', 'On Build box · Codex (model) · not checked']);
    assert.equal(sourceIdentityLabel({ kind: 'api_model', target_id: 'claudexor@vps::codex=gpt-6' },
        { remoteEngines: [{ name: 'vps', label: 'Build box' }] }), 'On Build box · codex · model');
    assert.equal(routeChoiceGroups({}).some((group) => group.label === 'Remote engines'), false);
});

test('Accounts facts keep unknown, offline, sign-in and failure apart and never invent success', () => {
    assert.deepEqual(engineConnectionRows('[{"name":"vps","ssh_host":"ouroboros@vds"}]'), [{ name: 'vps', ssh_host: 'ouroboros@vds' }]);
    assert.deepEqual(engineConnectionRows('not json'), []);
    assert.equal(engineRowFacts({ draft: true }).label, 'Not saved');
    assert.equal(engineRowFacts({}).label, 'Not checked');
    assert.equal(engineRowFacts({ observation: { state: 'offline' } }).label, 'Not connected');
    const ok = engineRowFacts({ check: { ok: true, engine_version: '5.1.0', store: 'store-remote-123456' } });
    assert.deepEqual([ok.label, ok.tone], ['Connected', 'ok']);
    assert.match(engineRowFacts({ check: { ok: true, engine_version: '5.1.0', user_mismatch: true } }).meta, /another user/);
    assert.equal(engineRowFacts({ check: { ok: false, code: 'remote_ssh_needs_interaction' } }).label, 'Needs sign-in');
    const failed = engineRowFacts({ check: { ok: false, code: 'remote_engine_not_running', message: 'No running engine' } });
    assert.deepEqual([failed.label, failed.tone], ['Unavailable', 'error']);
    assert.match(engineRowFacts({ enabled: false }).meta, /^Switched off/);
    const owed = engineRowFacts({ observation: { state: 'offline' }, owed: { runs: 1, starts: 1 } });
    assert.equal(owed.tone, 'warn');
    assert.equal(owed.owed, 'Emergency Stop: 2 runs are not yet confirmed stopped on this engine; the stop is delivered when it answers this computer, and the work may continue there until then');
    assert.equal(engineRowFacts({ owed: { runs: 0, starts: 0 } }).owed, '');
    assert.match(engineRowFacts({ observation: { state: 'offline' }, owed: null }).owed, /could not be read$/);
    const edited = engineRowFacts({ dirty: true, check: { ok: true, engine_version: '5.1.0' } });
    assert.deepEqual([edited.label, edited.meta], ['Unsaved changes', 'Saved engine: connected · Save, then Check reads the new fields']);
    assert.equal(engineRowFacts({ observation: { state: 'unread', message: 'HTTP 500' } }).label, 'Status unknown');
    const unstopped = engineRowFacts({ observation: { state: 'stop_unconfirmed' } });
    assert.deepEqual([unstopped.label, unstopped.tone], ['Stop unconfirmed', 'warn'], 'an unconfirmed stop is never "Not connected"');
    assert.doesNotMatch(engineRowFacts({ observation: { state: 'connected' } }).meta, /(^ · )|( · $)/);
});

test('a stop still owed on a removed connection stays in view and says how it is delivered', () => {
    const connections = [
        { name: 'vps', observation: { state: 'connected' }, owed_stops: { runs: 1, starts: 0 } },
        { name: 'old-box', removed: true, observation: { state: 'removed' }, owed_stops: { runs: 1, starts: 1 } },
        { name: 'quiet', removed: true, observation: { state: 'removed' }, owed_stops: { runs: 0, starts: 0 } },
    ];
    assert.equal(removedOwedLine(connections), 'Emergency Stop: 2 runs are not yet confirmed stopped on old-box, '
        + 'which is no longer saved; the stop is delivered only after a connection named old-box with the same SSH '
        + 'destination and engine home is saved again, and the work may continue there until then.');
    assert.equal(removedOwedLine(connections.slice(0, 1)), '');  // a saved row's stop is said on its own row
    assert.equal(removedOwedLine(undefined), '');
});

test('an engine\'s page has its own path, title and stored name; the list\'s heads say what was observed', () => {
    assert.deepEqual([enginePath('vps'), enginePath()], ['accounts/engine/vps', 'accounts/engine/new']);
    assert.deepEqual(parseEnginePath('accounts/engine/new'), { name: '', fresh: true });
    assert.deepEqual(parseEnginePath('accounts/engine/a%20b'), { name: 'a b', fresh: false });
    assert.equal(parseEnginePath('accounts/engine/x/y'), null);
    assert.equal(parseEnginePath('accounts/key/openai'), null, 'a key\'s page is not an engine\'s');
    assert.equal(engineTitle('accounts/engine/new'), 'New remote engine');
    // A new engine never takes a word a path or the owned engine already means.
    assert.deepEqual(['Build box', 'New', 'Local', 'default'].map((label) => engineNameFor(label)), ['build-box', 'new-2', 'local-2', 'default-2']);
    const settings = { OUROBOROS_ENGINE_CONNECTIONS: [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }, { name: 'lab', ssh_host: 'lab', enabled: false }] };
    assert.deepEqual(engineHeads(settings, null).map((head) => [head.name, head.chip, head.meta]),
        [['vps', null, 'ouroboros@vds'], ['lab', null, 'lab · Off: takes no new work']], 'an unread status claims nothing');
    const [vps] = engineHeads(settings, [{ name: 'vps', observation: { state: 'connected', engine_version: '5.1.0' }, owed_stops: { runs: 1, starts: 0 } }]);
    assert.deepEqual(vps.chip, { text: '1 stop owed', tone: 'warn' });
    const [unread] = engineHeads(settings, [{ name: 'vps', observation: { state: 'unread' }, owed_stops: null }]);
    assert.deepEqual(unread.chip, { text: 'Status unknown', tone: 'warn' }, 'a status that could not be read is said, never "connected"');
    assert.equal(engineTitle('accounts/engine/vps'), 'Build box', 'a head names its page by the saved label');
    // Search finds the section by the words an owner uses for it.
    const row = HAND_WRITTEN.find((entry) => entry.id === 'OUROBOROS_ENGINE_CONNECTIONS');
    assert.equal(row.page, 'accounts');
    assert.ok(['VPS', 'SSH', 'Remote engine'].every((word) => row.aliases.includes(word)));
});

test('the section\'s More says what runs where, what Emergency Stop reaches and that nothing falls back', () => {
    for (const words of [/never installs, starts, updates or stops it/, /Ouroboros acts only on its own runs and never shows it/,
        /while the tools it calls still run on this computer/, /An agent you put there runs on that host: its tools and files are there/,
        /the copy stays on the host until that run's retention ends/,
        /its stop may not reach that engine: the work there can continue until this computer reconnects/,
        /is never moved to another engine, this computer's included: switched off, removed or unreachable, it is refused/,
        /Fallback models you set are separate choices and apply as they do for any route/,
        /In a declared folder it works on the host directly: its changes stay in that folder, and only its work product and review evidence come back here/,
        /an older version refuses a remote choice/]) assert.match(ENGINE_DETAILS, words);
    assert.doesNotMatch(ENGINE_DETAILS, /nothing comes back|never falls back to this computer/);
});

test('a remote route is named where it runs, never as an API key that cannot send (capsule, Models, Accounts, home)', async () => {
    const index = buildModelIndex({ catalog: { model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }], items: [] }, connections: [] });
    const remote = { model: 'claudexor@vps::codex=gpt-6', use_local: false };
    const said = describeRoute(index, remote);
    assert.deepEqual([said.lane, said.sourceLabel, said.pay, said.harness, said.facts],
        ['remote', 'On vps · Codex', 'subscription', 'codex', 'On vps · Codex · subscription']);
    assert.equal(sourceUnlisted(index, remote), false, 'no key read is asked for to explain it');
    // Positive control: an API route with no key still says so, and is still an unlisted API source.
    const api = describeRoute(index, { model: 'openai::gpt-x', use_local: false });
    assert.deepEqual([api.lane, api.sourceLabel, sourceUnlisted(index, { model: 'openai::gpt-x' })], ['api', 'OpenAI (no key)', true]);
    assert.match(sourceMark(said), /^<svg class="model-source-mark"/, 'a remote Codex route wears Codex\'s mark');
    assert.match(sourceMark({ lane: 'api', kind: '', sourceLabel: 'On vps' }), /model-source-initial/);
    assert.equal(routeKeyProblem('claudexor@vps::codex=gpt-6', {}, {}), '');
    assert.equal(routeKeyProblem('openai::gpt-x', {}, {}), 'no key');
    assert.deepEqual(routeWords('claudexor@vps::codex=gpt-6', { account: 'remote-main', engines: [{ name: 'vps', label: 'Build box' }] }),
        { name: 'gpt-6', mark: 'codex', source: 'On Build box · Codex subscription · account remote-main' });
    assert.match(routeWords('claudexor@vps::codex=gpt-6').source, /^On vps · Codex subscription · its accounts in turn$/);
    const slots = [{ slot: 'main', settingKey: 'OUROBOROS_MODEL', label: 'Main Model' }, { slot: 'light', settingKey: 'OUROBOROS_MODEL_LIGHT', label: 'Light Model' }];
    const usage = connectionUsage({ OUROBOROS_MODEL: 'claudexor@vps::codex=gpt-6', OUROBOROS_MODEL_LIGHT: 'openai::m' }, slots);
    const named = usage instanceof Map ? [...usage.keys()] : Object.keys(usage.used || usage);
    assert.ok(named.includes('openai') && !named.some((id) => id.startsWith('remote:')), 'a remote route names no key of this computer');
    const { homeAnswers } = await import('../modules/settings_home.js');
    assert.deepEqual(homeAnswers({ settings: { OUROBOROS_MODEL: 'claudexor@vps::codex=gpt-6', _meta: {} } }).models, { text: 'gpt-6', owner: false });
    assert.equal(homeAnswers({ settings: { OUROBOROS_MODEL: 'openai::gpt-x', _meta: {} } }).models.owner, true, 'positive control: a keyless API Main asks the owner');
});

test('a remote role keeps that engine\'s own account through collect and a role page\'s Save body', () => {
    const settings = { OUROBOROS_MODEL: 'claudexor@vps::codex=gpt-6', OUROBOROS_MODEL_ACCOUNTS: { main: 'remote-main' } };
    const all = createModelRolesEditor({ hostId: 'remote-all', doc: () => null });
    all.load(settings, contract);
    assert.equal(all.collect().OUROBOROS_MODEL, 'claudexor@vps::codex=gpt-6');
    assert.deepEqual(all.collect().OUROBOROS_MODEL_ACCOUNTS, { main: 'remote-main' });
    all.destroy();
    const one = createModelRolesEditor({ hostId: 'remote-one', doc: () => null, only: { slot: 'main', index: 0 } });
    one.load(settings, contract);
    const entry = one.entry();
    assert.deepEqual([entry.value, entry.account], ['claudexor@vps::codex=gpt-6', 'remote-main']);
    assert.deepEqual(roleBody(settings, entry).OUROBOROS_MODEL_ACCOUNTS, { main: 'remote-main' });
    one.destroy();
    // Positive control: an API role has no account to keep.
    const api = createModelRolesEditor({ hostId: 'api-one', doc: () => null, only: { slot: 'main', index: 0 } });
    api.load({ OUROBOROS_MODEL: 'openai::gpt-x', OUROBOROS_MODEL_ACCOUNTS: { main: 'stale' } }, contract);
    assert.equal(api.entry().account, '');
    api.destroy();
});

test('an Agents row on a remote engine passes the editor\'s own row check and keeps its location when saved', async () => {
    const { buildAvailableSubagentsSetting, validateAvailableSubagentsSetting } = await import('../modules/subagents_settings.js');
    const setting = { enabled: true, items: [{ subagent_id: 'r1', recommended_use: 'x', access: 'full',
        route: { kind: 'agent_session', target_id: 'codex=gpt-6', engine: 'vps', workspace: '/srv/project' } }] };
    assert.deepEqual(validateAvailableSubagentsSetting(setting), []);
    assert.deepEqual(buildAvailableSubagentsSetting(setting).items[0].route,
        { kind: 'agent_session', target_id: 'codex=gpt-6', engine: 'vps', workspace: '/srv/project' });
});

const REMOTE = [{ name: 'vps', label: 'Build box', state: 'connected', harnesses: ['codex'], sources: [
    { id: 'codex', label: 'Codex', credentialHarness: 'codex',
        models: [{ id: 'gpt-6-remote', label: 'GPT-6 remote', value: 'claudexor@vps::codex=gpt-6-remote' }] }] }];

test('a remote route suggests only that engine\'s models, never this computer\'s same-named source', async () => {
    const { remoteCatalogItems, routeModelSuggestions } = await import('../modules/route_editor_primitives.js');
    const local = { id: 'gpt-local-only', value: 'claudexor::codex=gpt-local-only' };
    const items = [local, ...remoteCatalogItems(REMOTE)];
    assert.deepEqual(remoteCatalogItems(REMOTE), [{ id: 'gpt-6-remote', value: 'claudexor@vps::codex=gpt-6-remote', label: 'GPT-6 remote' }]);
    assert.deepEqual(remoteCatalogItems(REMOTE, { engine: 'lab' }), []);
    assert.deepEqual(remoteCatalogItems([{ name: 'lab', state: 'offline' }]), [], 'an engine not connected lists nothing');
    assert.deepEqual(routeModelSuggestions({ kind: 'api_model', target_id: 'claudexor@vps::codex=' }, items), ['gpt-6-remote']);
    assert.deepEqual(routeModelSuggestions({ kind: 'api_model', target_id: 'claudexor::codex=' }, items), ['gpt-local-only'],
        'the local source keeps its own models');
});

test('an Agents row on a remote engine offers that engine\'s models and no account of this computer', async () => {
    const { availableSubagentRowMarkup } = await import('../modules/subagents_settings.js');
    const { rowStatus } = await import('../modules/subagent_status_primitives.js');
    const snapshot = { harnesses: [{ id: 'codex', enabled: true, status: 'ok', models: [{ id: 'gpt-local-only' }] }],
        profiles: { profiles: [{ profile: { harness_id: 'codex', profile_id: 'local-codex', display_name: 'Local Codex', enabled: true },
            status: { verification: 'passed', availability: 'available' } }] } };
    const state = { snapshot, catalogKnown: true, accountsKnown: true, quotaKnown: true, remoteEngines: REMOTE };
    const row = (extra) => ({ subagent_id: 'r1', recommended_use: 'x', access: 'full',
        route: { kind: 'agent_session', target_id: 'codex=', ...extra } });
    const accounts = (html) => html.match(/<select[^>]*data-subagent-field="account"[^>]*>([\s\S]*?)<\/select>/)[1];
    const localHtml = availableSubagentRowMarkup(row({}), state);
    assert.match(accounts(localHtml), /local-codex/, 'positive control: a local row lists the local account');
    assert.doesNotMatch(localHtml, /data-subagent-field="workspace"/, 'and has no remote folder');
    const remoteHtml = availableSubagentRowMarkup(row({ engine: 'vps', credential_profile_id: 'remote-main', workspace: '/srv/gone' }), state);
    assert.doesNotMatch(accounts(remoteHtml), /local-codex/);
    assert.match(accounts(remoteHtml), /value="remote-main" selected/, 'the saved remote pin stays chosen');
    assert.match(remoteHtml, /gpt-6-remote/);
    assert.doesNotMatch(remoteHtml, /gpt-local-only/);
    assert.match(remoteHtml, /data-subagent-field="workspace"/);
    assert.match(remoteHtml, /\/srv\/gone · not declared on Build box/, 'a saved folder the engine no longer declares is kept and said');
    assert.match(rowStatus(row({ engine: 'vps' }), state).text,
        /^On Build box · its tools and files are on that host .*plan turns are not offered there\. An Emergency Stop here may leave it working until this computer reconnects, and it is never moved to another engine, this computer's included$/);
    assert.match(rowStatus(row({ engine: 'gone' }), state).text, /^On gone · /, 'an engine no longer listed keeps its saved name');
    // A model there only answers: the tools its child calls run on this computer.
    const model = rowStatus({ route: { kind: 'api_model', target_id: 'claudexor@vps::codex=gpt-6-remote' } }, state);
    assert.equal(model.label, 'Checked at start');
    assert.match(model.text, /^On Build box · its requests go to that host under that engine's accounts while the tools it calls run on this computer;/);
    assert.doesNotMatch(model.text, /its tools and files are on that host/);
    assert.notEqual(rowStatus(row({}), state).text, rowStatus(row({ engine: 'vps' }), state).text);
});

const ROSTERED = [{ ...REMOTE[0], accounts_state: 'ok', accounts: [
    { harness_id: 'codex', profile_id: 'remote-main', label: 'Remote main', enabled: true, verification: 'passed' },
    { harness_id: 'codex', profile_id: 'remote-spare', label: 'remote-spare', enabled: false, verification: 'not_run' },
    { harness_id: 'claude', profile_id: 'other-harness', label: 'other-harness', enabled: true, verification: 'passed' }] }];

test('remote pins come from that engine\'s roster, with disabled, unverified and unread said plainly', async () => {
    const { REMOTE_ROSTER_UNKNOWN, remoteProfileOptions, routeSupportsAccount } = await import('../modules/route_editor_primitives.js');
    const { engineAccountsLine } = await import('../modules/settings_engine_connections.js');
    assert.deepEqual(remoteProfileOptions(ROSTERED, 'vps', 'codex', 'remote-main').map((option) => [option.value, option.label]), [
        ['', 'automatic on Build box'], ['remote-main', 'Remote main · remote-main'], ['remote-spare', 'remote-spare (disabled, not verified)']]);
    assert.match(remoteProfileOptions(ROSTERED, 'vps', 'codex', 'gone-pin').at(-1).label, /^gone-pin · not on Build box$/);
    const unread = remoteProfileOptions([{ ...REMOTE[0], accounts_state: 'unread', accounts_code: 'accounts_unavailable' }],
        'vps', 'codex', 'remote-main');
    assert.deepEqual(unread.slice(1).map((option) => [option.value, option.label, Boolean(option.disabled)]), [
        ['remote-main', 'remote-main · not checked', false],
        [REMOTE_ROSTER_UNKNOWN, 'Accounts on Build box could not be read (accounts_unavailable)', true]]);
    assert.equal(remoteProfileOptions([], 'lab', 'codex', '').at(-1).label, 'Accounts on lab: check it in Accounts › Remote engines');
    assert.ok(routeSupportsAccount({ kind: 'api_model', target_id: 'claudexor@vps::codex=m' }));
    assert.match(engineAccountsLine({ state: 'connected', accounts_state: 'ok', accounts: ROSTERED[0].accounts }),
        /^Accounts there: Remote main \(codex\) · remote-spare \(codex, disabled, not verified\)/);
    assert.equal(engineAccountsLine({ state: 'connected', accounts_state: 'unread', accounts_code: 'malformed_response' }),
        'Accounts there could not be read (malformed_response)');
    assert.equal(engineAccountsLine({ state: 'offline' }), '');
});

test('an Agents row on a remote engine pins one of that engine\'s accounts, API and session rows alike', async () => {
    const { availableSubagentRowMarkup } = await import('../modules/subagents_settings.js');
    const state = { snapshot: { harnesses: [], profiles: { profiles: [{ profile: { harness_id: 'codex', profile_id: 'local-codex',
        enabled: true }, status: { verification: 'passed' } }] } }, catalogKnown: true, accountsKnown: true, remoteEngines: ROSTERED };
    const accounts = (html) => [...html.match(/<select[^>]*data-subagent-field="account"[^>]*>([\s\S]*?)<\/select>/)[1]
        .matchAll(/<option value="([^"]*)"/g)].map(([, value]) => value);
    const session = availableSubagentRowMarkup({ subagent_id: 's', recommended_use: 'x', access: 'full',
        route: { kind: 'agent_session', target_id: 'codex=gpt-6-remote', engine: 'vps', credential_profile_id: 'remote-main' } }, state);
    assert.deepEqual(accounts(session), ['', 'remote-main', 'remote-spare']);
    const api = availableSubagentRowMarkup({ subagent_id: 'a', recommended_use: 'x',
        route: { kind: 'api_model', target_id: 'claudexor@vps::codex=gpt-6-remote', credential_profile_id: 'remote-spare' } }, state);
    assert.deepEqual(accounts(api), ['', 'remote-main', 'remote-spare']);
    assert.match(api, /<option value="remote-spare" selected>/);
});

/* ---- Settings › Accounts and keys › Remote engines, mounted as the shell mounts it --------------------------------- */

// A field's value starts as its `value` attribute, as a browser's does.
Object.defineProperty(NodeStub.prototype, 'value', { configurable: true,
    get() { return this.typed ?? this.attrs.value ?? ''; }, set(text) { this.typed = String(text); } });
const settle = async (rounds = 16) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const KEY = 'OUROBOROS_ENGINE_CONNECTIONS';

/** The settings server with its remote engines: the document with each key's token, a 409 for a list that changed
 *  since its token was read, and the engine endpoints (`status`: GET /api/engine-connections, `null` = unreadable;
 *  `checks` and `catalogs` by name). No host is contacted by any of them here; every request is recorded. */
function server(t, { rows = [], status = [], checks = {}, catalogs = {}, disconnect = { closed: true } } = {}) {
    forgetSettingRows();
    const env = setup(t);
    const state = { calls: [], revision: 1, rows: structuredClone(rows), status: structuredClone(status), checks, catalogs, disconnect, answer: true };
    const documentNow = () => ({ OUROBOROS_MODEL: 'claudexor@vps::codex=gpt-6', OUROBOROS_CONNECTIONS: [], [KEY]: structuredClone(state.rows),
        _meta: { connections: [], effects: { [KEY]: 'next_call' }, tokens: { [KEY]: `engines-${state.revision}`, OUROBOROS_CONNECTIONS: 'list-1' },
            setup_contract: { modelSlots: [{ slot: 'main', settingKey: 'OUROBOROS_MODEL', label: 'Main Model' }] } } });
    globalThis.fetch = async (url, init = {}) => {
        const path = String(url).split('?')[0];
        const method = init.method || 'GET';
        const body = init.body ? JSON.parse(init.body) : null;
        state.calls.push({ method, path, body });
        if (path === '/api/settings' && method === 'GET') return response(documentNow());
        if (path === '/api/settings') {
            const { if_unchanged: wanted, ...changed } = body;
            if (wanted?.[KEY] && wanted[KEY] !== `engines-${state.revision}`) {
                return response({ saved: false, code: 'settings_changed_elsewhere', error: 'Changed on another device. Reload the current value before saving.' }, 409);
            }
            state.rows = structuredClone(changed[KEY]);
            state.revision += 1;
            return response({ status: 'saved', saved: true, changes: { [KEY]: { withheld: 'structure', token: `engines-${state.revision}` } } });
        }
        if (path === '/api/extensions') return response({ skills: [], live: { settings_sections: [] } });
        if (path === '/api/engine-connections') {
            return state.status === null ? response({ error: 'status unreadable' }, 500) : response({ connections: state.status });
        }
        const engine = path.match(/^\/api\/engine-connections\/([^/]+)\/(catalog|check|disconnect)$/);
        if (engine) {
            const [, name, act] = engine;
            if (act === 'catalog') return response(state.catalogs[decodeURIComponent(name)] || { name, state: 'offline' });
            if (act === 'check') {
                const result = state.checks[decodeURIComponent(name)] || { name, ok: false, code: 'remote_engine_not_running', message: 'No running engine' };
                if (result.ok) state.status = state.status.map((row) => (row.name === name ? { ...row, observation: { state: 'connected' } } : row));
                return response(result);
            }
            return response({ name, ...state.disconnect });
        }
        throw new Error(`Unexpected request ${method} ${url}`);
    };
    env.document.createElement = (tag) => new Stub(tag);
    const container = new Stub('div');
    const beside = new Stub('div');  // the list the shell keeps under an opened page
    env.document.body.appendChild(beside);
    env.document.body.appendChild(container);
    const asked = [];
    const opened = [];
    const ctx = { readSettings, openSettings: (path) => { opened.push(path); return true; },
        ask: async (question) => { asked.push(question); return state.answer; } };
    const mount = async (path, into = container) => {
        const handle = await settingsPageFor(path).mount(into, { path, row: '', ctx });
        globalThis.window.dispatchEvent({ type: 'ouro:settings-page-shown', detail: { path } });
        await settle();
        return handle;
    };
    const posts = () => state.calls.filter((call) => call.method === 'POST' && call.path === '/api/settings').map((call) => call.body);
    const click = async (selector) => { await container.fire('click', { target: container.querySelector(selector) }); await settle(); };
    const type = async (field, text) => {
        const input = container.querySelector(`[data-engine-field="${field}"]`);
        input.value = text;
        await container.fire('input', { target: input });
    };
    return { state, container, beside, asked, opened, mount, posts, click, type,
        find: (selector) => container.querySelector(selector), text: (selector) => container.querySelector(selector)?.textContent || '' };
}
const contacted = (world) => world.state.calls.filter((call) => /\/(check|disconnect)$/.test(call.path));

test('the list draws a head per saved engine from what this computer observed and contacts no host; a head\'s switch writes one entry', async (t) => {
    const world = server(t, {
        rows: [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }, { name: 'lab', ssh_host: 'lab', enabled: false }],
        status: [{ name: 'vps', observation: { state: 'connected', engine_version: '5.1.0' }, owed_stops: { runs: 1, starts: 0 } },
            { name: 'lab', observation: { state: 'offline' }, owed_stops: { runs: 0, starts: 0 } },
            { name: 'old-box', removed: true, observation: { state: 'removed' }, owed_stops: { runs: 1, starts: 0 } }] });
    const list = await world.mount('accounts');
    const heads = () => world.find('[data-engine-rows]').children;
    assert.deepEqual(heads().map((head) => head.dataset.engineName), ['vps', 'lab']);
    assert.deepEqual(heads().map((head) => head.dataset.openPath), ['accounts/engine/vps', 'accounts/engine/lab']);
    assert.match(heads()[0].textContent, /Build box.*1 stop owed/s);
    assert.match(heads()[1].textContent, /Off: takes no new work/);
    assert.match(world.text('[data-engine-owed]'), /old-box, which is no longer saved/);
    assert.equal(world.find('[data-engine-owed]').hidden, false);
    assert.deepEqual(contacted(world), [], 'drawing the list checks no engine');
    assert.match(world.find('[data-engine-more]').textContent, /never installs, starts, updates or stops it/);

    const lab = heads()[1].querySelector('[data-engine-switch]');
    lab.checked = true;
    await world.container.fire('change', { target: lab });
    await settle();
    const sent = world.posts().at(-1);
    assert.deepEqual(sent[KEY], [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }, { name: 'lab', ssh_host: 'lab' }]);
    assert.deepEqual(sent.if_unchanged, { [KEY]: 'engines-1' }, 'one entry, with the list\'s token');
    assert.equal(world.posts().length, 1);
    list.unmount();
});

test('a new engine needs its name and SSH destination; saved, it takes a stored name and the list says so', async (t) => {
    const world = server(t, { rows: [{ name: 'build-box', label: 'Old box', ssh_host: 'old' }], status: [] });
    const list = await world.mount('accounts', world.beside);
    const page = await world.mount('accounts/engine/new');
    assert.equal(world.text('[data-engine-status]'), 'Not saved');
    assert.equal(world.find('[data-engine-check]'), null, 'a form never saved has nothing to check');
    await world.click('[data-editor-save]');
    assert.equal(world.text('[data-editor-result]'), 'Not saved. Name the new remote engine.');
    await world.type('label', 'Build box');
    await world.click('[data-editor-save]');
    assert.equal(world.text('[data-editor-result]'), 'Not saved. Enter its SSH destination.');
    assert.equal(world.posts().length, 0);
    await world.type('ssh_host', 'ouroboros@vds');
    await world.type('remote_root', '/home/ouroboros/runs');
    await world.type('workspaces', '/srv/a\n\n  /srv/b ');
    await world.click('[data-editor-save]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'build-box', label: 'Old box', ssh_host: 'old' },
        { name: 'build-box-2', label: 'Build box', ssh_host: 'ouroboros@vds', remote_root: '/home/ouroboros/runs', workspaces: ['/srv/a', '/srv/b'] }],
    'the stored name is minted once and never takes a saved one');
    assert.deepEqual(world.opened, ['accounts']);
    assert.equal(world.beside.querySelector('[data-engine-result]').textContent, 'Added Build box.');
    page.unmount();
    list.unmount();
});

test('a saved engine\'s page: Check reads the saved engine and its accounts, an edit sets Check aside, Save writes what changed', async (t) => {
    const world = server(t, {
        rows: [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }],
        status: [{ name: 'vps', observation: { state: 'offline' }, owed_stops: { runs: 0, starts: 0 } }],
        checks: { vps: { name: 'vps', ok: true, engine_version: '5.1.0', store: 'f11dacf1-5052', harnesses: ['codex', 'claude'], model_sources: ['codex'] } },
        catalogs: { vps: { name: 'vps', state: 'connected', sources: [], harnesses: ['codex'], accounts_state: 'ok',
            accounts: [{ harness_id: 'codex', profile_id: 'vps-owner', label: 'vps-owner', enabled: true, verification: 'passed' }] } } });
    const page = await world.mount('accounts/engine/vps');
    assert.equal(world.find('[data-engine-field="ssh_host"]').value, 'ouroboros@vds');
    assert.equal(world.text('[data-engine-status]'), 'Not connected');
    assert.deepEqual(contacted(world), [], 'opening the page checks nothing');
    await world.click('[data-engine-check]');
    assert.deepEqual(contacted(world).map((call) => `${call.method} ${call.path}`), ['POST /api/engine-connections/vps/check']);
    assert.equal(world.text('[data-editor-result]'), 'Connected: codex · agents: codex, claude');
    assert.equal(world.text('[data-engine-status]'), 'Connected');
    assert.match(world.text('[data-engine-meta]'), /engine 5\.1\.0 · store …1-5052/);
    assert.equal(world.text('[data-engine-accounts]'), 'Accounts there: vps-owner (codex)');

    await world.type('remote_root', '/srv/runs');
    assert.equal(world.text('[data-engine-status]'), 'Unsaved changes');
    assert.equal(world.find('[data-engine-check]').disabled, true, 'a check speaks for the saved engine, never the typed fields');
    assert.equal(world.find('[data-engine-accounts]').hidden, true);
    await world.click('[data-editor-save]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds', remote_root: '/srv/runs' }]);
    assert.equal(world.text('[data-editor-result]'), 'Saved · applies from the next model call', 'the server\'s own effect for the key');
    assert.equal(world.find('[data-engine-check]').disabled, false);

    // Changed on another device since this form read it: shown, never written over.
    world.state.rows = [{ name: 'vps', label: 'Build box', ssh_host: 'other@vds', remote_root: '/srv/runs' }];
    world.state.revision += 1;
    await world.type('label', 'Renamed box');
    const before = world.posts().length;
    await world.click('[data-editor-save]');
    assert.equal(world.posts().length, before);
    assert.equal(world.text('[data-editor-result]'), 'Changed on another device. This is the current value.');
    assert.equal(world.find('[data-engine-field="ssh_host"]').value, 'other@vds');

    await world.click('[data-engine-toggle]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'vps', label: 'Build box', ssh_host: 'other@vds', remote_root: '/srv/runs', enabled: false }]);
    assert.equal(world.text('[data-editor-result]'), 'Turned off · applies from the next model call');
    page.unmount();
});

test('Turn off with unsaved edits finishes its button, keeps the draft as typed, and the draft\'s Save keeps the switch', async (t) => {
    const world = server(t, {
        rows: [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }],
        status: [{ name: 'vps', observation: { state: 'offline' }, owed_stops: { runs: 0, starts: 0 } }] });
    const page = await world.mount('accounts/engine/vps');
    await world.type('remote_root', '/srv/runs');
    await world.click('[data-engine-toggle]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds', enabled: false }],
        'the switch writes the saved entry alone, never the draft');
    const toggle = world.find('[data-engine-toggle]');
    assert.equal(toggle.textContent, 'Turn on');
    assert.equal(toggle.disabled, false, 'the switch finished: its button is usable again');
    assert.equal(world.text('[data-editor-result]'), 'Turned off · applies from the next model call');
    assert.equal(world.find('[data-engine-field="remote_root"]').value, '/srv/runs', 'the draft stays as typed');
    assert.equal(world.text('[data-engine-status]'), 'Unsaved changes');
    await world.click('[data-editor-save]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds',
        remote_root: '/srv/runs', enabled: false }], 'this page\'s own switch is no change made elsewhere');
    assert.equal(world.text('[data-editor-result]'), 'Saved · applies from the next model call');
    page.unmount();
});

test('an engine\'s status that cannot be read is "Status unknown", and a Disconnect this computer cannot confirm is said so', async (t) => {
    const world = server(t, { rows: [{ name: 'vps', ssh_host: 'ouroboros@vds' }], status: null, disconnect: { closed: null } });
    const page = await world.mount('accounts/engine/vps');
    assert.equal(world.text('[data-engine-status]'), 'Status unknown');
    assert.equal(world.find('[data-engine-status]').dataset.tone, 'warn');
    assert.match(world.text('[data-engine-owed]'), /could not be read/, 'whether a stop is owed is unknown too, never "none"');
    await world.click('[data-engine-disconnect]');
    assert.match(world.text('[data-editor-result]'), /^Unconfirmed: this computer could not confirm its tunnel stopped/);
    assert.equal(world.text('[data-engine-status]'), 'Stop unconfirmed');
    page.unmount();
});

test('Remove asks first, says the stops still owed there, and writes the engine out of the list only when confirmed', async (t) => {
    const world = server(t, { rows: [{ name: 'vps', label: 'Build box', ssh_host: 'ouroboros@vds' }, { name: 'lab', ssh_host: 'lab' }],
        status: [{ name: 'vps', observation: { state: 'offline' }, owed_stops: { runs: 1, starts: 0 } }] });
    const page = await world.mount('accounts/engine/vps');
    assert.match(world.text('[data-engine-owed]'), /^Emergency Stop: 1 run is not yet confirmed stopped on this engine/);
    world.state.answer = false;
    await world.click('[data-editor-remove]');
    assert.equal(world.posts().length, 0, 'declined: nothing is written');
    assert.equal(world.asked[0].title, 'Remove Build box?');
    assert.match(world.asked[0].body, /Work already running on that host continues there/);
    assert.match(world.asked[0].body, /1 run is not yet confirmed stopped on this engine.*removing it leaves those stops undelivered/s);
    world.state.answer = true;
    await world.click('[data-editor-remove]');
    assert.deepEqual(world.posts().at(-1)[KEY], [{ name: 'lab', ssh_host: 'lab' }]);
    assert.deepEqual(world.opened, ['accounts']);
    page.unmount();
});
