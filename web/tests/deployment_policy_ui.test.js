// A managed deployment's part of Settings and the wizard (settings_policy.js over `policy_state.deployment`): a
// bounded choice offers only what the policy file allows, a fixed one is said instead of edited, and nothing the
// employee cannot change is collected back. Unmanaged (no `deployment`), every surface stays as it was.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { allowedOptions, applyPolicyToRows, checkerFieldsNote, checkerReads, checkerUses, deploymentNote, deploymentOf,
    policyValueLabel, safetyExitsNote } from '../modules/settings_policy.js';
import { rowById } from '../modules/settings_rows.js';
import { connectionRowFacts } from '../modules/settings_connections.js';
import { createModelRolesEditor } from '../modules/model_roles.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));
const ACCESS = ['light', 'advanced', 'pro', 'cyber_pro'].map((value) => ({ value, label: value }));
const LEVELS = ['off', 'light', 'full'].map((value) => ({ value, label: value }));
const DEPLOYMENT = { source: 'deployment_policy', version: 1, digest: '0123456789abcdef', access_max: 'pro',
    min_coverage: 'full', organization_rules: true,
    checker: { model: 'anthropic@safety::checker', account: '', fallbacks: [], connections: ['anthropic@safety'], local: false } };

test('a ceiling and a floor leave only the choices the policy file allows; unmanaged leaves all', () => {
    assert.deepEqual(allowedOptions(ACCESS, DEPLOYMENT, 'access').map(({ value }) => value), ['light', 'advanced', 'pro']);
    assert.deepEqual(allowedOptions(LEVELS, DEPLOYMENT, 'coverage').map(({ value }) => value), ['full']);
    assert.deepEqual(allowedOptions(LEVELS, { min_coverage: 'light' }, 'coverage').map(({ value }) => value), ['light', 'full']);
    assert.equal(allowedOptions(ACCESS, null, 'access'), ACCESS);
    assert.equal(allowedOptions(LEVELS, { access_max: 'pro' }, 'coverage'), LEVELS, 'a field constrains only what it names');
    assert.equal(deploymentNote(DEPLOYMENT, 'access'), 'Your organization allows up to Pro');
    assert.equal(deploymentNote(DEPLOYMENT, 'coverage'), 'Set by your organization: + Commands');
    assert.equal(deploymentNote({ min_coverage: 'light' }, 'coverage'), 'Your organization requires at least Outgoing');
    assert.equal(deploymentNote({ access_max: 'light' }, 'access'), 'Set by your organization: Light');
    // `light` is a word of two scales: Access's unless the Safety scale is asked for.
    assert.deepEqual([policyValueLabel('light'), policyValueLabel('light', 'coverage'), policyValueLabel('full'), policyValueLabel('cyber_pro')],
        ['Light', 'Outgoing', '+ Commands', 'Cyber Pro']);
    assert.equal(deploymentNote(null, 'access'), '');
    assert.equal(deploymentOf({ policy_state: { deployment: DEPLOYMENT } }), DEPLOYMENT);
    assert.equal(deploymentOf({ policy_state: {} }), null);
});

test('the Safety line says only the exits the file closes, and nothing outside a deployment', () => {
    const note = (allowed_exits) => safetyExitsNote(allowed_exits === undefined ? null : { allowed_exits });
    const exits = ['repeat_after_dangerous', 'release_after_dangerous', 'release_unjudged'];
    const facts = ['Ouroboros cannot override a DANGEROUS refusal on its own',
        'your approval cannot override a DANGEROUS refusal', 'your approval cannot send a message the Supervisor has not checked'];
    assert.equal(note(exits), '', 'all open: no fixed fact to state');
    assert.equal(note(undefined), '');
    for (let mask = 0; mask < 8; mask++) {
        const allowed = exits.filter((_, i) => mask & (1 << i));
        const closed = facts.filter((_, i) => !(mask & (1 << i)));
        assert.equal(note(allowed), closed.length ? `Set by your organization: ${closed.join('; ')}.` : '');
    }
});

test('the checker names only its own connections and the local server', () => {
    assert.equal(checkerUses(DEPLOYMENT, 'anthropic@safety'), true);
    assert.equal(checkerUses(DEPLOYMENT, 'anthropic'), false, 'the built-in key stays the employee’s');
    assert.equal(checkerUses(DEPLOYMENT, 'local'), false);
    assert.equal(checkerUses({ checker: { connections: [], local: true } }, 'local'), true);
    assert.equal(checkerUses(null, 'anthropic@safety'), false);
});

test('a built-in checker fixes only the flat fields it reads, and says so beside them', () => {
    const legacy = { checker: { connections: ['openai-compatible'], local: false,
        fields: ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_COMPATIBLE_BASE_URL'] } };
    assert.equal(checkerReads(legacy, 'OPENAI_API_KEY'), true, 'the legacy pair borrows the OpenAI key');
    assert.equal(checkerUses(legacy, 'openai'), false, 'the OpenAI row itself stays the employee\u2019s');
    assert.equal(checkerReads({ checker: { fields: ['OPENAI_COMPATIBLE_API_KEY'] } }, 'OPENAI_API_KEY'), false);
    assert.equal(checkerReads(null, 'OPENAI_API_KEY'), false);
    assert.equal(checkerFieldsNote(['API key']), 'API key: set by your organization, its Safety Supervisor uses it.');
    assert.equal(checkerFieldsNote(['Base URL', 'API key']),
        'Base URL and API key: set by your organization, its Safety Supervisor uses them.');
    assert.equal(checkerFieldsNote([]), '');
    const shared = { enabled: true, configured: true, usedBy: ['Main'], legacyAddress: 'https://organization.invalid/v1' };
    assert.match(connectionRowFacts(shared).meta, /change the address in Settings → Accounts and keys/);
    assert.doesNotMatch(connectionRowFacts({ ...shared, fixed: true }).meta, /change the address/,
        'a fixed row never points at a change its fields refuse');
});

const POLICY_ROWS = ['OUROBOROS_RUNTIME_MODE', 'OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE'];
const policyRows = (policy_state) => {
    const rows = applyPolicyToRows(POLICY_ROWS.map((id) => structuredClone(rowById(id))), policy_state ? { _meta: { policy_state } } : null, rowById);
    return Object.fromEntries(rows.map((row) => [row.id, row]));
};
const offered = (row) => row.control.options.map((option) => option.value);

test('a row a deployment bounds offers only what is allowed and says whose limit it is; a fixed one says its value', () => {
    const managed = policyRows({ deployment: DEPLOYMENT });
    assert.deepEqual(offered(managed.OUROBOROS_RUNTIME_MODE), ['light', 'advanced', 'pro']);
    assert.equal(managed.OUROBOROS_RUNTIME_MODE.meta, 'Your organization allows up to Pro');
    for (const id of ['OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE']) {
        assert.deepEqual(offered(managed[id]), ['full'], 'a fixed level stays on the page as its one option, never hidden');
        assert.equal(managed[id].meta, 'Set by your organization: + Commands');
        // The chosen option's line is what the row shows: it carries the organization's sentence too.
        assert.equal(managed[id].control.options[0].note,
            'Shell commands and scripts are read first too. File edits are not. Set by your organization: + Commands.');
    }
    const floor = policyRows({ deployment: { min_coverage: 'light' } });
    assert.deepEqual(offered(floor.OUROBOROS_SAFETY_MODE), ['light', 'full']);
    assert.match(floor.OUROBOROS_SAFETY_MODE.control.options[0].note, / Your organization requires at least Outgoing\.$/);
    assert.deepEqual(offered(floor.OUROBOROS_RUNTIME_MODE), ['light', 'advanced', 'pro', 'cyber_pro'], 'a field constrains only what it names');
    // Unmanaged, or before any read: the rows as they are described.
    for (const plain of [policyRows({}), policyRows(null)]) {
        for (const id of POLICY_ROWS) assert.deepEqual([plain[id].meta, plain[id].control], [rowById(id).meta, rowById(id).control], id);
    }
    // The page's copies change; the descriptors never do.
    assert.equal(rowById('OUROBOROS_SAFETY_MODE').control.options.length, 3);
});

test('a managed Supervisor slot is shown, never collected: the raw choice stays the server’s to keep', () => {
    const settings = { OUROBOROS_MODEL: 'claudexor::codex=gpt-test', OUROBOROS_MODEL_SAFETY: 'anthropic@safety::checker',
        OUROBOROS_MODEL_ACCOUNTS: { main: 'mine', safety: 'organization' } };
    const managed = { ...contract, modelSlots: contract.modelSlots.map((slot) => ({ ...slot, managed: slot.slot === 'safety' })) };
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    editor.load(settings, managed);
    const collected = editor.collect();
    assert.ok(!('OUROBOROS_MODEL_SAFETY' in collected), 'no hidden control writes the checker');
    assert.equal(collected.OUROBOROS_MODEL, 'claudexor::codex=gpt-test');
    assert.deepEqual(collected.OUROBOROS_MODEL_ACCOUNTS, { main: 'mine', safety: 'organization' },
        'other roles keep their own entries; the Supervisor’s travels back exactly as loaded');
    assert.equal(editor.validate(), '');
    editor.destroy();
    const plain = createModelRolesEditor({ hostId: 'test', doc: () => null });
    plain.load(settings, contract);
    assert.equal(plain.collect().OUROBOROS_MODEL_SAFETY, 'anthropic@safety::checker', 'unmanaged: the row is the employee’s');
    plain.destroy();
});

test('a managed wait card never offers another model, in every branch, while an unmanaged one keeps its wording', async (t) => {
    const { installDom, restoreDom } = await import('./chat_dom_fixture.js');
    const { createModelWaitController } = await import('../modules/model_wait.js');
    const env = installDom();
    t.after(() => restoreDom(env.prior));
    const branches = { quota: { reason: 'quota' }, auth: { reason: 'auth' }, mixed: { reason: 'auth_quota' },
        pool: { reason: 'unavailable' }, keyOff: { reason: 'connection_unavailable', model: 'openai@work::checker' },
        unobserved: { reason: 'quota', availability_observation: 'unavailable' } };
    for (const managed of [true, false]) {
        for (const [name, patch] of Object.entries(branches)) {
            const root = document.createElement('div');
            const timelineEl = document.createElement('div');
            root.append(timelineEl);
            const record = { root, timelineEl, phaseEl: document.createElement('span'), finished: false };
            const controller = createModelWaitController({ getRecord: () => record });
            controller.observe('task', { model_waits: { w: { wait_id: 'w', revision: 1, task_attempt: 1, role: 'safety',
                model: 'claudexor::codex=checker', credential_harness: 'codex', credential_profile_id: 'org',
                auto_continue: true, state: 'waiting', ...patch, ...(managed ? { managed: true } : {}) } } });
            const card = root.querySelector('.model-wait-row');
            const said = card.querySelector('[data-wait-reset]').textContent;
            const label = `${managed ? 'managed' : 'unmanaged'} ${name}`;
            assert.equal(card.querySelector('[data-wait-change]').hidden, managed, label);
            assert.equal(/choose another model/i.test(said), !managed && ['keyOff', 'unobserved'].includes(name), `${label}: ${said}`);
            assert.equal(said.includes('Your organization sets this model'), managed, `${label}: ${said}`);
            assert.equal(said, said.trim(), label);
            if (managed && name === 'keyOff') assert.ok(!said.includes('Turn it back on'), said);
            controller.destroy();
        }
    }
});
