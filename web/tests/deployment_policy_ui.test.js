// A managed deployment's part of Settings and the wizard (settings_policy.js over `policy_state.deployment`): a
// bounded choice offers only what the policy file allows, a fixed one is said instead of edited, and nothing the
// employee cannot change is collected back. Unmanaged (no `deployment`), every surface stays as it was.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { allowedOptions, checkerFieldsNote, checkerReads, checkerUses, deploymentNote, deploymentOf, fixCheckerField,
    offerAllowedChoices, syncPolicyState } from '../modules/settings_policy.js';
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
    assert.equal(deploymentNote(DEPLOYMENT, 'coverage'), 'Set by your organization: Full');
    assert.equal(deploymentNote({ access_max: 'light' }, 'access'), 'Set by your organization: Light');
    assert.equal(deploymentNote(null, 'access'), '');
    assert.equal(deploymentOf({ policy_state: { deployment: DEPLOYMENT } }), DEPLOYMENT);
    assert.equal(deploymentOf({ policy_state: {} }), null);
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
    const input = { readOnly: false };
    const note = { hidden: true, textContent: '' };
    fixCheckerField(input, note, legacy, 'OPENAI_BASE_URL', 'Legacy OpenAI Base URL');
    assert.deepEqual([input.readOnly, note.hidden], [true, false]);
    assert.match(note.textContent, /^Legacy OpenAI Base URL: set by your organization/);
    fixCheckerField(input, note, null, 'OPENAI_BASE_URL', 'Legacy OpenAI Base URL');
    assert.deepEqual([input.readOnly, note.hidden, note.textContent], [false, true, ''], 'unmanaged: as it was');
    const shared = { enabled: true, configured: true, usedBy: ['Main'], legacyAddress: 'https://organization.invalid/v1' };
    assert.match(connectionRowFacts(shared).meta, /change the address in Settings → Accounts/);
    assert.doesNotMatch(connectionRowFacts({ ...shared, fixed: true }).meta, /change the address/,
        'a fixed row never points at a change its fields refuse');
});

test('the policy lines say a bound beside the facts and a fixed value alone', () => {
    const nodes = {};
    const root = { querySelector: (selector) => (selector.startsWith('[data-policy-state') ? (nodes[selector] ||= { textContent: '' }) : null) };
    const line = (key) => nodes[`[data-policy-state="${key}"]`].textContent;
    const same = { configured: 'full', current_process: 'full', next_task: 'full' };
    syncPolicyState(root, { policy_state: { access: { configured: 'pro', current_process: 'pro' }, supervisor: same,
        review: { configured: 'advisory', current_process: 'advisory' }, deployment: DEPLOYMENT } });
    assert.equal(line('access'), 'Pro now and after restart · Your organization allows up to Pro');
    assert.equal(line('supervisor'), 'Set by your organization: Full');
    assert.equal(line('supervisor_consciousness'), 'Set by your organization: Full');
    syncPolicyState(root, { policy_state: { access: { configured: 'pro', current_process: 'pro' }, supervisor: same,
        review: { configured: 'advisory', current_process: 'advisory' } } });
    assert.equal(line('access'), 'Pro now and after restart', 'unmanaged lines are unchanged');
    assert.equal(line('supervisor'), 'Full now and for new tasks');
});

function segmented(values, holder) {
    return { hidden: false, title: 'Lowering asks first', parentElement: holder, values,
        querySelectorAll() { return this.values.map((value) => ({ dataset: { effortValue: value } })); },
        set outerHTML(html) { holder.child = segmented(JSON.parse(html).map(({ value }) => value), holder); } };
}

test('a bounded segmented choice is drawn again by the one renderer; one left is no choice', () => {
    const holder = { child: null, querySelector() { return this.child; } };
    holder.child = segmented(LEVELS.map(({ value }) => value), holder);
    const root = { querySelector: () => holder.child };
    const renders = [];
    const binds = [];
    const render = ({ options, title }) => { renders.push({ options, title }); return JSON.stringify(options); };
    const fixed = offerAllowedChoices(root, { target: 's-safety-mode', options: LEVELS, axis: 'coverage' }, DEPLOYMENT,
        { render, bind: (node) => binds.push(node) });
    assert.equal(fixed, true);
    assert.deepEqual(renders.map(({ options }) => options.map(({ value }) => value)), [['full']]);
    assert.equal(renders[0].title, 'Lowering asks first', 'the drawn group keeps its own title');
    assert.deepEqual(binds, [holder], 'bound like the original, on its own holder');
    assert.equal(holder.child.hidden, true, 'a fixed level is said by its policy line, not offered');
    assert.equal(offerAllowedChoices(root, { target: 's-safety-mode', options: LEVELS, axis: 'coverage' }, DEPLOYMENT,
        { render, bind: (node) => binds.push(node) }), true);
    assert.equal(renders.length, 1, 'a reload draws nothing again');
    // Unmanaged: the drawn group already holds every option; nothing is drawn again or hidden.
    const plainHolder = { child: null, querySelector() { return this.child; } };
    plainHolder.child = segmented(LEVELS.map(({ value }) => value), plainHolder);
    const again = [];
    assert.equal(offerAllowedChoices({ querySelector: () => plainHolder.child }, { target: 's-safety-mode', options: LEVELS,
        axis: 'coverage' }, null, { render: (args) => again.push(args), bind: () => again.push('bind') }), false);
    assert.deepEqual(again, []);
    assert.equal(plainHolder.child.hidden, false);
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
