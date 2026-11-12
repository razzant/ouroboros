import test from 'node:test';
import assert from 'node:assert/strict';
import { activeModelWaits, createModelWaitController, isModelWaitReference, MODEL_WAIT_REASONS, mergeModelWaits, modelWaitAction, modelWaitReasonLabel, modelWaitRoleLabel, unsavedPersistenceNotice } from '../modules/model_wait.js';
import { desiredLiveCardPhase } from '../modules/task_phase_chip.js';
import { emptyWorkState, roomStatusView } from '../modules/work_status.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { room } from './work_fixture.js';

const row = (patch = {}) => ({ wait_id: 'wait-light', revision: 1, task_attempt: 1,
    role: 'light', model: 'claudexor::source-a=same-model', source: 'source-a',
    credential_harness: 'codex', credential_profile_id: 'personal', reason: 'quota',
    auto_continue: true, state: 'waiting', worker_slot_held: true, ...patch });

test('independent waiting roles keep their own account even on the same model', () => {
    const light = row();
    const main = row({ wait_id: 'wait-main', role: 'main', credential_profile_id: 'work' });
    const waits = mergeModelWaits({}, { [light.wait_id]: light, [main.wait_id]: main });
    assert.equal(activeModelWaits(waits).length, 2);
    assert.equal(waits['wait-light'].credential_profile_id, 'personal');
    assert.equal(waits['wait-main'].credential_profile_id, 'work');
    assert.deepEqual(activeModelWaits(waits, true), []);
});

test('out of order snapshots neither rewind auto-continue nor reopen a resolved episode', () => {
    const current = row({ revision: 4, auto_continue: false });
    const state = { [current.wait_id]: current };
    assert.equal(mergeModelWaits(state, { [current.wait_id]: row() })[current.wait_id], current);
    const resolved = row({ revision: 6, state: 'resolved', resolution: 'model_switched' });
    const next = mergeModelWaits(state, { [current.wait_id]: resolved });
    assert.deepEqual(activeModelWaits(next), []);
    assert.deepEqual(mergeModelWaits(next, { [current.wait_id]: row({ revision: 7 }) }), next);
});

test('a new task attempt does not display waits from its predecessor', () => {
    const old = row();
    const next = row({ wait_id: 'next-attempt', task_attempt: 2 });
    assert.deepEqual(activeModelWaits({ [old.wait_id]: old, [next.wait_id]: next }), [next]);
    assert.deepEqual(activeModelWaits({ [old.wait_id]: old }, false, 2), []);
});

test('current task attempt outranks historical waits even before its first wait', () => {
    const controller = createModelWaitController({ getRecord: () => null, doc: () => null });
    const old = row();
    controller.observe('task', { model_waits: { [old.wait_id]: old } });
    assert.equal(controller.waiting('task'), true);
    controller.observe('task', { activity_id: 'task', task_attempt: 2 });
    assert.equal(controller.waiting('task'), false);
    controller.observe('task', { model_waits: { [old.wait_id]: old } });
    controller.observe('task', { task_attempt: 1, status: 'completed' });
    assert.equal(controller.waiting('task'), false);
    const current = row({ wait_id: 'current-wait', task_attempt: 2 });
    controller.observe('task', { type: 'task_model_wait', ...current });
    assert.equal(controller.waiting('task'), true);
    controller.destroy();
});

test('same-revision acceptance is pending, not applied, and stale snapshots cannot erase it', () => {
    const initial = row();
    const pending = { ...initial, pending_action: { request_id: 'accepted', revision: initial.revision, action: 'retry' } };
    const state = mergeModelWaits({ [initial.wait_id]: initial }, { [initial.wait_id]: pending });
    assert.equal(state[initial.wait_id].pending_action.request_id, 'accepted');
    assert.equal(mergeModelWaits(state, { [initial.wait_id]: initial })[initial.wait_id], state[initial.wait_id]);
    const applied = row({ revision: 2, state: 'resolved', applied_request_id: 'accepted' });
    assert.deepEqual(activeModelWaits(mergeModelWaits(state, { [applied.wait_id]: applied })), []);
});

test('a malformed row cannot acquire actionable wait identity', () => {
    for (const patch of [{ revision: '1' }, { revision: 0 }, { task_attempt: null },
        { state: 'done' }, { reason: 'transport' }, { wait_id: 'different' }]) {
        assert.deepEqual(mergeModelWaits({}, { 'wait-light': row(patch) }), {});
    }
});

test('confirmed mixed access keeps an actionable replay row without inventing an account', () => {
    const mixed = row({ reason: 'auth_quota', credential_profile_id: '' });
    const state = mergeModelWaits({}, { [mixed.wait_id]: mixed });
    assert.deepEqual(activeModelWaits(state), [mixed]);
    assert.equal(state[mixed.wait_id].credential_profile_id, '');
    assert.deepEqual(mergeModelWaits({}, { [mixed.wait_id]: row({ reason: 'pool' }) }), {});
});

test('an engine-dated unavailable pool is its own actionable reason, not quota or sign-in', () => {
    const pool = row({ reason: 'unavailable', credential_profile_id: '', reset_at: '2099-01-01T00:00:00Z' });
    assert.deepEqual(activeModelWaits(mergeModelWaits({}, { [pool.wait_id]: pool })), [pool]);
    assert.ok(MODEL_WAIT_REASONS.includes('unavailable') && !MODEL_WAIT_REASONS.includes('transport'));
    // A route whose connection the owner disabled or removed waits on its own typed reason ($0, no observer).
    const gone = row({ reason: 'connection_unavailable', credential_profile_id: '', availability_observation: 'unavailable' });
    assert.deepEqual(activeModelWaits(mergeModelWaits({}, { [gone.wait_id]: gone })), [gone]);
});

test('a route whose API key was turned off or removed waits on its card instead of dropping out', () => {
    const keyOff = row({ reason: 'connection_unavailable', model: 'openrouter@claude-credits::anthropic/claude-opus-5.5',
        credential_profile_id: '' });
    assert.deepEqual(activeModelWaits(mergeModelWaits({}, { [keyOff.wait_id]: keyOff })), [keyOff]);
});

test('switch payload binds the exact wait revision and defaults to temporary role change', () => {
    const body = modelWaitAction('task-a', row(), 'switch', {
        model: 'openai::owner-model', credential_profile_id: '', ignored: 'never sent',
    }, 'request-a');
    assert.deepEqual(body, { request_id: 'request-a', decision_id: 'model_wait:task-a:wait-light',
        revision: 1, action: 'switch', model: 'openai::owner-model', credential_profile_id: '',
        use_local: false, persist_role: false });
    const saved = modelWaitAction('task-a', row(), 'switch', {
        model: row().model, credential_profile_id: 'work', persist_role: true,
    }, 'request-b');
    assert.equal(saved.persist_role, true);
    assert.equal(saved.credential_profile_id, 'work');
});

test('auto-continue and explicit retry are distinct text-free actions', () => {
    assert.deepEqual(modelWaitAction('t', row(), 'auto_continue', { auto_continue: false }, 'a'), {
        request_id: 'a', decision_id: 'model_wait:t:wait-light', revision: 1,
        action: 'auto_continue', auto_continue: false,
    });
    assert.deepEqual(modelWaitAction('t', row(), 'retry', {}, 'b'), {
        request_id: 'b', decision_id: 'model_wait:t:wait-light', revision: 1, action: 'retry',
    });
    assert.throws(() => modelWaitAction('t', row(), 'auto_continue', {}, 'c'));
});

test('role labels distinguish human-facing roles without model-name inference', () => {
    assert.equal(modelWaitRoleLabel('consciousness'), 'Consciousness');
    assert.equal(modelWaitRoleLabel('safety'), 'Safety Supervisor');
    assert.equal(modelWaitRoleLabel('fallback:2'), 'Fallback 3');
    assert.equal(modelWaitRoleLabel('reviewer:scope-a'), 'Reviewer · scope-a');
    assert.equal(modelWaitRoleLabel('subagent:writer'), 'Subagent · writer');
});

test('the typed history discriminator survives a chat transport wrapper', () => {
    assert.equal(isModelWaitReference({ type: 'chat', system_type: 'task_model_wait' }), true);
    assert.equal(isModelWaitReference({ type: 'task_model_wait' }), true);
    assert.equal(isModelWaitReference({ text: 'task_model_wait' }), false);
});

test('waiting has no computation animation and yields to real work and terminal truth', () => {
    // The room waits for the model (the server's room); the card says the access wait itself.
    assert.deepEqual(roomStatusView({ ...emptyWorkState(), room: room('waiting') }), {
        kind: 'online', text: 'Waiting', tone: 'held', mark: 'still', owner: false, amber: '',
    });
    assert.equal(roomStatusView({ ...emptyWorkState(), room: room('working') }).text, 'Working');
    assert.equal(desiredLiveCardPhase({ modelWaiting: true }).text, 'Waiting for access');
    assert.equal(desiredLiveCardPhase({ modelWaiting: true, cancelPendingPolicy: 'immediate' }).text, 'Cancelling…');
    assert.equal(desiredLiveCardPhase({ modelWaiting: true, finished: true }, 'done').text, 'Done');
});

// A wait that resumes by itself is a neutral chip with still dots; one only the owner can end (sign-in, automatic
// continuation off, unobservable access, an unproven account: the host's `requires_owner_action`) is the owner's move, amber.
// A parked task runs no wait loop, so its old waits ask nothing of the owner (gateway/main_decisions.owner_actions).
test('only a wait that needs the owner turns the card amber, and a parked card ignores its old waits', () => {
    const look = (record) => { const view = desiredLiveCardPhase(record); return [view.text, view.tone, view.owner, view.mark]; };
    assert.deepEqual(look({ modelWaiting: true }), ['Waiting for access', 'neutral', false, 'still']);
    assert.deepEqual(look({ modelWaiting: true, modelOwnerAction: true }), ['Waiting for access', 'owner', true, 'still']);
    assert.deepEqual(look({ modelWaiting: false, modelOwnerAction: true }), ['Working', 'live', false, 'moving']);
    assert.deepEqual(look({ modelWaiting: true, modelOwnerAction: true, parkedPhase: 'budget_paused' }),
        ['Paused', 'neutral', false, 'pause']);
});

test('the controller tells its card whether an admitted wait needs the owner, from the host\'s own fact', (t) => {
    const env = installDom();
    t.after(() => restoreDom(env.prior));
    const root = document.createElement('div');
    const timelineEl = document.createElement('div');
    root.append(timelineEl);
    const record = { root, timelineEl, phaseEl: document.createElement('span'), finished: false };
    const controller = createModelWaitController({ getRecord: () => record });
    t.after(() => controller.destroy());
    const tone = () => [record.modelWaiting, record.modelOwnerAction, record.phaseEl.dataset.chipTone];
    // The wait's own row says the same: a quiet tray with a neutral sentence, amber only for the owner's move.
    const frame = () => { const node = root.querySelector('.model-wait-row');
        return [node?.dataset.owner, node?.querySelector('[data-wait-reason]')?.dataset.tone]; };
    controller.observe('task', { model_waits: { 'wait-light': row() } });
    assert.deepEqual(tone(), [true, false, 'neutral']);
    assert.deepEqual(frame(), ['0', 'neutral']);
    controller.observe('task', { model_waits: { 'wait-light': row({ revision: 2, auto_continue: false, requires_owner_action: true }) } });
    assert.deepEqual(tone(), [true, true, 'owner']);
    assert.deepEqual(frame(), ['1', 'warn']);
    controller.observe('task', { model_waits: { 'wait-light': row({ revision: 3, state: 'resolved' }) } });
    assert.deepEqual(tone(), [false, false, 'live']);
});

test('a requested Settings save that did not land is told only once its wait resolved', () => {
    const resolved = row({ state: 'resolved', role: 'reviewer:one', applied_request_id: 'mine' });
    assert.equal(unsavedPersistenceNotice(resolved, { request_id: 'mine', saved: false }),
        'Reviewer · one: model change applied; it was not saved in Settings.');
    assert.equal(unsavedPersistenceNotice(resolved, { request_id: 'mine', saved: null }),
        'Reviewer · one: model change applied; whether it was saved in Settings is unconfirmed.');
    assert.equal(unsavedPersistenceNotice(resolved, { request_id: 'another', saved: false }),
        'Reviewer · one: model wait ended; it was not saved in Settings.');
    assert.equal(unsavedPersistenceNotice(resolved, { request_id: 'mine', saved: true }), '', 'a landed save is not news');
    assert.equal(unsavedPersistenceNotice(row(), { request_id: 'mine', saved: false }), '', 'a waiting row says it inline');
    assert.equal(unsavedPersistenceNotice({ ...resolved, applied_request_id: undefined }, { request_id: 'mine', saved: null }),
        'Reviewer · one: model wait ended; whether it was saved in Settings is unconfirmed.', 'missing receipt proves no application outcome');
    assert.equal(unsavedPersistenceNotice(resolved, null), '', 'no Settings save was requested');
});

test('a wait of a work on its own model says so and offers no save in Settings, while its task can still switch', (t) => {
    const env = installDom();
    t.after(() => restoreDom(env.prior));
    const root = document.createElement('div');
    const timelineEl = document.createElement('div');
    root.append(timelineEl);
    const controller = createModelWaitController({ getRecord: () => ({ root, timelineEl, finished: false }) });
    t.after(() => controller.destroy());
    assert.equal(modelWaitRoleLabel('lead'), 'This work\'s model');
    controller.observe('task', { model_waits: { 'wait-light': row({ role: 'lead' }) } });
    assert.equal(root.querySelector('[data-wait-role]').textContent, 'This work\'s model');
    const persist = root.querySelector('[data-wait-persist]');
    assert.equal(root.querySelector('[data-wait-persist-offer]').hidden, true, '"Also save this role in Settings" is not offered');
    assert.equal(persist.disabled, true);
    assert.equal(persist.checked, false);
    assert.match(root.querySelector('[data-wait-scope]').textContent, /this work's model only/);
    persist.checked = true;
    controller.observe('task', { model_waits: { 'wait-light': row({ role: 'lead', revision: 2 }) } });
    assert.equal(persist.checked, false);
    assert.equal(modelWaitAction('task', row({ role: 'lead' }), 'switch', { model: 'openai::replacement' }, 'r').persist_role, false);
    // A role of Settings keeps the offer.
    const other = document.createElement('div');
    const otherTimeline = document.createElement('div');
    other.append(otherTimeline);
    const settingsRole = createModelWaitController({ getRecord: () => ({ root: other, timelineEl: otherTimeline, finished: false }) });
    t.after(() => settingsRole.destroy());
    settingsRole.observe('task', { model_waits: { 'wait-light': row({ role: 'light' }) } });
    assert.equal(other.querySelector('[data-wait-persist-offer]').hidden, false);
    assert.equal(other.querySelector('[data-wait-persist]').disabled, false);
});

test('each wait reason has the one label its row shows, and a route whose access cannot be watched says so', () => {
    assert.deepEqual(MODEL_WAIT_REASONS.map((reason) => modelWaitReasonLabel(reason)),
        ['Waiting for quota', 'Sign-in required', 'Waiting for access', 'Waiting for an account', 'Key unavailable']);
    assert.equal(modelWaitReasonLabel('quota', false), 'Waiting for provider access');
    assert.equal(modelWaitReasonLabel('connection_unavailable', false), 'Key unavailable', 'a key that is off is off, watched or not');
});

test('the picker tells its role editor the tier the waiting call pays, and no other setting', async () => {
    const { pickerDocument } = await import('../modules/model_wait.js');
    const settings = { OPENROUTER_API_KEY: 'k', OUROBOROS_PROCESSING_PREFERENCE: 'fast', OUROBOROS_MODEL: 'anthropic/claude-opus-5.5' };
    const tier = (patch) => pickerDocument(settings, row(patch)).OUROBOROS_PROCESSING_PREFERENCE;
    assert.equal(tier({ processing_preference: 'economy' }), 'economy', 'a Fast or Economy call: the editor names no catalog price');
    assert.equal(tier({ processing_preference: 'standard' }), 'standard', 'the call pays Standard although the default is Fast');
    assert.equal(tier({ processing_preference: '' }), '', 'the route default, as captured');
    assert.equal(tier({}), 'fast', 'a row from before the field takes the default tier');
    const loaded = pickerDocument(settings, row({ processing_preference: 'standard' }));
    assert.equal(loaded.OPENROUTER_API_KEY, 'k');
    assert.equal(loaded.model, row().model);
    assert.deepEqual(loaded.OUROBOROS_MODEL_ACCOUNTS, { main: 'personal' });
    assert.equal('OUROBOROS_MODEL' in loaded, false, 'no other setting reaches the draft');
});

test('Sign in again opens the provider page that holds the sign-in card; Settings opens Accounts', async (t) => {
    const env = installDom();
    t.after(() => restoreDom(env.prior));
    const root = document.createElement('div');
    const timelineEl = document.createElement('div');
    root.append(timelineEl);
    const opened = [];
    const logins = [];
    const controller = createModelWaitController({ getRecord: () => ({ root, timelineEl, finished: false }),
        openSettings: async (path) => { opened.push(path); return true; },
        login: async (harness, profile) => { logins.push([harness, profile]); } });
    t.after(() => controller.destroy());
    controller.observe('task', { model_waits: { 'wait-light': row({ reason: 'auth', worker_slot_held: false }) } });
    root.querySelector('[data-wait-login]').click();
    await new Promise((done) => setTimeout(done, 0));
    assert.deepEqual(opened, ['accounts/codex']);
    assert.deepEqual(logins, [['codex', 'personal']]);
    root.querySelector('[data-wait-settings]').click();
    assert.deepEqual(opened, ['accounts/codex', 'accounts'], 'Settings opens the list, not a provider');
    controller.observe('task', { model_waits: { 'wait-light': row({ reason: 'auth', revision: 2, credential_profile_id: '' }) } });
    root.querySelector('[data-wait-login]').click();
    await new Promise((done) => setTimeout(done, 0));
    assert.equal(opened.at(-1), 'accounts', 'without an exact account there is no provider page to sign in on');
    assert.equal(logins.length, 1);
});
