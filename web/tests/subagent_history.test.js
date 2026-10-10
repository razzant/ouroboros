import test from 'node:test';
import assert from 'node:assert/strict';
import { rowTaskRun } from '../modules/subagent_status_primitives.js';

test('missing requested options are not a proved change of settings; known empty remains comparable', () => {
    const row = { subagent_id: 'worker', route: { kind: 'agent_session', target_id: 'codex=model' },
        access: 'full', effort: '', processing_preference: '' };
    const receipt = { selected_subagent_id: 'worker', route: 'codex', requested_model: 'model',
        applied_model: 'served-model', applied_profile: 'observed-account',
        identity: { kind: 'agent_session', target_id: 'codex=model', credential_profile_id: '', access: 'full', effort: '' } };
    const state = { snapshot: { subagent_last_delegation: receipt } };
    const unknown = rowTaskRun(row, state);
    assert.match(unknown, /^Settings not fully reported · /);
    assert.doesNotMatch(unknown, /Earlier settings/);
    assert.match(unknown, /account observed-account/);
    receipt.identity.processing_preference = '';
    assert.doesNotMatch(rowTaskRun(row, state), /settings/i);
    assert.match(rowTaskRun({ ...row, effort: 'high' }, state), /^Earlier settings · /);
});

// DECISIONS v3 §5: a receipt's identity.effort is the CONFIGURED row effort (the pin, or '' for
// Auto); the level the child actually ran at travels separately as the effort fact. History
// compares configured with configured, so an Auto row that ran at a chosen level is not
// "Earlier settings", while a changed pin is.
test('an Auto row that ran at a chosen level is the current settings; a changed pin is earlier settings', () => {
    const receipt = (effort) => ({ selected_subagent_id: 'worker', route: 'api_model', requested_model: 'model',
        applied_model: 'served-model', identity: { kind: 'api_model', target_id: 'openai::model', credential_profile_id: '',
            effort, processing_preference: '' } });
    const auto = { subagent_id: 'worker', route: { kind: 'api_model', target_id: 'openai::model' }, processing_preference: '' };
    // Auto row, leaf ran at High (the effort fact, not the identity): no "Earlier settings".
    const ranAtHigh = { snapshot: { subagent_last_delegation: { ...receipt(''), effort_level: 'high', effort_source: 'auto' } } };
    assert.doesNotMatch(rowTaskRun(auto, ranAtHigh), /Earlier settings/);
    assert.match(rowTaskRun(auto, ranAtHigh), /^API model · served-model/);
    // The same receipt after the owner pinned the row: the pin differs from the configured '' of the run.
    assert.match(rowTaskRun({ ...auto, effort: 'high' }, ranAtHigh), /^Earlier settings · /);
    // Cyber Pro: a pinned High row whose leaf ran at Ultra, the row still pinned High: current settings.
    const cyber = { snapshot: { subagent_last_delegation: { ...receipt('high'), effort_level: 'ultra', effort_requested: 'ultra', effort_source: 'cyber' } } };
    assert.doesNotMatch(rowTaskRun({ ...auto, effort: 'high' }, cyber), /Earlier settings/);
});
