import assert from 'node:assert/strict';
import test from 'node:test';
import { createModelRolesEditor } from '../modules/model_roles.js';
import { PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY, processingSelectHtml,
    processingIntentLabel, processingExecutionText, processingCapabilityNote, processingDetailsHtml } from '../modules/route_editor_primitives.js';
import { buildAvailableSubagentsSetting, parseAvailableSubagentsSetting, availableSubagentRowMarkup,
    lastReviewRunText } from '../modules/subagents_settings.js';
import { onboardingSettingsDraft } from '../modules/onboarding_agents_step.js';

const contract = { modelSlots: [
    { slot: 'main', label: 'Main', settingKey: 'OUROBOROS_MODEL', inputId: 'main' },
    { slot: 'light', label: 'Light', settingKey: 'OUROBOROS_MODEL_LIGHT', inputId: 'light' },
    { slot: 'fallback', label: 'Fallback', settingKey: 'OUROBOROS_MODEL_FALLBACKS', inputId: 'fallback' },
] };

test('role processing survives save/reopen independently of identical model names and native slugs', () => {
    const settings = {
        OUROBOROS_MODEL: 'openai::same', OUROBOROS_MODEL_LIGHT: 'openai::same',
        OUROBOROS_MODEL_FALLBACKS: 'claudexor::codex=gpt-high-fast, openai::same',
        [PROCESSING_PREFERENCE_KEY]: 'fast',
        [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'standard', light: '', fallback: ['economy', ''] },
    };
    for (let pass = 0; pass < 2; pass += 1) {
        const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
        editor.load(settings, contract);
        assert.equal(editor.validate(), '');
        assert.deepEqual(editor.collect(), settings);
        editor.adoptCatalog({ read_state: 'failed', items: [], errors: [{ error: 'offline' }] });
        assert.deepEqual(editor.collect(), settings);
        editor.destroy();
    }
});

test('untouched legacy Models and waiting model/account pickers do not author Processing', () => {
    const settings = { OUROBOROS_MODEL: 'claudexor::codex=gpt-high-fast' };
    for (const showContext of [true, false]) {
        const editor = createModelRolesEditor({ hostId: 'test', doc: () => null, showContext });
        editor.load(showContext ? settings : { ...settings, [PROCESSING_PREFERENCE_KEY]: 'fast',
            [MODEL_PROCESSING_PREFERENCES_KEY]: { main: 'economy' } }, contract);
        const saved = editor.collect();
        assert.equal(saved.OUROBOROS_MODEL, settings.OUROBOROS_MODEL);
        assert.ok(!(PROCESSING_PREFERENCE_KEY in saved));
        assert.ok(!(MODEL_PROCESSING_PREFERENCES_KEY in saved));
        editor.destroy();
    }
});

test('native Processing choices and collapsed overrides keep inheritance distinct from Standard', () => {
    const inherited = processingSelectHtml('aria-label="Processing"', '', { global: true });
    assert.match(inherited, /value="" selected>Keep route defaults/);
    assert.doesNotMatch(inherited, /value="standard" selected/);
    assert.match(processingSelectHtml('', 'standard'), /value="standard" selected/);
    assert.match(processingIntentLabel('', 'fast'), /Fast.*from Models/);
    assert.match(processingIntentLabel('standard', 'fast'), /Standard.*override/);
    assert.match(processingIntentLabel('', ''), /Route default.*inherited/);
    assert.match(processingDetailsHtml('data-test', '', 'fast'), /<details[^>]*><summary>/);
    assert.doesNotMatch(processingDetailsHtml('data-test', 'fast'), /<details[^>]*\bopen\b|disabled/);
});

test('actor overrides are optional, canonical and preserved on the existing route', () => {
    for (const preference of ['', 'standard', 'fast', 'economy']) {
        const actor = { subagent_id: 'worker', recommended_use: 'Do the work',
            route: { kind: 'agent_session', target_id: 'cursor=gpt-high-fast', credential_profile_id: 'owner' },
            effort: 'high', processing_preference: preference };
        const parsed = parseAvailableSubagentsSetting({ enabled: true, items: [actor] });
        assert.equal(parsed.error, '');
        const saved = buildAvailableSubagentsSetting(parsed.setting).items[0];
        assert.deepEqual(saved.route, actor.route);
        assert.equal(saved.effort, actor.effort);
        assert.equal(saved.processing_preference, preference || undefined);
        const markup = availableSubagentRowMarkup(saved, { processingPreference: 'fast' });
        assert.match(markup, /data-subagent-field="processing_preference"/);
        assert.match(markup, /An explicit native service choice takes precedence/);
    }
    const invalid = parseAvailableSubagentsSetting({ enabled: true, items: [{ subagent_id: 'worker',
        recommended_use: '', route: { kind: 'api_model', target_id: 'openai::x' }, processing_preference: 'turbo' }] });
    assert.match(invalid.error, /processing/);
});

test('a reviewer is a catalog row: its processing override and its mark persist together', () => {
    const row = { subagent_id: 'worker', name: 'Worker', recommended_use: '',
        route: { kind: 'api_model', target_id: 'openai::same' }, processing_preference: 'standard', review_eligible: true };
    const saved = buildAvailableSubagentsSetting({ enabled: true, items: [row] });
    assert.equal(saved.items[0].processing_preference, 'standard');
    assert.equal(saved.items[0].review_eligible, true);
    const reopened = parseAvailableSubagentsSetting(saved);
    assert.equal(reopened.error, '');
    assert.deepEqual(buildAvailableSubagentsSetting(reopened.setting), saved);
    const { processing_preference: _dropped, ...inherited } = row;
    assert.ok(!('processing_preference' in buildAvailableSubagentsSetting({ enabled: true, items: [inherited] }).items[0]));
});

test('only existing execution receipts disclose applied modes; unsupported speed stays editable', () => {
    assert.equal(processingExecutionText(), '');
    assert.equal(processingExecutionText({ requested: 'fast' }), '');
    const receipt = { requested: 'fast', submitted: 'standard', submittedNative: 'auto', observed: 'unknown',
        observedNative: [], reason: 'Processing preference unsupported', source: 'native' };
    assert.match(processingExecutionText(receipt), /Applied processing not reported.*requested Fast.*submitted service auto/);
    assert.doesNotMatch(processingExecutionText(receipt), /Applied processing: Fast/);
    assert.match(lastReviewRunText({ effective: { model: 'x', processing: { ...receipt, observed: 'mixed' } } }), /Applied processing: Mixed/);
    assert.equal(processingCapabilityNote('fast', undefined), '');
    assert.match(processingCapabilityNote('fast', { modes: ['standard'] }), /not advertised.*Ordinary service may be used/);
    assert.doesNotMatch(processingSelectHtml('', 'fast'), /disabled/);
});

test('onboarding preview and Finish carry the same optional global/role processing values', () => {
    const baseline = onboardingSettingsDraft({ state: {} });
    assert.ok(!(PROCESSING_PREFERENCE_KEY in baseline));
    const state = { processingPreference: 'fast', modelProcessingPreferences: { main: 'standard', fallback: ['', 'economy'] } };
    const draft = onboardingSettingsDraft({ state });
    assert.equal(draft[PROCESSING_PREFERENCE_KEY], 'fast');
    assert.deepEqual(draft[MODEL_PROCESSING_PREFERENCES_KEY], state.modelProcessingPreferences);
    assert.equal(onboardingSettingsDraft({ state: { processingPreference: '' } })[PROCESSING_PREFERENCE_KEY], '');
});
