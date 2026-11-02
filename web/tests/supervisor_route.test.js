import assert from 'node:assert/strict';
import test from 'node:test';

import { executorIdentityMarkup, formatCoordinatorProvenance } from '../modules/harness_presentation.js';
import { executorChip } from '../modules/log_events.js';

const snapshot = {
    schema: 2, model: 'claudexor::codex=cognitive-model', effort: 'high',
    use_local_model: false, source_role: 'subagent:cognition',
    credential_profile_id: 'saved-account', processing_preference: 'economy', chosen_by: 'delegator',
    selected_supervisor_snapshot: {
        selected_subagent_id: 'cognition', route: { kind: 'api_model', target_id: 'claudexor::codex=cognitive-model' },
    },
};

test('schema-2 scheduling snapshot feeds the existing coordinator card and chip', (t) => {
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ set textContent(value) {
        this.innerHTML = String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    } }) };
    t.after(() => { globalThis.document = priorDocument; });
    const event = { subagent_task_id: 'child', executor_route: 'codex', effective_executor: 'harness',
        model: snapshot.model, parent_cognitive_route: snapshot, coordinator_billing: 'subscription' };
    const chip = executorChip(event);
    assert.equal(chip.coordinatorProvenance, 'chosen by the delegator (cognition)');
    assert.match(chip.title, /this subagent itself runs on claudexor::codex=cognitive-model \(bills as subscription\)/);
    assert.equal(formatCoordinatorProvenance(event), 'chosen by the delegator (cognition)');
    assert.doesNotMatch(chip.title, /saved-account|processing_preference/);
    assert.doesNotMatch(executorIdentityMarkup(chip, { model: event.model }), /Coordinator|delegator/);
});

test('recorded fallback and inherited choices stay distinct even with the same model', () => {
    for (const [chosen_by, source_role, expected] of [
        ['inherited', 'light', 'inherited'],
        ['parent_fallback', 'fallback:1', "parent's fallback"],
    ]) {
        const { selected_supervisor_snapshot: _selected, ...route } = snapshot;
        assert.equal(formatCoordinatorProvenance({ parent_cognitive_route: { ...route, chosen_by, source_role } }), expected);
    }
    assert.equal(formatCoordinatorProvenance({ model: snapshot.model }), '');
});

test('chosen row names reach the renderer verbatim; the Coordinator row escapes them (effort_chip tests)', () => {
    const route = { ...snapshot, selected_supervisor_snapshot: { selected_subagent_id: '<chosen&row>' } };
    assert.equal(formatCoordinatorProvenance({ parent_cognitive_route: route }), 'chosen by the delegator (<chosen&row>)');
});
