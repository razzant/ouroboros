import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { coordinatorRouteClause, formatCoordinatorProvenance, executorIdentityMarkup } from '../modules/harness_presentation.js';
import { executorChip } from '../modules/log_events.js';

// Python checks these card facts against both subagent_message_meta and the history
// endpoint without a task result, including the absent-facts replay below.
const cases = JSON.parse(readFileSync(new URL('../../tests/fixtures/subagent_card_facts.json', import.meta.url), 'utf8'));
const coordinatorFields = ['parent_cognitive_route', 'coordinator_route', 'coordinator_billing', 'coordinator_provenance', 'use_local_model'];
for (const row of cases) {
    for (const mode of ['producer', 'recorded history', 'history without coordinator facts']) {
        test(`${mode}: ${row.name}`, (t) => {
            const priorDocument = globalThis.document;
            globalThis.document = { createElement: () => ({ set textContent(value) {
                this.innerHTML = String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
            } }) };
            t.after(() => { globalThis.document = priorDocument; });
            const frame = { task_id: 'child', subagent_task_id: 'child', delegation_role: 'subagent', subagent_event: 'running', ...row.facts };
            if (mode !== 'producer') frame.is_progress = true;
            const absent = mode === 'history without coordinator facts';
            if (absent) for (const field of coordinatorFields) delete frame[field];
            const provenance = absent ? '' : row.provenance;
            const clause = absent ? `this subagent itself runs on ${frame.model} (bills as unknown)` : row.clause;
            const chip = executorChip(frame);
            assert.equal(formatCoordinatorProvenance(frame), provenance);
            assert.equal(coordinatorRouteClause(frame), clause);
            assert.ok(chip.title.includes(clause));
            assert.equal(chip.coordinatorProvenance || '', provenance);
            // The chip line carries the executor only; the coordinator and its recorded provenance are a
            // details row (chat_activity.renderLiveCardMeta), so an absent record shows no provenance at all.
            assert.doesNotMatch(executorIdentityMarkup(chip, { model: 'exec-model' }), /Coordinator|inherited|fallback|delegator/);
            if (absent) assert.equal(chip.coordinatorProvenance || '', '');
        });
    }
}

test('the chip renders supplied billing and never prices or reclassifies the route', () => {
    for (const model of ['claudexor::codex=gpt-5', 'local-model (local)', 'openai/gpt-5']) {
        assert.match(coordinatorRouteClause({ model, cost_usd: .5 }), /bills as unknown/);
        assert.match(coordinatorRouteClause({ model, coordinator_billing: 'measured' }), /bills as measured/);
    }
    assert.equal(formatCoordinatorProvenance({ model: 'm', effective_model_lane: 'fallback' }), '');
    for (const snap of [{ model: 'm' }, { model: 'm', source_role: 'fallback' },
        { model: 'm', selected_supervisor_snapshot: { selected_subagent_id: 'row' } }]) {
        assert.equal(formatCoordinatorProvenance({ parent_cognitive_route: snap }), '');
        assert.equal(formatCoordinatorProvenance({ parent_cognitive_route: { ...snap, chosen_by: 'inherited' } }), 'inherited');
        assert.equal(formatCoordinatorProvenance({ parent_cognitive_route: { ...snap, chosen_by: 'parent_fallback' } }), "parent's fallback");
    }
});
