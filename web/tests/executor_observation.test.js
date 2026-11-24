import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { executorChip, keepStickyExecutorChip } from '../modules/executor_card_facts.js';

const event = (revision, extras = {}) => ({
    type: 'send_message', is_progress: true, task_id: 'child', subagent_task_id: 'child',
    delegation_role: 'subagent', subagent_role: 'UI reviewer', model: 'openai/gpt-6-astra',
    content: 'Checking the layout', ts: '2026-09-09T10:00:00Z',
    executor_observation: { task_id: 'child', task_attempt: '1', run_id: 'run-a',
        attempt_id: 'a01', harness_id: 'cursor', phase: 'harness.event', revision,
        model: 'cursor-grok-4.6-xhigh-fast', model_source: 'requested' },
    ...extras,
});

test('live executor is the typed actor and requested model stays distinct from the coordinating model', () => {
    const view = summarizeChatLiveEvent(event(5));
    assert.equal(view.headline, 'UI reviewer');
    // #360 A: the label keeps only the observation word; the requested model and
    // its source move to the detail sentence the expanded card prints as text.
    assert.equal(view.executorChip.label, 'Cursor · last update');
    assert.match(view.executorChip.title, /model cursor-grok-4\.6-xhigh-fast \(requested\)/);
    assert.match(view.executorChip.title, /progress, not a terminal execution receipt/);
    assert.doesNotMatch(view.executorChip.label, /gpt-6-astra|running|settled/);
    assert.doesNotMatch(view.executorChip.title, /gpt-6-astra/);
    assert.equal(view.executorChip.hasEvidence, false);
});

test('root progress and usage retain their own model for the coordinator line', () => {
    const root = summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, task_id: 'root',
        content: 'Inspecting the UI', model: 'openai/gpt-6-astra',
    });
    assert.equal(root.model, 'openai/gpt-6-astra');
    const usage = summarizeChatLiveEvent({
        type: 'llm_usage', task_id: 'root', model: 'openai/gpt-6-astra', round: 2,
    });
    assert.equal(usage.model, 'openai/gpt-6-astra');
});

test('absent serving model remains unconfirmed and foreign-task observations are ignored', () => {
    const e = event(5);
    delete e.executor_observation.model;
    assert.equal(executorChip(e).label, 'Cursor · last update');
    assert.match(executorChip(e).title, /model unconfirmed/);
    e.executor_observation.task_id = 'sibling';
    assert.equal(executorChip(e), null);
});

test('helper usage cannot replace the task model; a turn\'s own round reports it', () => {
    for (const helper of [
        {model_category: 'websearch'},
        {model_category: 'vision'},
        {source: 'loop', category: 'compaction'},
        // A wake-up is an ordinary turn: its category alone proves nothing.
        {category: 'consciousness'},
    ]) {
        const view = summarizeChatLiveEvent({type: 'llm_usage', task_id: 'root', model: 'helper', ...helper});
        assert.equal(view.model, undefined);
    }
    const wake = summarizeChatLiveEvent({type: 'llm_usage', task_id: 'wake-1', round: 1,
        initiator: 'consciousness', category: 'consciousness', model: 'wake-model'});
    assert.equal(wake.model, 'wake-model');
});

test('same-run stale revisions and older cross-run frames cannot replace newer observed identity', () => {
    const later = executorChip(event(9));
    assert.equal(keepStickyExecutorChip(later, executorChip(event(8))), true);
    const old = event(99, {ts:'2026-09-09T09:00:00Z'});
    old.executor_observation.run_id = 'old-run';
    assert.equal(keepStickyExecutorChip(later, executorChip(old)), true);
    const newer = event(1, {ts:'2026-09-09T11:00:00Z'});
    newer.executor_observation.run_id = 'next-run';
    newer.executor_observation.task_attempt = '2';
    assert.equal(keepStickyExecutorChip(later, executorChip(newer)), false);
});

test('ordinary coordinator progress cannot downgrade observed identity to a dispatched label', () => {
    assert.equal(keepStickyExecutorChip(executorChip(event(9)), executorChip({executor_route:'cursor'})), true);
});

test('a changed dispatch route cannot downgrade a typed observation', () => {
    assert.equal(keepStickyExecutorChip(executorChip(event(9)), executorChip({executor_route:'claude'})), true);
});

test('terminal receipt retains distinct observed models without inventing a current or requested model', () => {
    const terminal = executorChip({executor_route:'cursor', ts:'2026-09-09T11:00:00Z', execution_evidence:{
        delegated_runs_started:2,delegated_runs_settled:2,delegated_runs_failed:0,
        harness_models:['Grok 4.6','Fable 5.1','Grok 4.6'],
    }});
    assert.equal(terminal.hasEvidence,true);
    assert.equal(terminal.settledRunModel, '');
    assert.match(terminal.title, /models observed across settled runs: Grok 4\.6, Fable 5\.1$/);
    assert.equal(keepStickyExecutorChip(executorChip(event(9)),terminal),false);
    assert.equal(keepStickyExecutorChip(terminal,executorChip(event(9))),true);
});
