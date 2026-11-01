import assert from 'node:assert/strict';
import test from 'node:test';
import { renderMarkdown } from '../modules/utils.js';
import { prettyLogEvent, summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';
import { renderLiveCardMeta, withTaskCostMeta } from '../modules/chat_activity.js';

const priorDocument = globalThis.document;
globalThis.document = {
    createElement() {
        return {
            set textContent(value) {
                this.innerHTML = String(value ?? '').replaceAll('&', '&amp;')
                    .replaceAll('<', '&lt;').replaceAll('>', '&gt;');
            },
        };
    },
};
test.after(() => { globalThis.document = priorDocument; });

test('timeline markdown preserves inline labels with an explicit paragraph break', () => {
    for (let level = 1; level <= 6; level += 1) {
        const source = `${'#'.repeat(level)} Heading\n\nParagraph text`;
        const heading = `<strong class="md-h${Math.min(level, 3)}">Heading</strong>`;
        assert.equal(renderMarkdown(source, { inlineHeadingBreaks: true }), `${heading}<br>\n\nParagraph text`);
        assert.equal(renderMarkdown(source), `${heading}\n\nParagraph text`, 'other renderer consumers keep their existing output');
    }
    assert.equal(renderMarkdown('## Label\r\nParagraph', { inlineHeadingBreaks: true }),
        '<strong class="md-h2">Label</strong><br>\r\nParagraph');
    assert.equal(renderMarkdown('## Label', { inlineHeadingBreaks: true }),
        '<strong class="md-h2">Label</strong>');
});

test('inline heading breaks use the existing heading classifier and preserve ordinary rendering', () => {
    const label = 'x'.repeat(78);
    assert.equal(renderMarkdown(`## **${label}**\nParagraph`, { inlineHeadingBreaks: true }),
        `<strong class="md-h2"><strong>${label}</strong></strong><br>\nParagraph`);
    const paragraph = 'x'.repeat(81);
    assert.equal(renderMarkdown(`## ${paragraph}\nNext`, { inlineHeadingBreaks: true }), `${paragraph}\nNext`);
    for (const source of ['ordinary\nparagraph', '```sh\n# comment\nfalse\n```', '<b>literal</b>']) {
        assert.equal(renderMarkdown(source, { inlineHeadingBreaks: true }), renderMarkdown(source));
    }
});

test('Logs and Chat narration share the heading preview without changing the source', () => {
    const source = 'Intro\n## Summary\n\nParagraph text';
    const event = { type: 'send_message', task_id: 'task-1', is_progress: true, content: source };
    assert.equal(summarizeLogEvent(event).headline, 'Intro Summary — Paragraph text');
    assert.equal(summarizeChatLiveEvent(event).fullHeadline, source);
    const child = { ...event, delegation_role: 'subagent', subagent_task_id: 'child-1',
        parent_task_id: 'task-1', subagent_role: 'Researcher', subagent_event: 'running' };
    assert.equal(summarizeLogEvent(child).body, 'Intro Summary — Paragraph text');
    assert.equal(summarizeChatLiveEvent(child).fullBody, source);
    assert.equal(event.content, source);
    assert.equal(child.content, source);
});

test('shell, error and trace previews retain their literal hash lines', () => {
    const error = { type: 'llm_round_error', task_id: 'task-1', error: '# diagnostic\nconnection closed' };
    assert.equal(summarizeLogEvent(error).body, '# diagnostic connection closed');
    assert.equal(summarizeChatLiveEvent(error).body, '# diagnostic connection closed');
    assert.equal(summarizeChatLiveEvent(error).fullBody, error.error);
    const command = { type: 'tool_call_finished', task_id: 'task-1', tool: 'shell', is_error: true,
        status: 'non_zero_exit', exit_code: 1, args: { cmd: '# explanation\nfalse' },
        result_preview: '# failure\nexit 1' };
    assert.equal(summarizeLogEvent(command).body, '# failure exit 1');
    assert.equal(summarizeChatLiveEvent(command).body, 'Command: # explanation false # failure exit 1');
    const trace = { type: 'send_message', task_id: 'child-1', is_progress: true,
        delegation_role: 'subagent', parent_task_id: 'task-1', subagent_event: 'completed',
        trace_summary: '# trace\nstep ended' };
    assert.equal(summarizeChatLiveEvent(trace).body, '# trace step ended');
    assert.equal(summarizeChatLiveEvent(trace).fullBody, '[TRACE]\n# trace\nstep ended');
});

test('compact Chat removes internal write/status metadata while preserving executor and result facts', () => {
    const event = {
        type: 'send_message', is_progress: true, task_id: 'child-1', delegation_role: 'subagent',
        parent_task_id: 'task-1', subagent_role: 'Researcher', subagent_event: 'completed',
        status: 'completed', write_surface: 'workspace', model: 'openai/coordinator-model',
        result: 'Finished the requested analysis', executor_route: 'cursor=executor-model',
        execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 1,
            delegated_runs_succeeded: 1, delegated_runs_failed: 0, subscription_cost_usd: 0.4 },
        model_execution: { source: 'usable_solve_response', requested_model: 'openai/coordinator-model',
            used_model: 'openai/coordinator-model', used_local: false, requested_use_local: false },
        cost_accounting_status: 'ok', cost_final: true, accounted_upper_bound_usd: 3,
    };
    const before = JSON.stringify(event);
    const view = withTaskCostMeta(summarizeChatLiveEvent(event), event);
    assert.deepEqual(view.meta || [], []);
    assert.equal(view.phase, 'done');
    assert.equal(view.terminal, true);
    assert.equal(view.headline, 'Researcher');
    assert.equal(view.body, event.result);
    // A completed child's result is its report to its parent: the label replaces [RESULT] and leads the preview.
    assert.equal(view.fullBody, `## Report to parent\n${event.result}`);
    assert.equal(view.activityLead, 'Report to parent');
    assert.equal(view.model, event.model);
    assert.equal(view.executorChip.label, 'Cursor');
    assert.ok(summarizeLogEvent(event).meta.includes('write=workspace'));
    assert.equal(prettyLogEvent(event), JSON.stringify(event, null, 2));
    assert.equal(JSON.stringify(event), before);
    // The card's own details block holds the labelled facts beyond the compact line.
    const factsEl = { dataset: { liveFacts: '' }, innerHTML: '', hidden: true };
    const details = { classList: { contains: (name) => name === 'chat-live-details' }, children: [factsEl] };
    const record = { groupId: 'child-1', isSubagent: true, metaEl: { innerHTML: '', isConnected: true }, root: { children: [details] },
        executorChip: view.executorChip, agentModel: view.model, modelExecution: view.modelExecution,
        _lastFrameMeta: view.meta, costMeta: view.costProjection, latestActivityTs: '12:34:56' };
    renderLiveCardMeta(record);
    for (const fact of ['Cursor', '$3', 'updated 12:34:56']) {
        assert.ok(record.metaEl.innerHTML.includes(fact), fact);
    }
    // The coordinator and its solve evidence are the details' rows, never the executor's compact facts.
    assert.doesNotMatch(record.metaEl.innerHTML, /coordinator-model/);
    for (const fact of ['<dt>Coordinator</dt><dd>coordinator-model</dd>', '<dt>Last solve response</dt><dd>coordinator-model</dd>']) {
        assert.ok(factsEl.innerHTML.includes(fact), fact);
    }
    assert.equal(factsEl.hidden, false);
    assert.doesNotMatch(record.metaEl.innerHTML + factsEl.innerHTML, /write=workspace|status=completed/);
    record.costMeta = withTaskCostMeta(view, { cost_accounting_status: 'unavailable' }).costProjection;
    renderLiveCardMeta(record);
    assert.match(record.metaEl.innerHTML, /cost unavailable/);
});
