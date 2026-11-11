import assert from 'node:assert/strict';
import test from 'node:test';

import { noteToolCall, toolEvidenceView } from '../modules/chat_activity.js';
import { summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';

// #1316: a durable tool_call_started row (a Logs backfill) records that host
// processing began; replayed alone it must not read as a live "Running" call.
test('a tool start reads as started, its wait end as a timeout, its result as a result', () => {
    const base = { task_id: 't', tool: 'run_command', invocation_id: 'i1', args: { cmd: 'make' } };
    const started = summarizeLogEvent({ ...base, type: 'tool_call_started', timeout_sec: 60 });
    assert.equal(started.headline, 'Started run_command');
    assert.doesNotMatch(started.headline, /Running/);
    assert.equal(summarizeLogEvent({ ...base, type: 'tool_call_timeout', timeout_sec: 60 }).headline,
        'run_command timed out');
    assert.equal(summarizeLogEvent({ ...base, type: 'tool_call', result_preview: 'ok' }).headline,
        'run_command result');
});

// #1316: providers reuse their call ids across rounds (GigaChat always says "call_0"),
// so the live row key is the host's invocation id; a legacy frame keeps the provider id.
test('two invocations sharing a provider id stay two rows and never inherit a status', () => {
    const frame = (type, row) => summarizeChatLiveEvent({ type, task_id: 't', tool: 'read_file', tool_call_id: 'call_0', ...row });
    const first = [frame('tool_call_started', { invocation_id: 'i1' }),
        frame('tool_call_finished', { invocation_id: 'i1', is_error: true, error: 'boom' })];
    const second = [frame('tool_call_started', { invocation_id: 'i2' }), frame('tool_call_finished', { invocation_id: 'i2' })];
    assert.notEqual(first[0].toolCall.key, second[0].toolCall.key);
    assert.equal(first[1].dedupeKey, 'tool:t:i1', 'the failure row is keyed by its own invocation');
    const record = {};
    for (const view of [...first, ...second]) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status), ['error', 'ok']);
    const row = toolEvidenceView(record.toolFold);
    assert.deepEqual([row.calls, row.errors], [2, 1]);
    const legacy = frame('tool_call_finished', { is_error: true, error: 'boom' });
    assert.notEqual(legacy.dedupeKey, frame('tool_call_finished', { is_error: true }).dedupeKey,
        'legacy observations cannot be joined by reused provider ids');
});


test('independent wait end and durable settlement converge in both orders', () => {
    const row = type => summarizeChatLiveEvent({ type, task_id: 't', tool: 'read_file', invocation_id: 'i' });
    for (const order of [['tool_call_timeout', 'tool_call'], ['tool_call', 'tool_call_timeout']]) {
        const record = {};
        for (const type of order) noteToolCall(record, row(type).toolCall);
        record.toolFold.host = { calls: 1, errors: 1 };
        const view = toolEvidenceView(record.toolFold);
        assert.equal(view.errors, 0);
        assert.match(view.headline, /wait ended/);
    }
});

test('historical starts are unknown and event aliases retain their actual phase', () => {
    const record = {};
    noteToolCall(record, summarizeChatLiveEvent({ event: 'tool_call_started', task_id: 't', invocation_id: 'i' }).toolCall);
    assert.match(toolEvidenceView(record.toolFold).headline, /outcome unknown/);
    noteToolCall(record, summarizeChatLiveEvent({ type: 'tool_call', task_id: 't', invocation_id: 'i' }).toolCall);
    assert.doesNotMatch(toolEvidenceView(record.toolFold).headline, /unknown|error/);
});

test('a call Safety held reads as held outside the error phase, and the block counts no error for it', () => {
    const frame = (status, invocation_id) => summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: 't',
        tool: 'run_command', invocation_id, is_error: true, status });
    const views = [frame('safety_returned', 'i1'), frame('safety_unavailable', 'i2'), frame('error', 'i3')];
    assert.deepEqual(views.map((view) => [view.phase, view.headline]), [
        ['warn', 'A command was returned by the Safety Supervisor'],
        ['warn', 'A command was not run: the Safety Supervisor could not read it'],
        ['error', 'One of the steps failed · run_command']]);
    const record = {};
    for (const view of views) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status), ['held', 'held', 'error']);
    assert.deepEqual([toolEvidenceView(record.toolFold).calls, toolEvidenceView(record.toolFold).errors], [3, 1]);
    const logged = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true, status: 'safety_unavailable' });
    assert.deepEqual([logged.phase, logged.headline], ['warn', 'A command was not run: the Safety Supervisor could not read it']);
});
